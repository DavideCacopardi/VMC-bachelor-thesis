#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "harmonicoscillator.h"
#include "Solvers/metropolishastings.h"

VMCOptimizer::VMCOptimizer(
    MCEngine& engine,
    unsigned int numberOfMetropolisSteps,
    double BFGS_tol,
    double BFGS_VarOpt_weight,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : m_engine(engine),
m_numberOfMetropolisSteps(numberOfMetropolisSteps),
m_BFGS_tol(BFGS_tol),
m_BFGS_VarOpt_weight(BFGS_VarOpt_weight),
m_logfile(logfile),
m_outfile(outfile),
m_paramsfile(paramsfile) {}

double VMCOptimizer::computeMC(const std::vector<double>& params, std::vector<double>& grad) {
    std::cout << "\rComputing MC #" << m_mcCount + 1 << "      (nMCsteps: "
        << m_numberOfMetropolisSteps << ")              " << std::flush;

    auto sampler = m_engine.run(params, m_numberOfMetropolisSteps);

    if (m_logfile) {
        if (m_mcCount == 0) {
            sampler->logHeader(params, *m_logfile);
        }
        sampler->logOutput(params, *m_logfile);
    }
    double energy = sampler->getEnergy();
    double variance = sampler->getVariance();
    double error = sampler->getError();

    if (!grad.empty()) {
        std::vector<double> dEdW = sampler->get_dEdW();
        std::vector<double> dVardW = sampler->get_dVardW();
        for (unsigned int i = 0; i < grad.size(); i++) {
            grad[i] = dEdW[i] + m_BFGS_VarOpt_weight * dVardW[i];
        }
    }
    double objectiveValue = energy + m_BFGS_VarOpt_weight * variance;
    if (objectiveValue < m_bestObjective) {
        m_bestObjective = objectiveValue;
        m_bestParams = params; // deep copy of the best vector found so far
    }

    // Signal-to-Noise Ratio increment check (every c_wait cycles to avoid forced variance decrease)
    if (m_mcCount % c_wait == 1 && std::abs(objectiveValue - m_previousObjVal) < 8 * error) {
        m_numberOfMetropolisSteps *= 1.2; // HARD-CODED!
    }
    m_previousObjVal = objectiveValue;
    m_mcCount++;

    return objectiveValue;
}

std::vector<double> VMCOptimizer::optimize(std::vector<double> params, const std::vector<bool>& optimize_mask) {
    // nlopt setup
    nlopt::opt lib_optimizer(nlopt::LD_LBFGS, params.size());
    {
        auto tempWaveFunction = m_engine.makeWaveFunction(params);
        auto lb = tempWaveFunction->lowerBounds();
        auto ub = tempWaveFunction->upperBounds();
        // If the wavefunction didn't supply bounds
        if (lb.empty())
            lb.assign(params.size(), -std::numeric_limits<double>::infinity());
        if (ub.empty())
            ub.assign(params.size(), std::numeric_limits<double>::infinity());
        if (optimize_mask.size() == params.size()) {
            for (unsigned int i = 0; i < params.size(); ++i) {
                // If the mask dictates this parameter should not be optimized, freeze it
                if (!optimize_mask[i]) {
                    lb[i] = params[i];
                    ub[i] = params[i];
                }
            }
        }

        lib_optimizer.set_lower_bounds(lb);
        lib_optimizer.set_upper_bounds(ub);
    }
    lib_optimizer.set_min_objective(nloptObjective, this);
    lib_optimizer.set_xtol_rel(m_BFGS_tol);
    lib_optimizer.set_maxeval(400);
    lib_optimizer.set_maxtime(10800.0);
    m_mcCount = 0;

    // run the actual optimization
    double minEnergy;
    try {
        nlopt::result res = lib_optimizer.optimize(params, minEnergy);
        std::cout << "\nNLopt returned " << res << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "\nNLopt failed: " << e.what() << std::endl;
        std::cout << "Last energy: " << minEnergy << std::endl;
        std::cout << "Last params: ";
        for (auto p : params) std::cout << p << " ";
        if (m_outfile) {
            *m_outfile << "# NLopt failed: " << e.what() << std::endl;
            *m_outfile << "# Last energy: " << minEnergy << std::endl;
        }
    }
    std::cout << std::endl;

    if (m_outfile) {
        *m_outfile << "# Optimal parameters: " << std::setprecision(8);
        for (auto p : m_bestParams) *m_outfile << p << ", \t";
        *m_outfile << std::endl;
    }
    if (m_paramsfile) {
        for (auto p : m_bestParams) *m_paramsfile << p << std::endl;
    }

    return m_bestParams;
}

double VMCOptimizer::nloptObjective(
    const std::vector<double>& params, std::vector<double>& grad, void* data
) {
    return static_cast<VMCOptimizer*>(data)->computeMC(params, grad);
}