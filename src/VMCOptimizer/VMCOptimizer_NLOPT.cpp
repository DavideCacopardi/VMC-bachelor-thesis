#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer_NLOPT.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "harmonicoscillator.h"
#include "Solvers/metropolishastings.h"

VMCOptimizer_NLOPT::VMCOptimizer_NLOPT(
     const runConfig& cfg,
    MCEngine& engine,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : VMCOptimizer(cfg, engine, logfile, outfile, paramsfile) {}

std::vector<double> VMCOptimizer_NLOPT::optimize(std::vector<double> params, const std::vector<bool>& optimize_mask) {
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
    lib_optimizer.set_xtol_rel(m_cfg.BFGS_tol);
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
        std::cout << "\nNLopt threw an exception: " << e.what() << std::endl;
        std::cout << "  WRN: Falling back to the lowest energy parameters observed (m_bestParams).\n"
                    " NOTE: These parameters might be subject to statistical noise outliers.\n";
        
        params = m_bestParams;
        minEnergy = m_bestObjective;
        
        if (m_outfile) {
            *m_outfile << "# NLopt failed. Using fallback m_bestParams." << std::endl;
        }
    }
    std::cout << std::endl;

    if (m_outfile) {
        *m_outfile << "# Optimal parameters: " << std::setprecision(8);
        for (auto p : params) *m_outfile << p << ",  ";
        *m_outfile << std::endl;
    }
    if (m_paramsfile) {
        for (auto p : params) *m_paramsfile << p << std::endl;
    }

    return params;
}

double VMCOptimizer_NLOPT::nloptObjective(
    const std::vector<double>& params, std::vector<double>& grad, void* data
) {
    return static_cast<VMCOptimizer_NLOPT*>(data)->computeMC(params, grad);
}