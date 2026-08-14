#include <iostream>
#include <iomanip>
#include <nlopt.hpp>
#include <atomic>

#include "VMCOptimizer.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "harmonicoscillator.h"
#include "Solvers/metropolishastings.h"

extern std::atomic<bool> g_stop_optimization;

VMCOptimizer::VMCOptimizer(
    const runConfig& cfg,
    MCEngine& engine,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : m_cfg(cfg),
    m_engine(engine),
    m_logfile(logfile),
    m_outfile(outfile),
    m_paramsfile(paramsfile) {}

double VMCOptimizer::computeMC(const std::vector<double>& params, std::vector<double>& grad) {
    if (g_stop_optimization) {
        throw std::runtime_error("User requested graceful stop via Ctrl+C");
    }
    
    bool tryAgain;
    unsigned int count_improvement_tries = 0;
    double energy, variance, error, objectiveValue;
    std::unique_ptr<EnergySampler> sampler;

    do {
        tryAgain = false;
        std::cout << "\rComputing MC #" << m_mcCount + 1 << "      (nMCsteps: "
            << m_cfg.metropolisSteps << ")\033[K" << std::flush;

        sampler = m_engine.run(params, m_cfg.metropolisSteps);

        if (m_logfile) {
            if (m_mcCount == 0) {
                sampler->logHeader(params, *m_logfile);
            }
            sampler->logOutput(params, *m_logfile);
        }
        energy = sampler->getEnergy();
        variance = sampler->getVariance();
        error = sampler->getError();
        objectiveValue = energy + m_cfg.varOpt_weight * variance;

        // Signal-to-Noise Ratio increment check (every c_wait cycles to avoid forced variance decrease)
        if (m_mcCount % c_wait == 1
            && count_improvement_tries < c_max_improvement_tries
            && std::abs(objectiveValue - m_previousObjVal) < 4 * error
        ) {
            m_cfg.metropolisSteps *= 1.2; // HARD-CODED!
            tryAgain = true;
            count_improvement_tries++;
        }
    } while (tryAgain);

    m_mcCount++;
    m_previousObjVal = objectiveValue;
    if (!grad.empty()) {
        std::vector<double> dEdW = sampler->get_dEdW();
        std::vector<double> dVardW = sampler->get_dVardW();
        for (unsigned int i = 0; i < grad.size(); i++) {
            grad[i] = dEdW[i] + m_cfg.varOpt_weight * dVardW[i];
        }
    }
    if (objectiveValue < m_bestObjective) {
        m_bestObjective = objectiveValue;
        m_bestParams = params; // deep copy of the best vector found so far
    }

    return objectiveValue;
}