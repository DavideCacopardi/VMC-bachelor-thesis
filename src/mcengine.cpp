#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include "mcengine.h"
#include "common.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/energyekinsampler.h"
#include "Samplers/energyseparatesampler.h"
#include "Samplers/densitysampler.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/montecarlo.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

MCEngine::MCEngine(
    const runConfig& cfg,
    HamiltonianFactory hamiltonianFactory,
    WaveFunctionFactory waveFunctionFactory,
    SolverFactory solverFactory,
    EnSamplerFactory enSamplerFactory
) :
m_cfg(cfg),
m_hamiltonianFactory(std::move(hamiltonianFactory)),
m_waveFunctionFactory(std::move(waveFunctionFactory)),
m_solverFactory(std::move(solverFactory)),
m_enSamplerFactory(std::move(enSamplerFactory)) 
{
    if (m_hamiltonianFactory()->has_hardcore()) {
        m_rep_a = m_hamiltonianFactory()->getRepulsiveFactor();
    }
    else {
        m_rep_a = 0;
    }
}

std::unique_ptr<EnergySampler> MCEngine::run(
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    std::vector<std::vector<double>>* energiesOut) 
{
    omp_set_num_threads(m_cfg.numberOfThreads);
    unsigned int localSteps = numberOfMetropolisSteps / m_cfg.numberOfThreads;

    // Vectors to hold the results from each thread
    std::vector<std::unique_ptr<EnergySampler>> local_samplers(m_cfg.numberOfThreads);
    std::vector<std::vector<double>> local_energies(m_cfg.numberOfThreads);
    if (energiesOut != nullptr) {
        energiesOut->resize(m_cfg.numberOfThreads);
        for (unsigned int i = 0; i < m_cfg.numberOfThreads; ++i) {
            (*energiesOut)[i].reserve(localSteps);
        }
    }

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // thread-safe RNG
        unsigned long base_seed = (m_cfg.seed == 0) 
            ? std::chrono::system_clock::now().time_since_epoch().count() 
            : m_cfg.seed;
        auto rng = std::make_unique<Random>(base_seed + thread_id);
        // thread-local physical system
        auto particles = setupRandomUniformInitialState(
            m_cfg.numberOfDimensions, m_cfg.numberOfParticles, *rng);
        auto solver = m_solverFactory(std::move(rng));
        auto system = std::make_unique<System>(
            m_hamiltonianFactory(),
            m_waveFunctionFactory(params),
            std::move(solver),
            std::move(particles));

        double tuned_timeStep = system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
        local_samplers[thread_id] = system->runMetropolisSteps(
            tuned_timeStep, 
            localSteps, 
            energiesOut != nullptr ? &((*energiesOut)[thread_id]) : nullptr,
            m_cfg.log_grads,
            m_cfg.LJ_request_Ekin,
            &m_enSamplerFactory
        );
    }

    std::unique_ptr<EnergySampler> global_sampler = local_samplers[0]->constructMergedSampler(local_samplers);
    return global_sampler;
}

std::unique_ptr<DensitySampler> MCEngine::runSpatial(
    const std::vector<double>& params,
    std::ofstream* particlesOut,
    bool normalizing_PCF)
{
    auto rng = std::make_unique<Random>(m_cfg.seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_cfg.seed);
    auto particles = setupRandomUniformInitialState(
        m_cfg.numberOfDimensions, m_cfg.numberOfParticles, *rng);
    auto solver = m_solverFactory(std::move(rng));
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        m_waveFunctionFactory(params),
        std::move(solver),
        std::move(particles));
        
    double tuned_timeStep = system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
    std::unique_ptr<DensitySampler> main_sampler = system->runMetropolisStepsSpatial(
        tuned_timeStep, m_cfg.onebodyDensitySteps, m_cfg.onebodyDensity_rMax, m_cfg.onebodyDensity_nBins,
        m_cfg.normalize_by_nParticles, m_cfg.nParticleLogs, particlesOut);

    if (!normalizing_PCF && m_cfg.calc_normalized_PCF && system->getWaveFunction().hasJastrow()) {
        std::cout << "\rComputing reference PCF...           " << std::flush;
        Random refRng(m_cfg.seed == 0
            ? std::chrono::system_clock::now().time_since_epoch().count()
            : m_cfg.seed + 987654321);  // distinct stream from the physical run
        main_sampler->computeUncorrelatedReference(m_cfg.uncorrRefDraws, refRng);
        main_sampler->normalizeAgainstUncorrelated();
    }
    
    // if (!normalizing_PCF && m_cfg.calc_normalized_PCF && system->getWaveFunction().hasJastrow()) {
    //     std::cout << "\rComputing reference PCF...           " << std::flush;
    //     std::unique_ptr<DensitySampler> noInt_sampler = runSpatial(m_cfg.referenceParams, nullptr, true);
    //     main_sampler->load_normalized_PCF(*noInt_sampler);
    // }
    return main_sampler;
}

double MCEngine::getRepulsiveFactor() const {
    return m_rep_a;
}

std::unique_ptr<WaveFunction> MCEngine::makeWaveFunction(const std::vector<double>& params) const {
    return m_waveFunctionFactory(params);
}