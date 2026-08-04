#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "mcengine.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/montecarlo.h"
#include "Hamiltonians/hamiltonian.h"

MCEngine::MCEngine(
    const runConfig& cfg,
    HamiltonianFactory hamiltonianFactory,
    WaveFunctionFactory waveFunctionFactory,
    SolverFactory solverFactory
) :
m_cfg(cfg),
m_hamiltonianFactory(std::move(hamiltonianFactory)),
m_waveFunctionFactory(std::move(waveFunctionFactory)),
m_solverFactory(std::move(solverFactory)) 
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
    std::ofstream* energiesOut) {
    auto rng = std::make_unique<Random>(m_cfg.seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_cfg.seed);
    auto particles = setupRandomUniformInitialState(
        m_cfg.numberOfDimensions, m_cfg.numberOfParticles, *rng, m_rep_a);
    auto solver = m_solverFactory(std::move(rng));
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        m_waveFunctionFactory(params),
        std::move(solver),
        std::move(particles));

    system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
    return system->runMetropolisSteps(m_cfg.timeStep, numberOfMetropolisSteps, energiesOut);
}

std::unique_ptr<DensitySampler> MCEngine::runOnebodyDensity(
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins) {
    auto rng = std::make_unique<Random>(m_cfg.seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_cfg.seed);
    auto particles = setupRandomUniformInitialState(
        m_cfg.numberOfDimensions, m_cfg.numberOfParticles, *rng, m_rep_a);
    auto solver = m_solverFactory(std::move(rng));
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        m_waveFunctionFactory(params),
        std::move(solver),
        std::move(particles));

    system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
    return system->runMetropolisStepsOnebodyDensity(m_cfg.timeStep, numberOfMetropolisSteps, rMax, nBins);
}

double MCEngine::getRepulsiveFactor() const {
    return m_rep_a;
}

std::unique_ptr<WaveFunction> MCEngine::makeWaveFunction(const std::vector<double>& params) const {
    return m_waveFunctionFactory(params);
}