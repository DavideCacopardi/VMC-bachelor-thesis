#include <iostream>
#include <memory>
#include <cassert>

#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/energyekinsampler.h"
#include "Samplers/densitysampler.h"
#include "Samplers/NNsampler.h"
#include "Particles/particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/ljgaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/lennardjonesho.h"
#include "InitialStates/initialstate.h"
#include "Solvers/montecarlo.h"


System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles
) {
    m_numberOfParticles = particles.size();;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
}

System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction
) {
    m_numberOfParticles = 0;
    m_numberOfDimensions = 0;
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
}


double System::runEquilibrationSteps(double stepParameter,
    unsigned int numberOfEquilibrationSteps) {

    const double target = m_solver->get_target_acceptanceRatio();
    const unsigned int tuneBlockSize = 100; 

    const unsigned int blocks = numberOfEquilibrationSteps / tuneBlockSize;
    for (unsigned int b = 0; b < blocks; ++b) {
        unsigned int acceptedSteps = 0;
        
        for (unsigned int i = 0; i < tuneBlockSize; ++i) {
            acceptedSteps += m_solver->step(stepParameter, *m_waveFunction, m_particles);
        }

        double acceptanceRatio = static_cast<double>(acceptedSteps) / tuneBlockSize;

        // tune stepParameter
        if (acceptanceRatio < target - 0.05) {
            stepParameter *= 0.9; // shrink step to increase acceptance
        }
        else if (acceptanceRatio > target + 0.05) {
            stepParameter *= 1.1; // grow step to decrease acceptance
        }
    }

    // leftover steps without tuning
    const unsigned int remainingSteps = numberOfEquilibrationSteps % tuneBlockSize;
    for (unsigned int i = 0; i < remainingSteps; ++i) {
        m_solver->step(stepParameter, *m_waveFunction, m_particles);
    }

    return stepParameter;
}

std::unique_ptr<EnergySampler> System::runMetropolisSteps(double stepParameter,
    unsigned int numberOfMetropolisSteps, std::vector<double>* energiesOut, bool request_Ekin)
{
    std::unique_ptr<EnergySampler> sampler;
    auto* ptr_h = dynamic_cast<LennardJonesHO*>(&getHamiltonian());
    auto* ptr_wf = dynamic_cast<LJGaussian*>(&getWaveFunction());
    if (request_Ekin && ptr_h != nullptr && ptr_wf != nullptr) {
        sampler = std::make_unique<EnergyEkinSampler>(
            m_numberOfParticles,
            m_numberOfDimensions,
            m_waveFunction->getNumberOfParameters(),
            stepParameter,
            numberOfMetropolisSteps);
    }
    else {
        sampler = std::make_unique<EnergySampler>(
            m_numberOfParticles,
            m_numberOfDimensions,
            m_waveFunction->getNumberOfParameters(),
            stepParameter,
            numberOfMetropolisSteps);
    }

    auto watch_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);
        sampler->sample(acceptedStep, this, energiesOut);
    }
    auto watch_end = std::chrono::high_resolution_clock::now();

    sampler->computeAverages();
    sampler->setElapsedTime(watch_end - watch_start);

    return sampler;
}

std::unique_ptr<NNsampler> System::runMetropolisSteps_NN_pretrain(double stepParameter,
    unsigned int numberOfMetropolisSteps, WaveFunction& wf_train) {
    std::unique_ptr<NNsampler> sampler = std::make_unique<NNsampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps,
        wf_train);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        bool acceptedStep = m_solver->step(stepParameter, wf_train, m_particles);
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();

    return sampler;
}

std::unique_ptr<DensitySampler> System::runMetropolisStepsOnebodyDensity(double stepParameter,
    unsigned int numberOfMetropolisSteps, double rMax, unsigned int nBins, unsigned int numberOfParticleLogs, std::ofstream* particlesOut) {
    auto sampler = std::make_unique<DensitySampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps,
        rMax,
        nBins);

    if (particlesOut && numberOfMetropolisSteps / numberOfParticleLogs > 1) {
        sampler->logParticlesHeader(*particlesOut);
    }
    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

        sampler->sample(acceptedStep, this);

        if (particlesOut && ((i + 1) % (numberOfMetropolisSteps / numberOfParticleLogs) == 0)){
            sampler->logParticles(getParticles(), *particlesOut);
        }
    }

    sampler->computeAverages();

    return sampler;
}

const std::vector<double>& System::getWaveFunctionParameters() {
    // Helper function
    return m_waveFunction->getParameters();
}

Hamiltonian& System::getHamiltonian() {
    return *m_hamiltonian;
}

WaveFunction& System::getWaveFunction() {
    return *m_waveFunction;
}

void System::setParticles(std::vector<std::unique_ptr<Particle>> new_particles) {
    m_particles = std::move(new_particles);
    m_numberOfParticles = m_particles.size();
    m_numberOfDimensions = m_particles[0]->getNumberOfDimensions();
}

void System::setSolver(std::unique_ptr<MonteCarlo> new_solver) {
    m_solver = std::move(new_solver);
}

void System::setHamiltonian(std::unique_ptr<Hamiltonian> new_hamiltonian) {
    m_hamiltonian = std::move(new_hamiltonian);
}

std::unique_ptr<WaveFunction> System::setWaveFunction(std::unique_ptr<WaveFunction> new_waveFunction) {
    auto temp = std::move(m_waveFunction);
    m_waveFunction = std::move(new_waveFunction);
    return temp;
}
