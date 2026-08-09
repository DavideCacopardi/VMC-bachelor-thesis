#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "energyekinsampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/lennardjonesho.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/ljgaussian.h"

using namespace CommonUtils;

EnergyEkinSampler::EnergyEkinSampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps
) : EnergySampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps
) {}

EnergyEkinSampler::EnergyEkinSampler(const std::vector<std::unique_ptr<EnergyEkinSampler>>& others)
    : EnergySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        others[0]->m_numberOfParameters,
        others[0]->m_stepLength,
        0) {
    for (unsigned int i = 0; i < others.size(); i++) {
        this->mergeBaseData(others[i].get());
        m_cumulativeEkin1 += others[i]->m_cumulativeEkin1;
        m_cumulativeEkin2 += others[i]->m_cumulativeEkin2;
    }
    computeAverages();
}

void EnergyEkinSampler::sample(bool acceptedStep, System* system, std::vector<double>* energiesOut) {
    EnergySampler::sample(acceptedStep, system, energiesOut);

    auto* ljHam = dynamic_cast<LennardJonesHO*>(&system->getHamiltonian());
    if (ljHam == nullptr) {
        throw std::invalid_argument("ERR: EnergyEkinSampler requires an LJHamiltonian!");
    }
    m_cumulativeEkin1 += ljHam->computeLocalKineticEnergy(system->getWaveFunction(), system->getParticles(), 1);
    m_cumulativeEkin2 += ljHam->computeLocalKineticEnergy(system->getWaveFunction(), system->getParticles(), 2);
}

void EnergyEkinSampler::logHeader(const std::vector<double>& params, std::ofstream& outs) {
    outs << "#";
    const unsigned int width = 20;
    for (unsigned int i = 0; i < params.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "],";
        outs << std::setw(width - (i == 0)) << temp;
    }
    outs << std::setw(width) << "energy," << std::setw(width) << "variance,"
        << std::setw(width) << "error," << std::setw(width) << "    -(ℏ/4)⟨∇ᵢ²ln(ψ)⟩,"
        << std::setw(width) << "   (ℏ/2)⟨|∇ᵢln(ψ)|²⟩," << std::setw(width) << "elapsed time,"
        << std::setw(width) << "accept. ratio,"
        << std::setw(width) << "Monte Carlo steps " << std::endl;
}

void EnergyEkinSampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    const unsigned int prec = 12, width = 19;
    outs << std::scientific << std::setprecision(prec);
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << std::setw(width) << params[i] << ",";
    }
    outs << std::scientific << std::setprecision(prec)
        << std::setw(width) << m_energy << ","
        << std::setw(width) << m_variance << ","
        << std::setw(width) << m_error << ","
        << std::setw(width) << m_Ekin1 << ","
        << std::setw(width) << m_Ekin2 << ","
        << std::setw(width) << m_elapsedTime.count() << ","
        << std::setw(width) << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << ","
        << std::setw(width) << m_numberOfMetropolisSteps
        << std::endl;
}

void EnergyEkinSampler::computeAverages() {
    EnergySampler::computeAverages();
    m_Ekin1 = m_cumulativeEkin1 / m_numberOfMetropolisSteps;
    m_Ekin2 = m_cumulativeEkin2 / m_numberOfMetropolisSteps;
}