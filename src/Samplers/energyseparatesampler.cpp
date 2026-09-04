#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "energyseparatesampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/lennardjonesho.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/ljgaussian.h"

using namespace CommonUtils;

EnergySeparateSampler::EnergySeparateSampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    bool log_grads
) : EnergySampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps,
    log_grads
) {}

EnergySeparateSampler::EnergySeparateSampler(const std::vector<std::unique_ptr<EnergySeparateSampler>>& others)
    : EnergySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        others[0]->m_numberOfParameters,
        others[0]->m_stepLength,
        0,
        others[0]->m_log_grads)
{
    initVectors(others[0]->m_cumulativeEnergies.size());
    m_vectorsAreInitialized = true;
    for (unsigned int i = 0; i < others.size(); i++) {
        this->mergeBaseData(others[i].get());
        for (unsigned int j = 0; j < others[i]->m_cumulativeEnergies.size(); j++) {
            m_cumulativeEnergies[j] += others[i]->m_cumulativeEnergies[j];
            m_cumulativeEnergiesSQ[j] += others[i]->m_cumulativeEnergiesSQ[j];
        }
    }
    computeAverages();
}

std::unique_ptr<EnergySampler> EnergySeparateSampler::constructMergedSampler(
    std::vector<std::unique_ptr<EnergySampler>>& others
) {
    std::vector<std::unique_ptr<EnergySeparateSampler>> castedPtrs;
    castedPtrs.reserve(others.size());
    for (auto& o : others) {
        // release ownership from the base unique_ptr, reclaim as derived
        EnergySeparateSampler* derivedRaw = dynamic_cast<EnergySeparateSampler*>(o.get());
        if (!derivedRaw) {
            throw std::runtime_error("Expected EnergyEkinSampler in merge list");
        }
        o.release();  // base unique_ptr no longer owns it
        castedPtrs.emplace_back(derivedRaw);  // derived unique_ptr now owns it
    }
    return std::make_unique<EnergySeparateSampler>(castedPtrs);
}

void EnergySeparateSampler::sample(bool acceptedStep, System* system, std::vector<double>* energiesOut) {
    std::vector<double> localEnergies = system->computeLocalEnergies();
    if (!m_vectorsAreInitialized) {
        initVectors(localEnergies.size());
        m_vectorsAreInitialized = true;
    }
    for (unsigned int i = 0; i < localEnergies.size(); i++) {
        m_cumulativeEnergies[i] += localEnergies[i];
        m_cumulativeEnergiesSQ[i] += sq(localEnergies[i]);
    }

    // the rest follows identically to energysampler
    double localEnergy = localEnergies[0];
    if (energiesOut != nullptr) {
        energiesOut->push_back(localEnergy);
    }
    m_cumulativeEnergy += localEnergy;
    m_cumulativeEnergySQ += sq(localEnergy);    // to later evaluate ΔE

    // calculate <O> and <O E> to later evaluate Cov(O, E) = <O E> - <O><E>
    std::vector<double> OW = system->getWaveFunction().paramGradientLnAbs(system->getParticles());
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        m_cumulativeOpO[i] += OW[i];
        m_cumulativeOpOE[i] += OW[i] * localEnergy;
        m_cumulativeOpOE2[i] += OW[i] * sq(localEnergy);
    }

    m_numberOfAcceptedSteps += acceptedStep;
    m_watch_end = std::chrono::high_resolution_clock::now();
}

void EnergySeparateSampler::logHeader(const std::vector<double>& params, std::ofstream& outs) {
    for (unsigned int i = 0; i < params.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "]";
        print_colTitle(outs, temp, i == 0, false);
    }
    print_colTitle(outs, "energy");
    print_colTitle(outs, "variance");
    print_colTitle(outs, "error");
    for (unsigned int i = 0; i < m_energies.size(); i++) {
        std::string temp = "energies[" + std::to_string(i) + "]";
        print_colTitle(outs, temp);
        temp = "energiesVar[" + std::to_string(i) + "]";
        print_colTitle(outs, temp);
    }
    if (m_log_grads) {
        for (unsigned int i = 0; i < params.size(); i++) {
            std::string temp = "gradE[" + std::to_string(i) + "]";
            print_colTitle(outs, temp);
            temp = "gradVarE[" + std::to_string(i) + "]";
            print_colTitle(outs, temp);
        }
    }
    print_colTitle(outs, "elapsed time");
    print_colTitle(outs, "accept. ratio");
    print_colTitle(outs, "MC steps", false, true);
}

void EnergySeparateSampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    outs << std::scientific;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        print_colVal(outs, params[i], i==0, false);
    }
    print_colVal(outs, m_energy);
    print_colVal(outs, m_variance);
    print_colVal(outs, m_error);
    for (unsigned int i = 0; i < m_energies.size(); i++) {
        print_colVal(outs, m_energies[i]);
        print_colVal(outs, m_energies_variance[i]);
    }
    if (m_log_grads) {
        auto gradE = get_dEdW();
        auto gradVarE = get_dVardW();
        for (unsigned int i = 0; i < params.size(); i++) {
            print_colVal(outs, gradE[i]);
            print_colVal(outs, gradVarE[i]);
        }
    }
    print_colVal(outs, m_elapsedTime.count());
    print_colVal(outs, ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps));
    print_colVal(outs, m_numberOfMetropolisSteps, false, true);
    outs << std::flush;
}

void EnergySeparateSampler::computeAverages() {
    EnergySampler::computeAverages();
    for (unsigned int i = 0; i < m_energies.size(); i++) {
        m_energies[i] = m_cumulativeEnergies[i] / (double)m_numberOfMetropolisSteps;
        m_energiesSQ[i] = m_cumulativeEnergiesSQ[i] / (double)m_numberOfMetropolisSteps;
        m_energies_variance[i] = m_energiesSQ[i] - sq(m_energies[i]);
    }
}

void EnergySeparateSampler::initVectors(unsigned int size) {
    m_cumulativeEnergies.resize(size, 0);
    m_cumulativeEnergiesSQ.resize(size, 0);
    m_energies.resize(size, 0);
    m_energiesSQ.resize(size, 0);
    m_energies_variance.resize(size, 0);
}

void EnergySeparateSampler::printSpecial(std::ostream& outs) {
    outs << "Printing specials:\n" << std::scientific;
    for (unsigned int i = 0; i < m_energies.size(); i++) {
        std::string temp = "energies[" + std::to_string(i) + "]: ";
        outs << temp;
        print_colVal(outs, m_energies[i], false, true);
        temp = "energies_var[" + std::to_string(i) + "]: ";
        outs << temp;
        print_colVal(outs, m_energies_variance[i], false, true);
    }
    outs << std::defaultfloat;
}