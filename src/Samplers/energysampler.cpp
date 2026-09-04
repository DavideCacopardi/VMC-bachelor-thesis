#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "energysampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

EnergySampler::EnergySampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    bool log_grads
) : Sampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps)
{
    m_log_grads = log_grads;
    reset();
}

EnergySampler::EnergySampler(const std::vector<std::unique_ptr<EnergySampler>>& others)
    : EnergySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        others[0]->m_numberOfParameters,
        others[0]->m_stepLength,
        0,
        others[0]->m_log_grads) 
{
    for (unsigned int i = 0; i < others.size(); i++) {
        mergeBaseData(others[i].get());
    }
    computeAverages();
}

void EnergySampler::reset() {
    m_covarianceE.resize(m_numberOfParameters, 0);
    m_covarianceE2.resize(m_numberOfParameters, 0);
    m_energy = 0;
    m_energySQ = 0;
    m_variance = 0;
    m_error = 0;
    m_cumulativeEnergy = 0;
    m_cumulativeEnergySQ = 0;
    m_cumulativeOpO.resize(m_numberOfParameters, 0);
    m_cumulativeOpOE.resize(m_numberOfParameters, 0);
    m_cumulativeOpOE2.resize(m_numberOfParameters, 0);
    m_cumulativeEdEdW.resize(m_numberOfParameters, 0);
}

void EnergySampler::mergeBaseData(const EnergySampler* other) {
    if (other->m_elapsedTime > m_elapsedTime)
        m_elapsedTime = other->m_elapsedTime;
    m_cumulativeEnergy += other->m_cumulativeEnergy;
    m_cumulativeEnergySQ += other->m_cumulativeEnergySQ;
    m_numberOfMetropolisSteps += other->m_numberOfMetropolisSteps;
    m_numberOfAcceptedSteps += other->m_numberOfAcceptedSteps;
    for (unsigned int j = 0; j < m_numberOfParameters; j++) {
        m_cumulativeOpO[j] += other->m_cumulativeOpO[j];
        m_cumulativeOpOE[j] += other->m_cumulativeOpOE[j];
        m_cumulativeOpOE2[j] += other->m_cumulativeOpOE2[j];
        m_cumulativeEdEdW[j] += other->m_cumulativeEdEdW[j];
    }
}

void EnergySampler::sample(bool acceptedStep, System* system, std::vector<double>* energiesOut) {
    double localEnergy = system->computeLocalEnergy();
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

void EnergySampler::logHeader(const std::vector<double>& params, std::ofstream& outs) {
    for (unsigned int i = 0; i < params.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "]";
        print_colTitle(outs, temp, i == 0, false);
    }
    print_colTitle(outs, "energy");
    print_colTitle(outs, "variance");
    print_colTitle(outs, "error");
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

void EnergySampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    outs << std::scientific;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        print_colVal(outs, params[i], i==0, false);
    }
    print_colVal(outs, m_energy);
    print_colVal(outs, m_variance);
    print_colVal(outs, m_error);
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

void EnergySampler::logOutput(std::ofstream& outs, std::vector<double> additional_log) {
    outs << std::scientific;
    print_colVal(outs, m_energy, true, false);
    print_colVal(outs, m_variance);
    print_colVal(outs, m_error);
    for (int i = 0; i < (int)additional_log.size(); i++)
        print_colVal(outs, additional_log[i]);
    print_colVal(outs, m_elapsedTime.count());
    print_colVal(outs, ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps));
    print_colVal(outs, m_numberOfMetropolisSteps, false, true);
    outs << std::flush;
}

void EnergySampler::computeAverages() {
    m_energy = m_cumulativeEnergy / (double)m_numberOfMetropolisSteps;
    m_energySQ = m_cumulativeEnergySQ / (double)m_numberOfMetropolisSteps;
    m_variance = m_energySQ - sq(m_energy);
    m_error = sqrt(m_variance / (double)m_numberOfMetropolisSteps);
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        double mean_O = m_cumulativeOpO[i] / (double)m_numberOfMetropolisSteps;
        double mean_OE = m_cumulativeOpOE[i] / (double)m_numberOfMetropolisSteps;
        double mean_OE2 = m_cumulativeOpOE2[i] / (double)m_numberOfMetropolisSteps;

        // <OE> - <O><E>
        m_covarianceE[i] = mean_OE - mean_O * m_energy;
        m_covarianceE2[i] = mean_OE2 - mean_O * m_energySQ;
    }
}

std::vector<double> EnergySampler::get_dEdW() const {
    std::vector<double> dEdW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dEdW[i] = 2 * m_covarianceE[i];
    }
    return dEdW;
}

std::vector<double> EnergySampler::get_dVardW() const {
    std::vector<double> dVardW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dVardW[i] = 2 * m_covarianceE2[i] - 4 * m_energy * m_covarianceE[i];
    }
    return dVardW;
}

void EnergySampler::setElapsedTime(std::chrono::duration<double> time) {
    m_elapsedTime = time;
}