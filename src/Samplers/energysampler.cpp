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
    unsigned int numberOfMetropolisSteps
) : Sampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps) {
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
}

EnergySampler::EnergySampler(const std::vector<std::unique_ptr<EnergySampler>>& others)
    : EnergySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        others[0]->m_numberOfParameters,
        others[0]->m_stepLength,
        0) {
    for (unsigned int i = 0; i < others.size(); i++) {
        mergeBaseData(others[i].get());
    }
    computeAverages();
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

void EnergySampler::printOutputToTerminal(System& system) {
    std::cout << std::endl;
    std::cout << "  -- System info -- " << std::endl;
    std::cout << " Number of particles  : " << m_numberOfParticles << std::endl;
    std::cout << " Number of dimensions : " << m_numberOfDimensions << std::endl;
    std::cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << std::endl;
    std::cout << " Step length used : " << m_stepLength << std::endl;
    std::cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << std::endl;
    std::cout << " Elapsed time: " << m_elapsedTime.count() << " s\n";
    std::cout << std::endl;
    std::cout << "  -- Wave function parameters -- " << std::endl;
    std::cout << " Number of parameters : " << m_numberOfParameters << std::endl;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        std::cout << " Parameter " << i + 1 << " : " << system.getWaveFunctionParameters()[i] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "  -- Results -- " << std::endl;
    std::cout << " Energy : " << m_energy << std::endl;
    std::cout << " Variance : " << m_variance << std::endl;
    std::cout << " Error : " << m_error << std::endl;
    std::cout << std::endl;
}

void EnergySampler::printOutputToFile(System& system, std::ofstream& outs) {
    outs << std::endl;
    outs << "#  -- System info -- " << std::endl;
    outs << "# Number of particles  : " << m_numberOfParticles << std::endl;
    outs << "# Number of dimensions : " << m_numberOfDimensions << std::endl;
    outs << "# Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << std::endl;
    outs << "# Step length used : " << m_stepLength << std::endl;
    outs << "# Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << std::endl;
    outs << "# Elapsed time: " << m_elapsedTime.count() << " s\n";
    outs << std::endl;
    outs << "#  -- Wave function parameters -- " << std::endl;
    outs << "# Number of parameters : " << m_numberOfParameters << "\n#";
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << " p[" << i << "],  \t ";
    }
    outs << " energy,  \t  variance,  \t  error\n";
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << system.getWaveFunctionParameters()[i] << ", \t";
    }
    outs << std::setprecision(10);
    outs << m_energy << ", \t" << m_variance << ", \t" << m_error << std::endl;
}

void EnergySampler::logHeader(const std::vector<double>& params, std::ofstream& outs) {
    outs << "#";
    const unsigned int width = 20;
    for (unsigned int i = 0; i < params.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "],";
        outs << std::setw(width - (i == 0)) << temp;
    }
    outs << std::setw(width) << "energy," << std::setw(width) << "variance,"
        << std::setw(width) << "error," << std::setw(width) << "elapsed time,"
        << std::setw(width) << "accept. ratio,"
        << std::setw(width) << "Monte Carlo steps " << std::endl;
}

void EnergySampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    const unsigned int prec = 12, width = 19;
    outs << std::scientific << std::setprecision(prec);
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << std::setw(width) << params[i] << ",";
    }
    outs << std::scientific << std::setprecision(prec)
        << std::setw(width) << m_energy << ","
        << std::setw(width) << m_variance << ","
        << std::setw(width) << m_error << ","
        << std::setw(width) << m_elapsedTime.count() << ","
        << std::setw(width) << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << ","
        << std::setw(width) << m_numberOfMetropolisSteps
        << std::endl;
}

void EnergySampler::logOutput(std::ofstream& outs, std::vector<double> additional_log) {
    const unsigned int prec = 12, width = 19;
    outs << std::scientific << std::setprecision(prec)
        << std::setw(width) << m_energy << ","
        << std::setw(width) << m_variance << ","
        << std::setw(width) << m_error << ",";
    for (int i = 0; i < (int)additional_log.size(); i++)
        outs << std::setw(width) << additional_log[i] << ",";
    outs << std::setw(width) << m_elapsedTime.count() << ","
        << std::setw(width) << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << ","
        << std::setw(width) << m_numberOfMetropolisSteps
        << std::endl;
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