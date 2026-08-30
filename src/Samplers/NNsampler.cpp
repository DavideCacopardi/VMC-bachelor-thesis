#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "system.h"
#include "NNsampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/wavefunction.h"
#include "common.h"

using namespace CommonUtils;

NNsampler::NNsampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    WaveFunction& wf_train
) : Sampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps),
    m_wf_train(wf_train) {
    m_cumulativeOW.assign(m_numberOfParameters, 0);
    m_OW.assign(m_numberOfParameters, 0);
    m_cumulativeBOW.assign(m_numberOfParameters, 0);
    m_BOW.assign(m_numberOfParameters, 0);
    m_cumulativeB2OW.assign(m_numberOfParameters, 0);
    m_B2OW.assign(m_numberOfParameters, 0);
}

void NNsampler::sample(bool acceptedStep, System* system, std::vector<double>*) {
    auto* ptr = dynamic_cast<NN_envelope*>(&system->getWaveFunction());
    if (ptr == nullptr) {
        throw std::logic_error("NNsampler requires a NN_envelope wave function.");
    }

    // auto OW = system->computeLogParDer_vect();
    auto OW = system->getWaveFunction().paramGradientLnAbs(system->getParticles());

    auto& particles = system->getParticles();
    double psi = system->getWaveFunction().eval(particles); // NN
    double psi_train = m_wf_train.eval(particles);          // Gaussian

    double B = psi / (psi_train + c_eps);
    B = std::max(1e-10, std::min(B, 1e10));

    m_cumulativeB += B;
    m_cumulativeB2 += sq(B);

    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        m_cumulativeBOW[i] += B * OW[i];
        m_cumulativeB2OW[i] += sq(B) * OW[i];
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
    m_watch_end = std::chrono::high_resolution_clock::now();
}

void NNsampler::computeAverages() {
    m_acceptanceRatio = m_numberOfAcceptedSteps / (double)m_stepNumber;
    m_B = m_cumulativeB / (double)m_stepNumber;
    m_B2 = m_cumulativeB2 / (double)m_stepNumber;
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        m_OW[i] = m_cumulativeOW[i] / (double)m_stepNumber;
        m_BOW[i] = m_cumulativeBOW[i] / (double)m_stepNumber;
        m_B2OW[i] = m_cumulativeB2OW[i] / (double)m_stepNumber;
    }
    m_K = sq(m_B) / (m_B2 + c_eps);
    m_elapsedTime = m_watch_end - m_watch_start;
}



double NNsampler::get_K() const { return m_K; }
double NNsampler::get_Kerr() const { return m_Kerr; }
double NNsampler::getAcceptanceRatio() const { return m_acceptanceRatio; }

void NNsampler::logOutput(std::ofstream& outs) {
    outs << std::scientific;
    print_colVal(outs, m_K);
    print_colVal(outs, m_elapsedTime.count());
    print_colVal(outs, getAcceptanceRatio());
    outs << std::flush;
}

std::vector<double> NNsampler::get_dKdW() const {
    std::vector<double> dKdW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dKdW[i] = 2 * m_K * (m_BOW[i] / (m_B + c_eps) - m_B2OW[i] / (m_B2 + c_eps));
    }
    return dKdW;
}