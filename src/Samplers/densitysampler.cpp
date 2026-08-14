#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "densitysampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

DensitySampler::DensitySampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins
) : Sampler(numberOfParticles, numberOfDimensions, numberOfParameters, stepLength, numberOfMetropolisSteps),
m_rMax(rMax), m_nBins(nBins) {

    m_dr = m_rMax / m_nBins;

    m_histogram.assign(m_nBins, 0.0);
    m_histFlavor.resize(m_nBins);
    m_densityProfile.assign(m_nBins, 0.0);
    m_densityProfileFlavor.resize(m_nBins);
    m_densityError.assign(m_nBins, 0.0);
    m_densityErrorFlavor.resize(m_nBins);
    m_probabilityProfile.assign(m_nBins, 0.0);
    m_probabilityProfileFlavor.resize(m_nBins);
    m_probabilityError.assign(m_nBins, 0.0);
    m_probabilityErrorFlavor.resize(m_nBins);
    m_rGrid.assign(m_nBins, 0.0);

    for (unsigned int i = 0; i < m_nBins; i++) {
        m_rGrid[i] = (i + 0.5) * m_dr;
        m_histFlavor[i].assign(N_FLAVORS, 0.0);
        m_densityProfileFlavor[i].assign(N_FLAVORS, 0.0);
        m_densityErrorFlavor[i].assign(N_FLAVORS, 0.0);
        m_probabilityProfileFlavor[i].assign(N_FLAVORS, 0.0);
        m_probabilityErrorFlavor[i].assign(N_FLAVORS, 0.0);
    }
}

void DensitySampler::sample(bool acceptedStep, System* system, std::vector<double>*) {
    const std::vector<std::unique_ptr<Particle>>& particles = system->getParticles();

    // radial distance and fill histogram
    for (unsigned int p = 0; p < m_numberOfParticles; p++) {
        double dist = norm(particles[p]->getPosition());
        Flavor flav = particles[p]->getFlavor();

        unsigned int bin = dist / m_dr;
        if (bin < m_nBins) {
            m_histogram[bin] += 1.0;
            m_histFlavor[bin][flav] += 1.0;
        }
    }

    m_numberOfAcceptedSteps += acceptedStep;
    m_watch_end = std::chrono::high_resolution_clock::now();
}

void DensitySampler::computeAverages() {
    double half_d = 0.5 * m_numberOfDimensions;
    double volume_coeff = pow(M_PI, half_d) / tgamma(half_d + 1.0);

    for (unsigned int i = 0; i < m_nBins; i++) {
        double r_inner = i * m_dr;
        double r_outer = (i + 1) * m_dr;
        // volume
        double volume = volume_coeff * (pow(r_outer, m_numberOfDimensions)
            - pow(r_inner, m_numberOfDimensions));

        if (volume > 0) {
            // 1: density averaged over radius
            // not dividing by the number of particles, accounting for volume
            double normalization = m_numberOfMetropolisSteps * volume;
            // Density = bin_counts / normalization
            m_densityProfile[i] = m_histogram[i] / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_densityProfileFlavor[i][flav] = m_histFlavor[i][flav] / normalization;
            }
            // Poisson error estimate 
            m_densityError[i] = sqrt(m_histogram[i]) / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_densityErrorFlavor[i][flav] = sqrt(m_histFlavor[i][flav]) / normalization;
            }
            // 2: radial probability
            // not dividing by the number of particles, nor by the volume
            normalization = m_numberOfMetropolisSteps * (r_outer - r_inner);
            // Probability = bin_counts / normalization 
            m_probabilityProfile[i] = m_histogram[i] / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_probabilityProfileFlavor[i][flav] = m_histFlavor[i][flav] / normalization;
            }
            // Poisson error estimate 
            m_probabilityError[i] = sqrt(m_histogram[i]) / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_probabilityErrorFlavor[i][flav] = sqrt(m_histFlavor[i][flav]) / normalization;
            }
        }
    }
    m_elapsedTime = m_watch_end - m_watch_start;
}