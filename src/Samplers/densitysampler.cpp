#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <format>
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
    unsigned int nBins,
    bool normalize_by_nParticles
) : Sampler(numberOfParticles,
        numberOfDimensions,
        numberOfParameters,
        stepLength,
        numberOfMetropolisSteps
    ),
    m_rMax(rMax),
    m_nBins(nBins),
    m_normalize_by_nParticles(normalize_by_nParticles)
{
    m_dr = m_rMax / m_nBins;

    m_histogram.assign(m_nBins, 0);
    m_histFlavor.resize(m_nBins);
    m_histAlike.assign(m_nBins, 0);
    m_histUnlike.assign(m_nBins, 0);

    m_rGrid.assign(m_nBins, 0.0);

    m_dens.assign(m_nBins, 0.0);
    m_dens_err.assign(m_nBins, 0.0);
    m_dens_f.resize(m_nBins);
    m_dens_err_f.resize(m_nBins);
    m_dens_alike.resize(m_nBins, 0.0);
    m_dens_err_alike.resize(m_nBins, 0.0);
    m_dens_unlike.resize(m_nBins, 0.0);
    m_dens_err_unlike.resize(m_nBins, 0.0);

    m_prob.assign(m_nBins, 0.0);
    m_prob_err.assign(m_nBins, 0.0);
    m_prob_f.resize(m_nBins);
    m_prob_err_f.resize(m_nBins);
    m_prob_alike.resize(m_nBins, 0.0);
    m_prob_err_alike.resize(m_nBins, 0.0);
    m_prob_unlike.resize(m_nBins, 0.0);
    m_prob_err_unlike.resize(m_nBins, 0.0);
    for (unsigned int i = 0; i < m_nBins; i++) {
        m_rGrid[i] = (i + 0.5) * m_dr;

        m_histFlavor[i].assign(N_FLAVORS, 0);
        m_dens_f[i].assign(N_FLAVORS, 0.0);
        m_dens_err_f[i].assign(N_FLAVORS, 0.0);
        m_prob_f[i].assign(N_FLAVORS, 0.0);
        m_prob_err_f[i].assign(N_FLAVORS, 0.0);
    }
}

void DensitySampler::sample(bool acceptedStep, System* system, std::vector<double>*) {
    const std::vector<std::unique_ptr<Particle>>& particles = system->getParticles();

    // One-Body
    for (unsigned int p = 0; p < m_numberOfParticles; p++) {
        double dist = norm(particles[p]->getPosition());

        unsigned int bin = dist / m_dr;
        if (bin < m_nBins) {
            m_histogram[bin]++;
            m_histFlavor[bin][particles[p]->getFlavor()]++;
        }
    }

    // Pair Correlation
    int countAlike = 0;
    int countUnlike = 0;
    for (unsigned int i = 0; i < m_numberOfParticles; i++) {
        for (unsigned int j = i + 1; j < m_numberOfParticles; j++) {
            double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
            bool is_alike = particles[i]->getFlavor() == particles[j]->getFlavor();
            if (is_alike) countAlike++;
            else countUnlike++;

            unsigned int bin = dist / m_dr;
            if (bin < m_nBins) {
                if (is_alike) {
                    m_histAlike[bin]++;
                }
                else {
                    m_histUnlike[bin]++;
                }
            }
        }
    }
    // check Alike and Unlike haven't changed in number to prevent normalization afterwards
    if (m_nAlike != -1 && (m_nAlike != countAlike || m_nUnlike != countUnlike)) {
        m_nAlike_nUnlike_haveChanged = true;
    }
    m_nAlike = countAlike;
    m_nUnlike = countUnlike;


    m_numberOfAcceptedSteps += acceptedStep;
    m_watch_end = std::chrono::high_resolution_clock::now();
}

void DensitySampler::computeAverages() {
    if (m_nAlike_nUnlike_haveChanged) {
        std::cout << " WRN: nAlike or nUnlike have changed; pair correlations won't be normalized\n";
    }

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
            // accounting for volume
            double normalization = m_numberOfMetropolisSteps * volume;
            if (m_normalize_by_nParticles) normalization *= (double)m_numberOfParticles;
            // Density = bin_counts / normalization
            m_dens[i] = m_histogram[i] / normalization;
            m_dens_err[i] = sqrt(m_histogram[i]) / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_dens_f[i][flav] = m_histFlavor[i][flav] / normalization;
                m_dens_err_f[i][flav] = sqrt(m_histFlavor[i][flav]) / normalization;
            }
            normalization = m_numberOfMetropolisSteps * volume;
            if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
                normalization *= (double)m_nAlike;
            m_dens_alike[i] = m_histAlike[i] / normalization;
            m_dens_err_alike[i] = sqrt(m_histAlike[i]) / normalization;
            normalization = m_numberOfMetropolisSteps * volume;
            if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
                normalization *= (double)m_nUnlike;
            m_dens_unlike[i] = m_histUnlike[i] / normalization;
            m_dens_err_unlike[i] = sqrt(m_histUnlike[i]) / normalization;

            // 2: radial probability
            // not dividing by the volume
            normalization = m_numberOfMetropolisSteps * (r_outer - r_inner);
            if (m_normalize_by_nParticles) normalization *= (double)m_numberOfParticles;
            // Probability = bin_counts / normalization 
            m_prob[i] = m_histogram[i] / normalization;
            m_prob_err[i] = sqrt(m_histogram[i]) / normalization;
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                m_prob_f[i][flav] = m_histFlavor[i][flav] / normalization;
                m_prob_err_f[i][flav] = sqrt(m_histFlavor[i][flav]) / normalization;
            }
            normalization = m_numberOfMetropolisSteps * (r_outer - r_inner);
            if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
                normalization *= (double)m_nAlike;
            m_prob_alike[i] = m_histAlike[i] / normalization;
            m_prob_err_alike[i] = sqrt(m_histAlike[i]) / normalization;
            normalization = m_numberOfMetropolisSteps * (r_outer - r_inner);
            if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
                normalization *= (double)m_nUnlike;
            m_prob_unlike[i] = m_histUnlike[i] / normalization;
            m_prob_err_unlike[i] = sqrt(m_histUnlike[i]) / normalization;
        }
    }
    m_elapsedTime = m_watch_end - m_watch_start;
}

void DensitySampler::logDensity(std::ofstream& outs) const {
    outs << std::scientific;
    for (unsigned int i = 0; i < m_nBins; i++) {
        // result[i] = std::make_pair(grid[i], densityProf[i]);
        print_colVal(outs, m_rGrid[i], true, false);

        print_colVal(outs, m_dens[i]);
        print_colVal(outs, m_dens_err[i]);
        if (N_FLAVORS > 1) {
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                print_colVal(outs, m_dens_f[i][flav]);
                print_colVal(outs, m_dens_err_f[i][flav]);
            }
            print_colVal(outs, m_dens_alike[i]); 
            print_colVal(outs, m_dens_err_alike[i]);
            print_colVal(outs, m_dens_unlike[i]); 
            print_colVal(outs, m_dens_err_unlike[i]);
        }
        print_colVal(outs, m_prob[i]);
        print_colVal(outs, m_prob_err[i]);
        if (N_FLAVORS > 1) {
            for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                print_colVal(outs, m_prob_f[i][flav]);
                print_colVal(outs, m_prob_err_f[i][flav]);
            }
            print_colVal(outs, m_prob_alike[i]); 
            print_colVal(outs, m_prob_err_alike[i]);
            print_colVal(outs, m_prob_unlike[i]); 
            print_colVal(outs, m_prob_err_unlike[i], false, true);
        }
        
    }
}

void DensitySampler::logDensityHeader(std::ofstream& outs) const {
    outs << '#';
    print_colTitle(outs, "r", true, false);
    print_colTitle(outs, "dens_tot");
    print_colTitle(outs, "dens_err");
    if (N_FLAVORS > 1) {
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            std::string f(1, (char)('A' + flav));
            print_colTitle(outs, "dens_" + f); 
            print_colTitle(outs, "dens_err_" + f);
        }
        print_colTitle(outs, "dens_alike");
        print_colTitle(outs, "dens_alike_err");
        print_colTitle(outs, "dens_unlike");
        print_colTitle(outs, "dens_unlike_err");
    }
    print_colTitle(outs, "prob_tot");
    print_colTitle(outs, "prob_err");
    if (N_FLAVORS > 1) {
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            std::string f(1, (char)('A' + flav));
            print_colTitle(outs, "prob_" + f); 
            print_colTitle(outs, "prob_err_" + f);
        }
        print_colTitle(outs, "prob_alike");
        print_colTitle(outs, "prob_alike_err");
        print_colTitle(outs, "prob_unlike");
        print_colTitle(outs, "prob_unlike_err", false, true);
    }
}

void DensitySampler::logParticles(std::vector<std::unique_ptr<class Particle>>& particles, std::ofstream& outs) const {
    const unsigned int prec = 12, width = 19;
    outs << std::scientific << std::setprecision(prec);
    for (unsigned int i = 0; i < m_numberOfParticles; i++) {
        if (N_FLAVORS > 1) {
            outs << std::setw(width) << (char)('A' + particles[i]->getFlavor()) << ",";
        }
        for (unsigned int d = 0; d < m_numberOfDimensions; d++) {
            outs << std::setw(width) << particles[i]->getPosition()[d];
            outs << ((i + 1 == m_numberOfParticles && d + 1 == m_numberOfDimensions) ? "\n" : ",");
        }
    }
}

void DensitySampler::logParticlesHeader(std::ofstream& outs) const {
    const unsigned int width = 19;
    outs << '#';
    std::string str;
    for (unsigned int i = 0; i < m_numberOfParticles; i++) {
        if (N_FLAVORS > 1) {
            str = "p" + std::to_string(i) + " flavor";
            outs << std::setw(width - (i == 0)) << str << ",";
        }
        for (unsigned int d = 0; d < m_numberOfDimensions; d++) {
            str = "p" + std::to_string(i) + " x" + std::to_string(d);
            outs << std::setw(width - (N_FLAVORS <= 1 && i == 0)) << str;
            outs << ((i + 1 == m_numberOfParticles && d + 1 == m_numberOfDimensions) ? "\n" : ",");
        }
    }
}