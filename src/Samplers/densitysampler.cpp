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
#include "Math/random.h"
#include "densitysampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

DensitySampler::DensitySampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    bool normalize_by_nParticles
) : Sampler(numberOfParticles,
    numberOfDimensions,
    0,
    0,
    numberOfMetropolisSteps
),
m_rMax(rMax),
m_nBins(nBins),
m_normalize_by_nParticles(normalize_by_nParticles)
{
    reset();
}

DensitySampler::DensitySampler(const std::vector<std::unique_ptr<DensitySampler>>& others)
    : DensitySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        0,
        others[0]->m_rMax,
        others[0]->m_nBins,
        others[0]->m_normalize_by_nParticles) 
{
    // m_rGrid = others[0]->m_rGrid;
    m_nAlike = others[0]->m_nAlike;
    m_nUnlike = others[0]->m_nUnlike;
    m_nAlike_nUnlike_haveChanged = others[0]->m_nAlike_nUnlike_haveChanged;
    for (unsigned int i = 0; i < others.size(); i++) {
        mergeBaseData(others[i].get());
    }
    computeAverages();
    computeAveragesUncorrelatedReference();
    normalizeAgainstUncorrelated();
}

void DensitySampler::reset() {
    m_dr = m_rMax / m_nBins;

    m_histogram.assign(m_nBins, 0);
    m_histFlavor.resize(m_nBins);
    m_histAlike.assign(m_nBins, 0);
    m_histUnlike.assign(m_nBins, 0);
    m_histAlikeUncorr.assign(m_nBins, 0);
    m_histUnlikeUncorr.assign(m_nBins, 0);

    m_rGrid.assign(m_nBins, 0.0);

    m_dens.assign(m_nBins, 0.0);
    m_dens_err.assign(m_nBins, 0.0);
    m_dens_f.resize(m_nBins);
    m_dens_err_f.resize(m_nBins);
    m_dens_alike.resize(m_nBins, 0.0);
    m_dens_err_alike.resize(m_nBins, 0.0);
    m_dens_unlike.resize(m_nBins, 0.0);
    m_dens_err_unlike.resize(m_nBins, 0.0);
    m_dens_alike_uncorr.resize(m_nBins, 0.0);
    m_dens_err_alike_uncorr.resize(m_nBins, 0.0);
    m_dens_unlike_uncorr.resize(m_nBins, 0.0);
    m_dens_err_unlike_uncorr.resize(m_nBins, 0.0);

    m_prob.assign(m_nBins, 0.0);
    m_prob_err.assign(m_nBins, 0.0);
    m_prob_f.resize(m_nBins);
    m_prob_err_f.resize(m_nBins);
    m_prob_alike.resize(m_nBins, 0.0);
    m_prob_err_alike.resize(m_nBins, 0.0);
    m_prob_unlike.resize(m_nBins, 0.0);
    m_prob_err_unlike.resize(m_nBins, 0.0);
    m_prob_alike_uncorr.resize(m_nBins, 0.0);
    m_prob_err_alike_uncorr.resize(m_nBins, 0.0);
    m_prob_unlike_uncorr.resize(m_nBins, 0.0);
    m_prob_err_unlike_uncorr.resize(m_nBins, 0.0);

    for (unsigned int i = 0; i < m_nBins; i++) {
        m_rGrid[i] = (i + 0.5) * m_dr;

        m_histFlavor[i].assign(N_FLAVORS, 0);
        m_dens_f[i].assign(N_FLAVORS, 0.0);
        m_dens_err_f[i].assign(N_FLAVORS, 0.0);
        m_prob_f[i].assign(N_FLAVORS, 0.0);
        m_prob_err_f[i].assign(N_FLAVORS, 0.0);
    }
}

std::unique_ptr<DensitySampler> DensitySampler::constructMergedSampler(
    std::vector<std::unique_ptr<DensitySampler>>& others, bool calc_merged_errors
) {
    std::unique_ptr<DensitySampler> merged = std::make_unique<DensitySampler>(others);
    if (calc_merged_errors) {
        merged->calcStatErrors(others);
    }
    return merged;
};

void DensitySampler::mergeBaseData(const DensitySampler* other) {
    if (other->m_elapsedTime > m_elapsedTime)
        m_elapsedTime = other->m_elapsedTime;
    m_numberOfMetropolisSteps += other->m_numberOfMetropolisSteps;
    m_numberOfAcceptedSteps += other->m_numberOfAcceptedSteps;
    m_uncorrReference_nDraws += other->m_uncorrReference_nDraws;
    

    for (unsigned int i = 0; i < m_nBins; i++) {
        m_histogram[i] += other->m_histogram[i];
        m_histAlike[i] += other->m_histAlike[i];
        m_histUnlike[i] += other->m_histUnlike[i];
        m_histAlikeUncorr[i] += other->m_histAlikeUncorr[i];
        m_histUnlikeUncorr[i] += other->m_histUnlikeUncorr[i];
        for (unsigned int j = 0; j < N_FLAVORS; j++) {
            m_histFlavor[i][j] += other->m_histFlavor[i][j];
        }
    }
}

void DensitySampler::calcStatErrors(const std::vector<std::unique_ptr<DensitySampler>>& others) {
    unsigned int M = others.size();
    double sum = 0;
    for (unsigned int i = 0; i < m_nBins; i++) {
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_dens[i] - m_dens[i]);
        m_dens_err[i] = std::max(sqrt(sum) / M, m_dens_err[i]);
        sum = 0;
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            for (unsigned int j = 0; j < M; j++)
                sum += sq(others[j]->m_dens_f[i][flav] - m_dens_f[i][flav]);
            m_dens_err_f[i][flav] = std::max(sqrt(sum) / M, m_dens_err_f[i][flav]);
        }
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_dens_alike[i] - m_dens_alike[i]);
        m_dens_err_alike[i] = std::max(sqrt(sum) / M, m_dens_err_alike[i]);
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_dens_unlike[i] - m_dens_unlike[i]);
        m_dens_err_unlike[i] = std::max(sqrt(sum) / M, m_dens_err_unlike[i]);

        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_prob[i] - m_prob[i]);
        m_prob_err[i] = std::max(sqrt(sum) / M, m_prob_err[i]);
        sum = 0;
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            for (unsigned int j = 0; j < M; j++)
                sum += sq(others[j]->m_prob_f[i][flav] - m_prob_f[i][flav]);
            m_prob_err_f[i][flav] = std::max(sqrt(sum) / M, m_prob_err_f[i][flav]);
        }
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_prob_alike[i] - m_prob_alike[i]);
        m_prob_err_alike[i] = std::max(sqrt(sum) / M, m_prob_err_alike[i]);
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_prob_unlike[i] - m_prob_unlike[i]);
        m_prob_err_unlike[i] = std::max(sqrt(sum) / M, m_prob_err_unlike[i]);

        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_dens_alike_uncorr[i] - m_dens_alike_uncorr[i]);
        m_dens_err_alike_uncorr[i] = std::max(sqrt(sum) / M, m_dens_err_alike_uncorr[i]);
        sum = 0;
        for (unsigned int j = 0; j < M; j++)
            sum += sq(others[j]->m_dens_unlike_uncorr[i] - m_dens_unlike_uncorr[i]);
        m_dens_err_unlike_uncorr[i] = std::max(sqrt(sum) / M, m_dens_err_unlike_uncorr[i]);
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

            print_colVal(outs, m_dens_alike_norm[i]);
            print_colVal(outs, m_dens_err_alike_norm[i]);
            print_colVal(outs, m_dens_unlike_norm[i]);
            print_colVal(outs, m_dens_err_unlike_norm[i]);

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
        print_colTitle(outs, "dens_err_alike");
        print_colTitle(outs, "dens_unlike");
        print_colTitle(outs, "dens_err_unlike");
        print_colTitle(outs, "dens_alike_n");
        print_colTitle(outs, "dens_err_alike_n");
        print_colTitle(outs, "dens_unlike_n");
        print_colTitle(outs, "dens_err_unlike_n");
        
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
        print_colTitle(outs, "prob_err_alike");
        print_colTitle(outs, "prob_unlike");
        print_colTitle(outs, "prob_err_unlike", false);
        print_colTitle(outs, "prob_alike_n");
        print_colTitle(outs, "prob_err_alike_n");
        print_colTitle(outs, "prob_unlike_n");
        print_colTitle(outs, "prob_err_unlike_n", false, true);
    }
}

void DensitySampler::logParticles(std::vector<std::unique_ptr<class Particle>>& particles, std::ofstream& outs) const {
    outs << std::scientific;
    for (unsigned int i = 0; i < m_numberOfParticles; i++) {
        if (N_FLAVORS > 1) {
            print_colVal(outs, (char)('A' + particles[i]->getFlavor()));
        }
        for (unsigned int d = 0; d < m_numberOfDimensions; d++) {
            print_colVal(outs, particles[i]->getPosition()[d], false, (i + 1 == m_numberOfParticles && d + 1 == m_numberOfDimensions));
        }
    }
    outs << std::flush;
}

void DensitySampler::logParticlesHeader(std::ofstream& outs) const {
    outs << '#';
    std::string str;
    for (unsigned int i = 0; i < m_numberOfParticles; i++) {
        if (N_FLAVORS > 1) {
            str = "p" + std::to_string(i) + " flavor";
            print_colTitle(outs, str, i == 0, false);
        }
        for (unsigned int d = 0; d < m_numberOfDimensions; d++) {
            str = "p" + std::to_string(i) + " x" + std::to_string(d);
            print_colTitle(outs, str, (N_FLAVORS <= 1 && i == 0), (i + 1 == m_numberOfParticles && d + 1 == m_numberOfDimensions));
        }
    }
}

std::vector<double> DensitySampler::buildRadialCDF(unsigned int flavor) const {
    if (flavor >= N_FLAVORS)
        throw std::invalid_argument(" ERR: Flavor " + std::to_string(flavor) + " does not exist.");
    
    std::vector<double> cdf(m_nBins, 0.0);
    double running = 0.0;
    for (unsigned int i = 0; i < m_nBins; i++) {
        running += static_cast<double>(m_histFlavor[i][flavor]);
        cdf[i] = running;
    }
    double total = cdf.back();
    if (total > 0.0)
        for (unsigned int i = 0; i < m_nBins; i++) cdf[i] /= total;
    return cdf;
}

double DensitySampler::sampleRadiusFromCDF(const std::vector<double>& cdf, Random& rng) const {
    double u = rng.nextDouble();
    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    unsigned int bin = static_cast<unsigned int>(
        std::min<size_t>(it - cdf.begin(), m_nBins - 1));
    return m_rGrid[bin] + (rng.nextDouble() - 0.5) * m_dr;  // jitter within the bin
}

void DensitySampler::computeUncorrelatedReference(unsigned int nDraws, Random& rng) {
    m_uncorrReference_nDraws = nDraws;
    // Assumes two-species case (A=0, B=1), N_A == N_B in 3D
    std::vector<double> cdfA = buildRadialCDF(0);
    std::vector<double> cdfB = buildRadialCDF(1);

    for (unsigned long k = 0; k < m_uncorrReference_nDraws; k++) {
        // alike: alternate AA / BB draws (exact 50/50 since N_A == N_B)
        for (unsigned int i = 0; i < m_nAlike; i++) {
            const std::vector<double>& cdfAlike = (k % 2 == 0) ? cdfA : cdfB;
            double r1 = sampleRadiusFromCDF(cdfAlike, rng);
            double r2 = sampleRadiusFromCDF(cdfAlike, rng);
            double cosG = 2 * rng.nextDouble() - 1;
            double R = std::sqrt(std::max(0.0, r1 * r1 + r2 * r2 - 2 * r1 * r2 * cosG));
            unsigned int bin = static_cast<unsigned int>(R / m_dr);
            if (bin < m_nBins) m_histAlikeUncorr[bin]++;
        }

        // unlike: one A, one B, always independent
        for (unsigned int i = 0; i < m_nUnlike; i++) {
            double r1u = sampleRadiusFromCDF(cdfA, rng);
            double r2u = sampleRadiusFromCDF(cdfB, rng);
            double cosGu = 2 * rng.nextDouble() - 1;
            double Ru = std::sqrt(std::max(0.0, r1u * r1u + r2u * r2u - 2 * r1u * r2u * cosGu));
            unsigned int binU = static_cast<unsigned int>(Ru / m_dr);
            if (binU < m_nBins) m_histUnlikeUncorr[binU]++;
        }
    }
}

void DensitySampler::computeAveragesUncorrelatedReference() {
    double half_d = 0.5 * m_numberOfDimensions;
    double volume_coeff = pow(M_PI, half_d) / tgamma(half_d + 1.0);

    for (unsigned int i = 0; i < m_nBins; i++) {
        double r_inner = i * m_dr, r_outer = (i + 1) * m_dr;
        double volume = volume_coeff * (pow(r_outer, m_numberOfDimensions) - pow(r_inner, m_numberOfDimensions));
        if (volume <= 0.0) continue;

        double normalization = static_cast<double>(m_uncorrReference_nDraws) * volume;
        if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
            normalization *= (double)m_nAlike;
        m_dens_alike_uncorr[i] = m_histAlikeUncorr[i] / normalization;
        m_dens_err_alike_uncorr[i] = sqrt((double)m_histAlikeUncorr[i]) / normalization;
        normalization = static_cast<double>(m_uncorrReference_nDraws) * volume;
        if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
                normalization *= (double)m_nUnlike;
        m_dens_unlike_uncorr[i] = m_histUnlikeUncorr[i] / normalization;
        m_dens_err_unlike_uncorr[i] = sqrt((double)m_histUnlikeUncorr[i]) / normalization;

        normalization = static_cast<double>(m_uncorrReference_nDraws) * (r_outer - r_inner);
        if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
            normalization *= (double)m_nAlike;
        m_prob_alike_uncorr[i] = m_histAlikeUncorr[i] / normalization;
        m_prob_err_alike_uncorr[i] = sqrt((double)m_histAlikeUncorr[i]) / normalization;
        if (m_normalize_by_nParticles && !m_nAlike_nUnlike_haveChanged)
            normalization *= (double)m_nUnlike;
        m_prob_unlike_uncorr[i] = m_histUnlikeUncorr[i] / normalization;
        m_prob_err_unlike_uncorr[i] = sqrt((double)m_histUnlikeUncorr[i]) / normalization;
    }
}

void DensitySampler::normalizeAgainstUncorrelated() {
    m_dens_alike_norm.resize(m_nBins);
    m_dens_err_alike_norm.resize(m_nBins);
    m_dens_unlike_norm.resize(m_nBins);
    m_dens_err_unlike_norm.resize(m_nBins);

    for (unsigned int i = 0; i < m_nBins; i++) {
        m_dens_alike_norm[i] = m_dens_alike[i] / m_dens_alike_uncorr[i];
        m_dens_err_alike_norm[i] = norm({
            m_dens_alike[i] / sq(m_dens_alike_uncorr[i]) * m_dens_err_alike_uncorr[i],
            m_dens_err_alike[i] / m_dens_alike_uncorr[i] });
        m_dens_unlike_norm[i] = m_dens_unlike[i] / m_dens_unlike_uncorr[i];
        m_dens_err_unlike_norm[i] = norm({
            m_dens_unlike[i] / sq(m_dens_unlike_uncorr[i]) * m_dens_err_unlike_uncorr[i],
            m_dens_err_unlike[i] / m_dens_unlike_uncorr[i] });
    }
}


/*
DEPRECATED:
void DensitySampler::load_normalized_PCF(DensitySampler& oth) {
    m_dens_alike_norm.resize(m_nBins);
    m_dens_err_alike_norm.resize(m_nBins);
    m_dens_unlike_norm.resize(m_nBins);
    m_dens_err_unlike_norm.resize(m_nBins);
    m_prob_alike_norm.resize(m_nBins);
    m_prob_err_alike_norm.resize(m_nBins);
    m_prob_unlike_norm.resize(m_nBins);
    m_prob_err_unlike_norm.resize(m_nBins);
    m_normalized_PCF = true;
    for (unsigned int i = 0; i < m_nBins; i++) {
        m_dens_alike_norm[i] = m_dens_alike[i] / oth.getDensAlike()[i];
        m_dens_err_alike_norm[i] = norm(
            { m_dens_alike[i] / sq(oth.getDensAlike()[i]) * oth.getDensErrAlike()[i],
            m_dens_err_alike[i] / oth.getDensAlike()[i] }
        );
        m_dens_unlike_norm[i] = m_dens_unlike[i] / oth.getDensUnlike()[i];
        m_dens_err_unlike_norm[i] = norm(
            { m_dens_unlike[i] / sq(oth.getDensUnlike()[i]) * oth.getDensErrUnlike()[i],
            m_dens_err_unlike[i] / oth.getDensUnlike()[i] }
        );

        m_prob_alike_norm[i] = m_prob_alike[i] / oth.getProbAlike()[i];
        m_prob_err_alike_norm[i] = norm(
            { m_prob_alike[i] / sq(oth.getProbAlike()[i]) * oth.getProbErrAlike()[i],
            m_prob_err_alike[i] / oth.getProbAlike()[i] }
        );
        m_prob_unlike_norm[i] = m_prob_unlike[i] / oth.getProbUnlike()[i];
        m_prob_err_unlike_norm[i] = norm(
            { m_prob_unlike[i] / sq(oth.getProbUnlike()[i]) * oth.getProbErrUnlike()[i],
            m_prob_err_unlike[i] / oth.getProbUnlike()[i] }
        );
    }
}
*/
