#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "sampler.h"

/**
 * @brief Sampler dedicated to computing the radial one-body density of the system.
 * * Uses a histogram approach to track particle positions as a function of 
 * their distance from the center of the trap, yielding the spatial density \f$\rho(r)\f$.
 */
class DensitySampler : Sampler {
public:
    DensitySampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        double rMax,
        unsigned int nBins,
        bool normalize_by_nParticles);


    void sample(bool acceptedStep, class System* system, std::vector<double>* outfile = nullptr) override;
    void computeAverages();
    void computeUncorrelatedReference(unsigned long numberOfDraws, class Random& rng);
    void normalizeAgainstUncorrelated();

    // deprecated
    // void load_normalized_PCF(DensitySampler& noInt_sampler);

    void logDensity(std::ofstream& outs) const;
    void logDensityHeader(std::ofstream& outs) const;
    void logParticles(std::vector<std::unique_ptr<class Particle>>& particles, std::ofstream& outs) const;
    void logParticlesHeader(std::ofstream& outs) const;

    
    const std::vector<double>& getRadialGrid() const { return m_rGrid; }

    std::vector<double>& getDensAlike() { return m_dens_alike; };
    std::vector<double>& getDensErrAlike() { return m_dens_err_alike; };
    std::vector<double>& getDensUnlike() { return m_dens_unlike; };
    std::vector<double>& getDensErrUnlike() { return m_dens_err_unlike; };

    std::vector<double>& getProbAlike() { return m_prob_alike; };
    std::vector<double>& getProbErrAlike() { return m_prob_err_alike; };
    std::vector<double>& getProbUnlike() { return m_prob_unlike; };
    std::vector<double>& getProbErrUnlike() { return m_prob_err_unlike; };
private:
    double m_rMax;
    unsigned int m_nBins;
    bool m_normalize_by_nParticles;

    double m_dr;
    int m_nAlike = -1;
    int m_nUnlike = -1;
    bool m_nAlike_nUnlike_haveChanged = false;
    bool m_normalized_PCF = false;
    std::vector<double> m_rGrid;

    std::vector<unsigned int> m_histogram;
    std::vector<std::vector<unsigned int>> m_histFlavor;
    std::vector<unsigned int> m_histAlike;
    std::vector<unsigned int> m_histUnlike;

    std::vector<double> m_dens;
    std::vector<double> m_dens_err;
    std::vector<std::vector<double>> m_dens_f;
    std::vector<std::vector<double>> m_dens_err_f;
    std::vector<double> m_dens_alike;
    std::vector<double> m_dens_err_alike;
    std::vector<double> m_dens_unlike;
    std::vector<double> m_dens_err_unlike;
    std::vector<double> m_dens_alike_uncorr;
    std::vector<double> m_dens_err_alike_uncorr;
    std::vector<double> m_dens_unlike_uncorr;
    std::vector<double> m_dens_err_unlike_uncorr;
    std::vector<double> m_dens_alike_norm;
    std::vector<double> m_dens_err_alike_norm;
    std::vector<double> m_dens_unlike_norm;
    std::vector<double> m_dens_err_unlike_norm;

    std::vector<double> m_prob;
    std::vector<double> m_prob_err;
    std::vector<std::vector<double>> m_prob_f;
    std::vector<std::vector<double>> m_prob_err_f;
    std::vector<double> m_prob_alike;
    std::vector<double> m_prob_err_alike;
    std::vector<double> m_prob_unlike;
    std::vector<double> m_prob_err_unlike;
    std::vector<double> m_prob_alike_uncorr;
    std::vector<double> m_prob_err_alike_uncorr;
    std::vector<double> m_prob_unlike_uncorr;
    std::vector<double> m_prob_err_unlike_uncorr;
    std::vector<double> m_prob_alike_norm;
    std::vector<double> m_prob_err_alike_norm;
    std::vector<double> m_prob_unlike_norm;
    std::vector<double> m_prob_err_unlike_norm;


    std::vector<double> buildRadialCDF(unsigned int flavor) const;
    double sampleRadiusFromCDF(const std::vector<double>& cdf, class Random& rng) const;
};
