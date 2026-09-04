#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "sampler.h"

/**
 * @brief Sampler dedicated to computing the energy of the quantum system.
 * * Accumulates the local energy at each Metropolis step to calculate 
 * the expectation value of the Hamiltonian, its variance, and the standard error.
 */
class EnergySampler : public Sampler {
public:
    EnergySampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool log_grads);
    // Construct merged sampler
    EnergySampler(const std::vector<std::unique_ptr<EnergySampler>>& others);
    virtual ~EnergySampler() = default;

    virtual void reset();
    virtual std::unique_ptr<EnergySampler> constructMergedSampler(
        std::vector<std::unique_ptr<EnergySampler>>& others
    ) {
        return std::make_unique<EnergySampler>(others);
    };

    virtual void sample(bool acceptedStep, class System* system, std::vector<double>* energiesOut = nullptr);

    virtual void logHeader(const std::vector<double>& params, std::ofstream& outs);
    virtual void logOutput(const std::vector<double>& params, std::ofstream& outs);
    void logOutput(std::ofstream& outs, std::vector<double> additional_log = {});
    virtual void printSpecial(std::ostream&) {};
    virtual void computeAverages();
    double getEnergy() { return m_energy; }
    double getVariance() { return m_variance; }
    double getError() { return m_error; }
    double getAcceptanceRatio() { return (double)m_numberOfAcceptedSteps / (double)m_numberOfMetropolisSteps; }
    double getCovariance(unsigned int param_idx) { return m_covarianceE[param_idx]; }
    std::vector<double> get_dEdW() const;
    std::vector<double> get_dVardW() const;
    void setElapsedTime(std::chrono::duration<double> time);
protected:
    void mergeBaseData(const EnergySampler* other);
    double m_energy = 0;
    double m_energySQ = 0;
    double m_variance = 0;
    double m_error = 0;
    bool m_log_grads = false;
    std::vector<double> m_covarianceE;
    std::vector<double> m_covarianceE2;
    long double m_cumulativeEnergy = 0;
    long double m_cumulativeEnergySQ = 0;
    std::vector<long double> m_cumulativeOpO;
    std::vector<long double> m_cumulativeOpOE;
    std::vector<long double> m_cumulativeOpOE2;
    std::vector<long double> m_cumulativeEdEdW;
};
