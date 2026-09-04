#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "energysampler.h"
#include "Hamiltonians/lennardjonesho.h"

/**
 * @brief Sampler dedicated to computing the energy of the quantum system.
 * * Accumulates the local energy at each Metropolis step to calculate 
 * the expectation value of the Hamiltonian, its variance, and the standard error.
 */
class EnergyEkinSampler : public EnergySampler {
public:
    EnergyEkinSampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool log_grads);
    EnergyEkinSampler(const std::vector<std::unique_ptr<EnergyEkinSampler>>& others);
    ~EnergyEkinSampler() = default;

    std::unique_ptr<EnergySampler> constructMergedSampler(
        std::vector<std::unique_ptr<EnergySampler>>& others
    ) override;

    void sample(bool acceptedStep, class System* system, std::vector<double>* energiesOut = nullptr) override;
    void logOutput(const std::vector<double>& params, std::ofstream& outs) override;
    void logHeader(const std::vector<double>& params, std::ofstream& outs) override;
    void computeAverages() override;
    double getEkin1() { return m_Ekin1; }
    double getEkin1_variance() { return m_Ekin1_variance; }
    double getEkin2() { return m_Ekin2; }
    double getEkin2_variance() { return m_Ekin2_variance; }
    void printSpecial(std::ostream& outs) override;
private:
    LennardJonesHO* m_ljHam;
    long double m_cumulativeEkin1 = 0;
    long double m_cumulativeEkin1SQ = 0;
    long double m_cumulativeEkin2 = 0;
    long double m_cumulativeEkin2SQ = 0;
    double m_Ekin1;
    double m_Ekin1SQ;
    double m_Ekin1_variance;
    double m_Ekin2;
    double m_Ekin2SQ;
    double m_Ekin2_variance;
};
