#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "energysampler.h"
#include "Hamiltonians/lennardjonesho.h"

class EnergySeparateSampler : public EnergySampler {
public:
    EnergySeparateSampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool log_grads);
    EnergySeparateSampler(const std::vector<std::unique_ptr<EnergySeparateSampler>>& others);
    ~EnergySeparateSampler() = default;

    std::unique_ptr<EnergySampler> constructMergedSampler(
        std::vector<std::unique_ptr<EnergySampler>>& others
    ) override;

    void sample(bool acceptedStep, class System* system, std::vector<double>* energiesOut = nullptr) override;
    void logOutput(const std::vector<double>& params, std::ofstream& outs) override;
    void logHeader(const std::vector<double>& params, std::ofstream& outs) override;
    void printSpecial(std::ostream& outs) override;
    void computeAverages() override;

private:
    LennardJonesHO* m_ljHam;
    std::vector<long double> m_cumulativeEnergies;
    std::vector<long double> m_cumulativeEnergiesSQ;
    std::vector<double> m_energies;
    std::vector<double> m_energiesSQ;
    std::vector<double> m_energies_variance;
    bool m_vectorsAreInitialized = false;
    void initVectors(unsigned int size);
};
