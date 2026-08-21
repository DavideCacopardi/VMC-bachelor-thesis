#pragma once

#include <memory>
#include <vector>
#include <fstream>
#include <functional>
#include <chrono>
#include "config.h"

/**
 * @brief Main engine for Variational Monte Carlo simulations.
 * * Directs the creation of the physical environment (via Factories),
 * system initialization, and execution of the desired sampling loop.
 */
class MCEngine {
public:
    /// @brief Type for the Hamiltonian factory.
    using HamiltonianFactory = std::function<std::unique_ptr<class Hamiltonian>()>;
    /// @brief Type for the WaveFunction factory (accepts variational parameters).
    using WaveFunctionFactory = std::function<std::unique_ptr<class WaveFunction>(const std::vector<double>&)>;
    /// @brief Type for the Solver factory (accepts a random generator).
    using SolverFactory = std::function<std::unique_ptr<class MonteCarlo>(std::unique_ptr<class Random>)>;

    /**
     * @brief Initializes the VMC engine, configuring physical parameters and Factories.
     * @param hamiltonianFactory Function dynamically generating the chosen Hamiltonian.
     * @param waveFunctionFactory Function dynamically generating the WaveFunction.
     * @param solverFactory Function dynamically generating the solver (e.g., Metropolis-Hastings).
     */
    MCEngine(
        const runConfig& cfg,
        HamiltonianFactory hamiltonianFactory,
        WaveFunctionFactory waveFunctionFactory,
        SolverFactory solverFactory
    );

    std::unique_ptr<class EnergySampler> run(
        const std::vector<double>& params,
        unsigned int numberOfMetropolisSteps,
        std::vector<std::vector<double>>* energiesOut = nullptr
    );

    /**
     * @brief Executes a VMC simulation uniquely dedicated to the one-body density.
     * @param params Optimal variational parameters to utilize.
     * @param numberOfMetropolisSteps How many Metropolis steps to generate.
     * @param rMax Maximum radius of the histogram.
     * @param nBins Number of histogram subdivisions.
     * @return A DensitySampler object containing the calculated density profile.
     */
    std::unique_ptr<class DensitySampler> runOnebodyDensity(
        const std::vector<double>& params,
        unsigned int numberOfMetropolisSteps,
        double rMax, 
        unsigned int nBins,
        bool normalize_by_nParticles,
        unsigned int numberOfParticleLogs,
        std::ofstream* particlesOut);

    /**
     * @brief Retrieves the repulsive interaction parameter (hard-core diameter).
     * @return Value of the hard-core radius 'a'.
     */
    double getRepulsiveFactor() const;

    /**
     * @brief Manually instantiates a wave function for extra calculations (e.g., external derivatives).
     * @param params Vector of variational parameters.
     * @return Unique pointer to the created WaveFunction.
     */
    std::unique_ptr<class WaveFunction> makeWaveFunction(const std::vector<double>& params) const;

private:
    runConfig m_cfg;

    HamiltonianFactory m_hamiltonianFactory;
    WaveFunctionFactory m_waveFunctionFactory;
    SolverFactory m_solverFactory;
    double m_rep_a;
};