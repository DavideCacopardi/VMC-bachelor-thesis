#pragma once

#include <memory>
#include <vector>
#include <functional>

#include "Hamiltonians/hamiltonian.h"

/**
 * @brief Represents the complete quantum system.
 * * This class acts as a bridge between the Hamiltonian, WaveFunction,
 * * particles, and the Monte Carlo solver. It coordinates the step-by-step
 * * sampling process.
 */
class System {
public:
    using EnSamplerFactory = std::function<std::unique_ptr<class EnergySampler>(unsigned int, unsigned int, unsigned int, double, unsigned int, bool)>;

    /**
     * @brief Constructs the system by assembling all physical and numerical components.
     * @param hamiltonian Unique pointer to the Hamiltonian object.
     * @param waveFunction Unique pointer to the WaveFunction object.
     * @param solver Unique pointer to the Monte Carlo engine (e.g., Metropolis).
     * @param particles Vector of unique pointers to the system's particles.
     */
    System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles
    );

    /**
     * @brief Constructs the system by assembling all physical and numerical components.
     * @param hamiltonian Unique pointer to the Hamiltonian object.
     * @param waveFunction Unique pointer to the WaveFunction object.
     */
    System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction
    );

    /**
     * @brief Executes equilibration (thermalization) steps.
     * * Moves the particles towards the state of highest probability
     * without collecting any statistics.
     * @param stepParameter Step length for the solver (\f$\Delta t\f$).
     * @param numberOfEquilibrationSteps Number of steps to take and discard.
     * @return Tuned step parameter.
     */
    double runEquilibrationSteps(
        double stepParameter,
        unsigned int numberOfEquilibrationSteps);

    /**
     * @brief Executes the Metropolis simulation to sample the system's energy.
     * @param stepParameter Step length (\f$\Delta t\f$).
     * @param numberOfMetropolisSteps Total number of steps to execute.
     * @param energiesOut Vector where to store the sampled local energy at each step.
     * @param log_grads Toggles the logging of gradients.
     * @param request_Ekin Toggles the logging of the different local energy estimators (only LJGaussian).
     * @param enSamplerFactory Selects the type of EnergySampler.
     * @return An EnergySampler containing final statistics.
     */
    std::unique_ptr<class EnergySampler> runMetropolisSteps(
        double stepParameter,
        unsigned int numberOfMetropolisSteps,
        std::vector<double>* energiesOut = nullptr,
        bool log_grads = false,
        bool request_Ekin = false,
        EnSamplerFactory* enSamplerFactory = nullptr
    );

    /**
     * @brief Executes the Metropolis simulation to train the system's wavefunction
     * * to overlap the reference wf_train wavefunction.
     * @param stepParameter Step length (\f$\Delta t\f$).
     * @param numberOfMetropolisSteps Total number of steps to execute.
     * @param wf_train Pointer to the reference wavefunction to try to mimic.
     * @return An NNsampler containing final statistics.
     */
    std::unique_ptr<class NNsampler> runMetropolisSteps_NN_pretrain(double stepParameter,
        unsigned int numberOfMetropolisSteps, WaveFunction& wf_train);

    /**
     * @brief Executes a Metropolis simulation dedicated to sampling density distributions.
     * @param stepParameter Step length (\f$\Delta t\f$).
     * @param numberOfMetropolisSteps Total number of steps to execute.
     * @param rMax Maximum spatial radius covered by the histogram.
     * @param nBins Number of bins for the radial histogram.
     * @param normalize_by_nParticles Toggles whether to normalize the bin count wrt
     * * the number of particles or number of interactions, respective of histogram.
     * @param numberOfParticleLogs Number snapshots of the particle positions to log.
     * @param particlesOut Logging file for particle positions.
     * @return A DensitySampler containing the radial density histograms and distributions.
     */
    std::unique_ptr<class DensitySampler> runMetropolisStepsSpatial(
        double stepParameter, unsigned int numberOfMetropolisSteps,
        double rMax, unsigned int nBins, bool normalize_by_nParticles, unsigned int numberOfParticleLogs, std::ofstream* particlesOut);

    /**
     * @brief Helper to call the hamiltonian's computeLocalEnergy method.
     * @return The local energy.
     */
    double inline computeLocalEnergy() {
        return m_hamiltonian->computeLocalEnergy(
            *m_waveFunction, m_particles);
    }

    /**
     * @brief Helper to call the hamiltonian's computeLocalEnergies method.
     * @return The local energies.
     */
    std::vector<double> inline computeLocalEnergies() {
        return m_hamiltonian->computeLocalEnergies(
            *m_waveFunction, m_particles);
    }

    /**
     * @brief Gets the system's wavefunction.
     * @return The system's wavefunction.
     */
    class WaveFunction& getWaveFunction();

    /**
     * @brief Gets the system's Hamiltonian.
     * @return The system's Hamiltonian.
     */
    class Hamiltonian& getHamiltonian();

    /**
     * @brief Gets the current variational parameters.
     * @return Vector containing the wave function's parameters.
     */
    const std::vector<double>& getWaveFunctionParameters();

    /**
     * @brief Retrieves the particles currently in the system.
     * @return Reference to the vector of particle pointers.
     */
    std::vector<std::unique_ptr<class Particle>>& getParticles() { return m_particles; }

    /**
     * @brief Sets new vector of particles for the system.
     */
    void setParticles(std::vector<std::unique_ptr<class Particle>> new_particles);

    /**
     * @brief Sets the Solver to employ for MonteCarlo simultations.
     */
    void setSolver(std::unique_ptr<class MonteCarlo> new_solver);

    /**
     * @brief Sets a new Hamiltonian for the system.
     */
    void setHamiltonian(std::unique_ptr<class Hamiltonian> new_hamiltonian);

    /**
     * @brief Sets a new WaveFunction for the system.
     */
    std::unique_ptr<WaveFunction> setWaveFunction(std::unique_ptr<WaveFunction> new_waveFunction);

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};

