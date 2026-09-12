#pragma once

#include <memory>

#include "montecarlo.h"

/**
 * @brief Implements the Metropolis-Hastings algorithm with Importance Sampling.
 * * Uses the Langevin equation to guide particles towards regions of higher 
 * probability density, guided by the "quantum force" (gradient of the wave function).
 * This significantly improves the acceptance ratio and convergence speed.
 */
class MetropolisHastings : public MonteCarlo {
public:
    /**
     * @brief Constructs the Importance Sampling Metropolis-Hastings engine.
     * @param rng Unique pointer to the random number generator.
     * @param useUmrigarDrift Toggles the use of Umrigar Drift
     */
    MetropolisHastings(std::unique_ptr<class Random> rng, bool useUmrigarDrift = false);

    /**
     * @brief Proposes and evaluates a single Monte Carlo step for the system.
     * @param stepParameter The time step for the proposal.
     * @param waveFunction Reference to the system's trial wave function.
     * @param particles Vector of unique pointers to the system's particles.
     * @return True if the proposed move was accepted, False if rejected.
     */
    bool step(double stepParameter, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    /**
     * @brief Get the targeted acceptance ratio
     * @return acceptance ratio
     */
    double get_target_acceptanceRatio() const override { return 0.9; }
private:
    double m_D = 0.5;
    bool m_useUmrigarDrift = false;

    std::vector<double> quantumForce_particleWise(
        WaveFunction& wf, std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx
    );
    std::vector<double> calcDrift(
        const std::vector<double>& qforce, double timeStep);
};
