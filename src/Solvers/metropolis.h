#pragma once

#include <memory>

#include "montecarlo.h"

/**
 * @brief Implements the standard Metropolis algorithm (Brute Force).
 * * Proposes symmetric, uniformly distributed random moves for the particles.
 */
class Metropolis : public MonteCarlo {
public:
    Metropolis(std::unique_ptr<class Random> rng);

    /**
     * @brief Proposes and evaluates a single Monte Carlo step for the system.
     * @param stepParameter The step length for the proposal.
     * @param waveFunction Reference to the system's trial wave function.
     * @param particles Vector of unique pointers to the system's particles.
     * @return True if the proposed move was accepted, False if rejected.
     */
    bool step(
        double stepLength,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    /**
     * @brief Get the targeted acceptance ratio
     * @return acceptance ratio
     */
    double get_target_acceptanceRatio() const override { return 0.5; }
};
