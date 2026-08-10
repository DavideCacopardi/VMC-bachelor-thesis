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
    bool step(
        double stepLength,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    double get_target_acceptanceRatio() const override { return 0.5; }
};
