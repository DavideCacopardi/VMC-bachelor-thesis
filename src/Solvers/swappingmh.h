#pragma once

#include <memory>

#include "metropolishastings.h"

/**
 * @brief Implements the Metropolis-Hastings algorithm with Importance Sampling.
 * * Uses the Langevin equation to guide particles towards regions of higher 
 * probability density, guided by the "quantum force" (gradient of the wave function).
 * This significantly improves the acceptance ratio and convergence speed.
 */
class SwappingMH : public MetropolisHastings {
public:
    SwappingMH(std::unique_ptr<class Random> rng);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    double get_target_acceptanceRatio() const override { return 0.93; }
};
