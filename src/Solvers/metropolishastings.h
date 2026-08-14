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
    MetropolisHastings(std::unique_ptr<class Random> rng, bool useUmrigarDrift = false);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    
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
