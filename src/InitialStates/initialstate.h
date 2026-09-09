#pragma once

#include <memory>
#include <vector>

#include "../Particles/particle.h"
#include "Math/random.h"

/**
 * @brief Sets up the initial random configuration of particles in the system.
 * * This function generates a starting state for the Variational Monte Carlo simulation.
 * It places particles randomly within a defined bounding sphere using a uniform distribution
 * and at a distance greater than min_dist from each other.
 *
 * @param numberOfDimensions The spatial dimensions of the system (e.g., 1, 2, or 3).
 * @param numberOfParticles The total number of particles to place in the trap.
 * @param randomEngine Reference to the random number generator used for placement.
 * @param min_dist Minimum reciprocal distance between particles
 * @param max_radius Maxmimum radius from the origin at which particles can be generated
 * @return std::vector<std::unique_ptr<Particle>> A vector containing the initialized particles.
 */
std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random& randomEngine,
    double min_dist = Particle::s_min_dist,
    double max_radius = Particle::s_max_radius
);
