#include <memory>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "../common.h"
#include "repulsiveho.h"
#include "../Particles/particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/nn_envelope.h"

using namespace CommonUtils;

RepulsiveHO::RepulsiveHO(double omega)
    : RepulsiveHO(omega, omega) {}

RepulsiveHO::RepulsiveHO(double omega, double omega_z)
    : RepulsiveHO(omega, omega_z, 0.0043) {}

RepulsiveHO::RepulsiveHO(double omega, double omega_z, double repulsive_a_factor)
    : RepulsiveHO(omega, omega_z, repulsive_a_factor, std::numeric_limits<double>::infinity()) {}

RepulsiveHO::RepulsiveHO(double omega, double omega_z, double repulsive_a_factor, double hardcore_strength) 
: m_omega(omega), m_omega_z(omega_z), m_rep_a(repulsive_a_factor / sqrt(omega)), m_strength(hardcore_strength)
{
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (omega_z <= 0) throw std::invalid_argument("omega_z needs to be a positive value");
    if (repulsive_a_factor < 0) throw std::invalid_argument("repulsive_a_factor needs to be a non-negative value");
}

double RepulsiveHO::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
) {
    double kineticEnergy, potentialEnergy = 0;

    // repulsive hard-core contribution
    if (m_rep_a != 0 && m_strength != 0) {
        // if a pair of particles is found at relative distance shorter than m_rep_a, then return +∞
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int j = 0; j < i; j++) {
                double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
                if (dist <= m_rep_a) {
                    if (m_strength == std::numeric_limits<double>::infinity()) {
                        return std::numeric_limits<double>::infinity();
                    }
                    else {
                        potentialEnergy += m_strength;
                    }
                }
            }
        }
    }

    // elliptical harmonic contribution
    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
            if (j == 2) {
                sum += sq(m_omega_z * particles[i]->getPosition()[j]);
            }
            else {
                sum += sq(m_omega * particles[i]->getPosition()[j]);
            }
        }
    }
    potentialEnergy += 0.5 * sum; // * m

    kineticEnergy = -0.5 * waveFunction.spatialNormalizedLaplacian(particles);

    return kineticEnergy + potentialEnergy;
}