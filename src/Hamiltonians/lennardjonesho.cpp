#include <memory>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "../common.h"
#include "lennardjonesho.h"
#include "../Particles/particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/ljgaussian.h"

using namespace CommonUtils;

unsigned int LennardJonesHO::s_loc_Ken_method = 0;

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions)
    : m_omega(omega), m_sigma(sigma), m_enEps(enEps),
m_alpha(alpha), m_kinetic_factor(0.1 * enEps * sq(sigma)), m_activate_interactions(activate_interactions)
{
    if (sigma <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (enEps <= 0) throw std::invalid_argument("enEps needs to be a positive value");
    if (alpha < 0) throw std::invalid_argument("alpha needs to be a non-negative value");
}

double LennardJonesHO::computeLocalEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles
) {
    double potentialEnergy = 0;

    for (unsigned int i = 0; i < particles.size(); i++) {
        if (m_activate_interactions) {
            // Lennard-Jones term
            for (unsigned int j = i + 1; j < particles.size(); j++) {
                double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
                double temp = pow(m_sigma / (dist + c_eps), 6);
                temp = 4 * m_enEps * temp * (temp - 1);
                if (particles[i]->getFlavor() != particles[j]->getFlavor())
                    temp *= m_alpha;
                potentialEnergy += temp;
            }
        }
        // Harmonic term
        potentialEnergy += 0.5 * sq(m_omega) * sqNorm(particles[i]->getPosition());
    }

    return computeLocalKineticEnergy(waveFunction, particles) + potentialEnergy;
}

double LennardJonesHO::computeLocalKineticEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles,
    unsigned int method
) {
    auto* ptr = dynamic_cast<LJGaussian*>(&waveFunction);
    if (ptr != nullptr) {   // if wf is LJGaussian
        if (method == 1) {
            double sum = 0;
            for (unsigned int i = 0; i < particles.size(); i++) {
                sum += ptr->analyticalParticleLaplacian2_lnPsi(particles, i);
            }
            return -0.5 * m_kinetic_factor * sum;
        }
        else if (method == 2) {
            double sum = 0;
            for (unsigned int i = 0; i < particles.size(); i++) {
                sum += ptr->analyticalSqNorm_ParticleGradlnPsi(particles, i);
            }
            return m_kinetic_factor * sum;
        }
        // else proceed as with any other wf
    }
    
    return -m_kinetic_factor * waveFunction.spatialNormalizedLaplacian(particles);
}