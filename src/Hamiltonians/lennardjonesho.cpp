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

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength, double percStrength)
    : m_omega(omega), m_sigma(sigma), m_enEps(enEps),
m_alpha(alpha), m_activate_interactions(activate_interactions), m_maxStrength(maxStrength), m_percStrength(percStrength)
{
    if (sigma <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (enEps <= 0) throw std::invalid_argument("enEps needs to be a positive value");
    if (alpha < 0) throw std::invalid_argument("alpha needs to be a non-negative value");
}

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength)
    : LennardJonesHO(omega, sigma, enEps, alpha, activate_interactions, maxStrength, 1) {}

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions)
    : LennardJonesHO(omega, sigma, enEps, alpha, activate_interactions, 1) {}

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha)
    : LennardJonesHO(omega, sigma, enEps, alpha, true) {}


double LennardJonesHO::computeLocalEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles
) {
    double potentialEnergy = 0;

    // Lennard-Jones term
    for (unsigned int i = 0; i < particles.size(); i++) {
        if (m_activate_interactions) {
            for (unsigned int j = i + 1; j < particles.size(); j++) {
                double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
                double temp = pow(m_sigma / (dist + c_eps), 6);
                temp = 4 * m_enEps * temp * (temp - 1);
                if (particles[i]->getFlavor() != particles[j]->getFlavor())
                    temp *= m_alpha;
                potentialEnergy += temp;
            }
        }
    }
    potentialEnergy *= m_maxStrength * m_percStrength;
    
    // Harmonic term
    for (unsigned int i = 0; i < particles.size(); i++) {
        potentialEnergy += 0.5 * sq(m_omega) * sqNorm(particles[i]->getPosition());
    }

    return computeLocalKineticEnergy(waveFunction, particles) + potentialEnergy;
}

double LennardJonesHO::computeLocalKineticEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles,
    unsigned int method,
    bool use_cached_result)
{
    auto* ptr = dynamic_cast<LJGaussian*>(&waveFunction);
    if (ptr != nullptr) {   // if wf is LJGaussian
        if (method == 1) {
            double sum_Laplacian2_lnPsi = 0;
            if (use_cached_result) {
                sum_Laplacian2_lnPsi = ptr->get_cachedSum_Laplacian2_lnPsi();
            }
            else {
                for (unsigned int i = 0; i < particles.size(); i++) {
                    sum_Laplacian2_lnPsi += ptr->analyticalParticleLaplacian2_lnPsi(particles, i);
                }
                ptr->set_cachedSum_Laplacian2_lnPsi(sum_Laplacian2_lnPsi);
            }
            return -0.5 * s_kinetic_factor * sum_Laplacian2_lnPsi;
        }
        else if (method == 2) {
            double sum_SqNorm_GradlnPsi = 0;
            if (use_cached_result) {
                sum_SqNorm_GradlnPsi = ptr->get_cachedSum_SqNorm_GradlnPsi();
            }
            else {
                for (unsigned int i = 0; i < particles.size(); i++) {
                    sum_SqNorm_GradlnPsi += ptr->analyticalSqNorm_ParticleGradlnPsi(particles, i);
                }
                ptr->set_cachedSum_SqNorm_GradlnPsi(sum_SqNorm_GradlnPsi);
            }
            return s_kinetic_factor * sum_SqNorm_GradlnPsi;
        }
        // else proceed as with any other wf
    }
    
    return -s_kinetic_factor * waveFunction.spatialNormalizedLaplacian(particles);
}

void LennardJonesHO::set_percStrength(double percStrength) {
    if (!(0 <= percStrength && percStrength <= 1))
        throw std::invalid_argument("percStrength needs to be a value within [0,1]");
    m_percStrength = percStrength;
}

double LennardJonesHO::get_interaction_strength() {
    return m_percStrength * m_maxStrength;
}