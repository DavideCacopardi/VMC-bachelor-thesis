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
#include "../WaveFunctions/nn_envelope.h"

using namespace CommonUtils;

LennardJonesHO::LennardJonesHO(double omega, double sigma, double enEps, double alpha)
    : m_omega(omega), m_sigma(sigma), m_enEps(enEps),
    m_alpha(alpha), m_kinetic_factor(0.1 * enEps * sq(sigma))
{
    if (sigma <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (enEps <= 0) throw std::invalid_argument("enEps needs to be a positive value");
    if (alpha < 0) throw std::invalid_argument("alpha needs to be a non-negative value");
}

double LennardJonesHO::computeLocalEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles
) {
    double kineticEnergy = 0, potentialEnergy = 0;

    for (unsigned int i = 0; i < particles.size(); i++) {
        // Lennard-Jones term
        for (unsigned int j = i + 1; j < particles.size(); j++) {
            double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
            double temp = pow(m_sigma / (dist + c_eps), 6);
            temp = 4 * m_enEps * temp * (temp - 1);
            if (particles[i]->getFlavor() != particles[j]->getFlavor())
                temp *= m_alpha;
            potentialEnergy += temp;
        }
        // Harmonic term
        potentialEnergy += 0.5 * m_omega * sqNorm(particles[i]->getPosition());
    }
    
    kineticEnergy = -m_kinetic_factor * waveFunction.spatialNormalizedLaplacian(particles);
    // ...or...
    // for (unsigned int i = 0; i < particles.size(); i++) {
    //     for (unsigned int d = 0; d < particles[0]->getPosition().size(); d++) {
    //         kineticEnergy += m_kinetic_factor * sq(diff(waveFunction.evalLn, &particles[i]->getPosition()[d], particles));
    //     }
    // }

    return kineticEnergy + potentialEnergy;
}
