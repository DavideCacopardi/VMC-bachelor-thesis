#include <memory>
#include <vector>
#include <iostream>

#include "common.h"
#include "metropolis.h"
#include "WaveFunctions/wavefunction.h"
#include "Particles/particle.h"
#include "Math/random.h"

using namespace CommonUtils;

Metropolis::Metropolis(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng)) {
    // std::cout << "DEBUG: Created Metropolis" << std::endl;
}


bool Metropolis::step(
    double stepLength,
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles) {
    double psi_old = waveFunction.eval(particles);

    unsigned int particle_idx = m_rng->nextInt(particles.size() - 1);

    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = (m_rng->nextDouble() - .5) * stepLength;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }

    double ratio = waveFunction.eval(particles) / psi_old;

    bool accepted = m_rng->nextDouble() <= sq(ratio);
    if (!accepted) {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}
