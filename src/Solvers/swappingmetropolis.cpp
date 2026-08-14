#include <memory>
#include <vector>
#include <iostream>

#include "common.h"
#include "swappingmetropolis.h"
#include "WaveFunctions/wavefunction.h"
#include "Particles/particle.h"
#include "Math/random.h"

using namespace CommonUtils;

SwappingMetropolis::SwappingMetropolis(std::unique_ptr<class Random> rng)
    : Metropolis(std::move(rng)) {}

bool SwappingMetropolis::step(double timeStep, class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles) {
    unsigned int whatToDo = m_rng->nextInt(0, 9);   // HARD-CODED!

    if (whatToDo == 0 && particles.size() > 1) {    // swap particle (every 1 in 10 extractions)
        double psi_old = waveFunction.eval(particles);

        Flavor f_0, f_1;
        unsigned int particle_idx_0, particle_idx_1;
        particle_idx_0 = m_rng->nextInt(0, particles.size() - 1);
        f_0 = particles[particle_idx_0]->getFlavor();
        do {
            particle_idx_1 = m_rng->nextInt(0, particles.size() - 1);
            f_1 = particles[particle_idx_1]->getFlavor();
        } while (f_1 == f_0);
        
        // swap flavor
        particles[particle_idx_0]->setFlavor(f_1);
        particles[particle_idx_1]->setFlavor(f_0);
        
        double ratio = waveFunction.eval(particles) / psi_old;
        
        bool accepted = m_rng->nextDouble() <= sq(ratio);
        if (!accepted) {
            particles[particle_idx_0]->setFlavor(f_0);
            particles[particle_idx_1]->setFlavor(f_1);
        }

        return accepted;
    }
    // else: perform the usual Metropolis-Hastings step
    return Metropolis::step(timeStep, waveFunction, particles);
}