#include <memory>
#include <vector>
#include <iostream>

#include "common.h"
#include "metropolishastings.h"
#include "WaveFunctions/wavefunction.h"
#include "Particles/particle.h"
#include "Math/random.h"

using namespace CommonUtils;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng)) {}

bool MetropolisHastings::step(double timeStep, class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles) {    
    // Perform the actual Metropolis-Hastings step
    double psi_old = waveFunction.eval(particles);

    unsigned int particle_idx = m_rng->nextInt(0, particles.size() - 1);
    std::vector<double> qforceold = quantumForce_particleWise(waveFunction, particles, particle_idx);
    
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    // update position
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = m_rng->nextGaussian(0.0, 1.0) * sqrt(timeStep) + qforceold[i] * timeStep * m_D;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }
    double ratio = waveFunction.eval(particles) / psi_old;
    std::vector<double> qforcenew = quantumForce_particleWise(waveFunction, particles, particle_idx);
    
    // evaluate GreensFunction
    double GreensFunction = 0;
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        GreensFunction += 0.5 * (qforceold[i] + qforcenew[i]) *
        (m_D * timeStep * 0.5 * (qforceold[i] - qforcenew[i]) - displacement[i]);
    }
    GreensFunction = exp(GreensFunction);

    // accept or reject
    bool accepted = m_rng->nextDouble() <= GreensFunction * sq(ratio);
    if (!accepted) {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}

std::vector<double> MetropolisHastings::quantumForce_particleWise(
    WaveFunction& wf, std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx
) {
    std::vector<double> vec = wf.spatialGradientLn(particles, particle_idx);
    for (unsigned int i = 0; i < vec.size(); i++)
        vec[i] *= 2;
    return vec;
}