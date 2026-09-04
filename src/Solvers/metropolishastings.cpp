#include <memory>
#include <vector>
#include <iostream>

#include "common.h"
#include "metropolishastings.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "Particles/particle.h"
#include "Math/random.h"

using namespace CommonUtils;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng, bool useUmrigarDrift)
    : MonteCarlo(std::move(rng)), m_useUmrigarDrift(useUmrigarDrift) {
    m_D = Hamiltonian::get_kinetic_factor();
}

bool MetropolisHastings::step(double timeStep, class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles) {    
    // Perform the actual Metropolis-Hastings step
    double psi_old = waveFunction.eval(particles);

    unsigned int particle_idx = m_rng->nextInt(0, particles.size() - 1);
    std::vector<double> qforceold = quantumForce_particleWise(waveFunction, particles, particle_idx);
    
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    // update position
    std::vector<double> drift = calcDrift(qforceold, timeStep);
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = m_rng->nextGaussian(0.0, 1.0) * sqrt(2 * m_D * timeStep) + drift[i];
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }
    double ratio = waveFunction.eval(particles) / psi_old;
    std::vector<double> qforcenew = quantumForce_particleWise(waveFunction, particles, particle_idx);
    
    // evaluate GreensFunction
    double greensExponent = 0;
    if (m_useUmrigarDrift) {
        std::vector<double> drift_new = calcDrift(qforcenew, timeStep);
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            // exponent = 1/(4*D*tau) * [ (displacement - drift)^2 - (-displacement - drift_new)^2 ]
            greensExponent += (-displacement[i] * (drift[i] + drift_new[i]) * 0.5 
                            + 0.25 * (sq(drift[i]) - sq(drift_new[i]))) / (m_D * timeStep);
        }
    }
    else {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            greensExponent += 0.5 * (qforceold[i] + qforcenew[i]) *
                (m_D * timeStep * 0.5 * (qforceold[i] - qforcenew[i]) - displacement[i]);
        }
    }
    double GreensFunction = exp(greensExponent);

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

std::vector<double> MetropolisHastings::calcDrift(
    const std::vector<double>& qforce, double timeStep
) {
    const double a = 1;
    std::vector<double> drift(qforce.size());
    for (unsigned int i = 0; i < drift.size(); i++) {
        drift[i] = qforce[i] * timeStep * m_D;
    }
    if (m_useUmrigarDrift) {
        double sq_qforce = sqNorm(qforce); 
        double dimensionless = m_D * sq_qforce * timeStep;

        // double sqn = sqNorm(drift);
        double umrigar_multiplier = 1.0; // sqn -> 0
        if (dimensionless > 1e-12) {
            umrigar_multiplier = (sqrt(1 + 2 * a * dimensionless) - 1) / (a * dimensionless);
        }
        for (unsigned int i = 0; i < drift.size(); i++) {
            drift[i] *= umrigar_multiplier;
        }
    }
    return drift;
}