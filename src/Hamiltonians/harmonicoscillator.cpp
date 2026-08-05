#include <memory>
#include <cassert>
#include <iostream>

#include "../common.h"
#include "harmonicoscillator.h"
#include "../Particles/particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/nn_envelope.h"

using namespace CommonUtils;

HarmonicOscillator::HarmonicOscillator(double omega) {
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    m_omega = omega;
}

double HarmonicOscillator::computeLocalEnergy(
    WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles
) {
    double kineticEnergy, potentialEnergy = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        potentialEnergy += 0.5 * m_omega * sqNorm(particles[i]->getPosition());
    }
    kineticEnergy = -0.5 * waveFunction.spatialNormalizedLaplacian(particles);
    return kineticEnergy + potentialEnergy;
}
