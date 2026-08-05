#include <memory>
#include <cmath>
#include <stdexcept>
#include <cassert>

#include "simplegaussian.h"
#include "common.h"
#include "wavefunction.h"
#include "../system.h"
#include "../Particles/particle.h"

using namespace CommonUtils;

SimpleGaussian::SimpleGaussian(double alpha)
    : WaveFunction(1, { alpha }) {
    if (alpha < 0) throw std::invalid_argument("alpha must be non-negative");
}

double SimpleGaussian::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
    return exp(evalLn(particles));
}

double SimpleGaussian::evalLn(std::vector<std::unique_ptr<class Particle>>& particles) {
    long double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        sum += sqNorm(particles[i]->getPosition());
    }
    return -m_parameters[0] * sum;
}

double SimpleGaussian::analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim) {
    return -2 * m_parameters[0] * particles[particle_idx]->getPosition()[dim];
}

double SimpleGaussian::analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx) {
    if (param_idx != 0) throw std::invalid_argument("ERR: invalid param_idx in analyticalParamDerivativeLnAbs");

    long double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        sum += sqNorm(particles[i]->getPosition());
    }
    return -sum;
}

double SimpleGaussian::analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) {
    unsigned int N = particles.size();
    unsigned int D = particles[0]->getNumberOfDimensions();

    long double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        sum += sqNorm(particles[i]->getPosition());
    }
    return -2 * m_parameters[0] * N * D + 4 * sq(m_parameters[0]) * sum;
}