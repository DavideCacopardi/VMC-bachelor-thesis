#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include "../common.h"
#include "ellipticgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../Particles/particle.h"

using namespace CommonUtils;

EllipticGaussian::EllipticGaussian(double alpha, double beta)
    : WaveFunction(2, { alpha, beta }) {
    if (alpha < 0) throw std::invalid_argument("ERR: alpha must be non-negative");
    if (beta < 0) throw std::invalid_argument("ERR: beta must be non-negative");
}

double EllipticGaussian::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
    return exp(evalLn(particles));
}

double EllipticGaussian::evalLn(std::vector<std::unique_ptr<class Particle>>& particles) {
    long double sum = 0;

    // sum all coordinates squared:  Σᵢ [xᵢ² + yᵢ² + β zᵢ²]
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2)
                sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
            else
                sum += sq(particles[i]->getPosition()[j]);
        }
    }

    return -m_parameters[0] * sum;  // -α * sum
}

double EllipticGaussian::analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim) {
    // For z, multiply by beta (m_parameters[1])
    if (dim == 2) {
        return -2 * m_parameters[0] * m_parameters[1] * particles[particle_idx]->getPosition()[dim];
    }
    // For x and y
    return -2 * m_parameters[0] * particles[particle_idx]->getPosition()[dim];
}

double EllipticGaussian::analyticalParamDerivativeLnAbs(
    std::vector<std::unique_ptr<class Particle>>& particles, unsigned int param_idx) {
    long double sum = 0;

    if (param_idx == 0) {       // derivative wrt alpha
        // sum all coordinates squared:  Σ_i [x_i² + y_i² + βz_i²]
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int j = 0; j < m_NDIM; j++) {
                if (j == 2)
                    sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
                else
                    sum += sq(particles[i]->getPosition()[j]);
            }
        }
        return -sum;
    }
    else if (param_idx == 1) {  // derivative wrt beta
        // sum Σ_i z_i²
        for (unsigned int i = 0; i < particles.size(); i++) {
            sum += sq(particles[i]->getPosition()[2]);
        }
        return -m_parameters[0] * sum;  // -α * sum
    }

    throw std::invalid_argument("ERR: Invalid param_idx requested in EllipticGaussian.");
}

double EllipticGaussian::analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) {
    unsigned int N = particles.size();

    long double sum = 0;
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2) {
                // beta squared * z squared
                sum += sq(m_parameters[1] * particles[i]->getPosition()[j]);
            }
            else {
                sum += sq(particles[i]->getPosition()[j]);
            }
        }
    }

    return -2 * m_parameters[0] * N * (2 + m_parameters[1]) + 4 * sq(m_parameters[0]) * sum;
}