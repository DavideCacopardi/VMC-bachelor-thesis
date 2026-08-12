#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <limits>

#include "../common.h"
#include "flavoredljgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../Particles/particle.h"

using namespace CommonUtils;

FlavoredLJGaussian::FlavoredLJGaussian(double alpha, double betaAA, double betaAB)
    : WaveFunction(3, { alpha, betaAA, betaAB }) {
    if (alpha <= 0) throw std::invalid_argument("alpha must be positive");
}

double FlavoredLJGaussian::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
    return exp(evalLn(particles));
}

double FlavoredLJGaussian::evalLn(std::vector<std::unique_ptr<class Particle>>& particles) {
    return evalLn_noInteraction(particles) + evalLn_onlyInteraction(particles);
}

double FlavoredLJGaussian::evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        sum += sqNorm(particles[i]->getPosition());
    }
    return -0.5 * sum / m_parameters[0];
}

double FlavoredLJGaussian::evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        Flavor f_i = particles[i]->getFlavor();
        for (unsigned int j = i + 1; j < particles.size(); j++) {
            Flavor f_j = particles[j]->getFlavor();
            sum += pow5(m_parameters[f_i == f_j ? 1 : 2] / distance(particles[i]->getPosition(), particles[j]->getPosition()));
        }
    }
    return -0.5 * sum;
}

double FlavoredLJGaussian::analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim) {
    double coord = particles[particle_idx]->getPosition()[dim];
    Flavor f = particles[particle_idx]->getFlavor();

    double sum = 0;
    for (unsigned int j = 0; j < particles.size(); j++) {
        if (j == particle_idx) continue;
        double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
        Flavor f_j = particles[j]->getFlavor();
        sum += pow5(m_parameters[f == f_j ? 1 : 2]) * (coord - particles[j]->getPosition()[dim]) * pow(dist, -7);
    }

    return -coord / m_parameters[0] + 0.5 * 5 * sum;
}

double FlavoredLJGaussian::analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx) {
    if (param_idx == 0) {
        double sum = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            sum += sqNorm(particles[i]->getPosition());
        }
        return sum / (2 * sq(m_parameters[0]));
    }
    else if (param_idx == 1) {
        double sum = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            Flavor f_i = particles[i]->getFlavor();
            for (unsigned int j = i + 1; j < particles.size(); j++) {
                Flavor f_j = particles[j]->getFlavor();
                if (f_i != f_j) continue;
                
                double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
                sum += pow5(m_parameters[1] / dist);
            }
        }
        return -0.5 * 5 / m_parameters[1] * sum;
    }
    else if (param_idx == 2) {
        double sum = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            Flavor f_i = particles[i]->getFlavor();
            for (unsigned int j = i + 1; j < particles.size(); j++) {
                Flavor f_j = particles[j]->getFlavor();
                if (f_i == f_j) continue;
                
                double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
                sum += pow5(m_parameters[2] / dist);
            }
        }
        return -0.5 * 5 / m_parameters[2] * sum;
    }

    throw std::invalid_argument("ERR: Invalid param_idx requested in LJGaussian.");
}

// (∇²ψ)/ψ = sum_i( ∇ᵢ²ln(ψ) + ||∇ᵢln(ψ)||² )
double FlavoredLJGaussian::analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) {
    double sum_Laplacian2_lnPsi = 0;
    double sum_SqNorm_GradlnPsi = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        sum_Laplacian2_lnPsi += analyticalParticleLaplacian2_lnPsi(particles, i);
        sum_SqNorm_GradlnPsi += analyticalSqNorm_ParticleGradlnPsi(particles, i);
    }
    m_cachedSum_Laplacian2_lnPsi = sum_Laplacian2_lnPsi;
    m_cachedSum_SqNorm_GradlnPsi = sum_SqNorm_GradlnPsi;
    return sum_Laplacian2_lnPsi + sum_SqNorm_GradlnPsi;
}

// ∇ᵢ²ln(ψ)
double FlavoredLJGaussian::analyticalParticleLaplacian2_lnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) {
    double nDim = particles[0]->getNumberOfDimensions();
    Flavor f = particles[particle_idx]->getFlavor();

    double sum = 0;
    for (unsigned int j = 0; j < particles.size(); j++) {
        if (j == particle_idx) continue;
        Flavor f_j = particles[j]->getFlavor();

        double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
        sum += pow5(m_parameters[f == f_j ? 1 : 2]) * pow(dist, -7);
    }
    // ∇ᵢ²ln(ψ)
    return -nDim / m_parameters[0] + 0.5 * 5 * (nDim - 7) * sum;
}

// ||∇ᵢln(ψ)||²
double FlavoredLJGaussian::analyticalSqNorm_ParticleGradlnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) {
    Flavor f = particles[particle_idx]->getFlavor();

    double sqNorm_lnPsi = 0;
    for (unsigned int d = 0; d < particles[0]->getNumberOfDimensions(); d++) {
        double sum_d = 0;
        for (unsigned int j = 0; j < particles.size(); j++) {
            if (j == particle_idx) continue;
            Flavor f_j = particles[j]->getFlavor();

            double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
            sum_d += pow5(m_parameters[f == f_j ? 1 : 2]) * pow(dist, -7)
                * (particles[particle_idx]->getPosition()[d] - particles[j]->getPosition()[d]);
        }
        sqNorm_lnPsi += sq(-particles[particle_idx]->getPosition()[d] / m_parameters[0] + 0.5 * 5 * sum_d);
    }
    return sqNorm_lnPsi;
}