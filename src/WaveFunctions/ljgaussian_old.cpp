// #include <memory>
// #include <cmath>
// #include <stdexcept>
// #include <iostream>
// #include <cassert>
// #include <limits>

// #include "../common.h"
// #include "ljgaussian_old.h"
// #include "wavefunction.h"
// #include "../system.h"
// #include "../Particles/particle.h"

// using namespace CommonUtils;

// LJGaussian::LJGaussian(double alpha, double beta1, double beta2)
//     : WaveFunction(3, { alpha, beta1, beta2 }) {
//     if (alpha <= 0) throw std::invalid_argument("alpha must be positive");
// }

// double LJGaussian::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
//     return exp(evalLn(particles));
// }

// double LJGaussian::evalLn(std::vector<std::unique_ptr<class Particle>>& particles) {
//     return evalLn_noInteraction(particles) + evalLn_onlyInteraction(particles);
// }

// double LJGaussian::evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
//     double sum = 0;
//     for (unsigned int i = 0; i < particles.size(); i++) {
//         sum += sqNorm(particles[i]->getPosition());
//     }
//     return -0.5 * sum / m_parameters[0];
// }

// double LJGaussian::evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
//     double sum = 0;
//     for (unsigned int i = 0; i < particles.size(); i++) {
//         for (unsigned int j = i + 1; j < particles.size(); j++) {
//             sum += pow(m_parameters[1] / distance(particles[i]->getPosition(), particles[j]->getPosition()), m_parameters[2]);
//         }
//     }
//     return -0.5 * sum;
// }

// double LJGaussian::analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim) {
//     double coord = particles[particle_idx]->getPosition()[dim];

//     double sum = 0;
//     for (unsigned int j = 0; j < particles.size(); j++) {
//         if (j == particle_idx) continue;
//         double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
//         sum += (coord - particles[j]->getPosition()[dim]) * pow(dist, -(m_parameters[2] + 2));
//     }

//     return -coord / m_parameters[0] + 0.5 * m_parameters[2] * pow(m_parameters[1], m_parameters[2]) * sum;
// }

// double LJGaussian::analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx) {
//     if (param_idx == 0) {
//         double sum = 0;
//         for (unsigned int i = 0; i < particles.size(); i++) {
//             sum += sqNorm(particles[i]->getPosition());
//         }
//         return sum / (2 * sq(m_parameters[0]));
//     }
//     else if (param_idx == 1) {
//         double sum = 0;
//         for (unsigned int i = 0; i < particles.size(); i++) {
//             for (unsigned int j = i + 1; j < particles.size(); j++) {
//                 double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
//                 sum += pow(m_parameters[1] / dist, m_parameters[2]);
//             }
//         }
//         return -0.5 * m_parameters[2] / m_parameters[1] * sum;
//     }
//     else if (param_idx == 2) {
//         double sum = 0;
//         for (unsigned int i = 0; i < particles.size(); i++) {
//             for (unsigned int j = i + 1; j < particles.size(); j++) {
//                 double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
//                 double logarithm = log(m_parameters[1] / dist);
//                 sum += logarithm * exp(logarithm * m_parameters[2]);
//             }
//         }
//         return -0.5 * sum;
//     }
//     throw std::invalid_argument("ERR: Invalid param_idx requested in LJGaussian.");
// }

// // (∇²ψ)/ψ = sum_i( ∇ᵢ²ln(ψ) + ||∇ᵢln(ψ)||² )
// double LJGaussian::analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) {
//     double sum_Laplacian2_lnPsi = 0;
//     double sum_SqNorm_GradlnPsi = 0;
//     for (unsigned int i = 0; i < particles.size(); i++) {
//         sum_Laplacian2_lnPsi += analyticalParticleLaplacian2_lnPsi(particles, i);
//         sum_SqNorm_GradlnPsi += analyticalSqNorm_ParticleGradlnPsi(particles, i);
//     }
//     m_cachedSum_Laplacian2_lnPsi = sum_Laplacian2_lnPsi;
//     m_cachedSum_SqNorm_GradlnPsi = sum_SqNorm_GradlnPsi;
//     return sum_Laplacian2_lnPsi + sum_SqNorm_GradlnPsi;
// }

// // ∇ᵢ²ln(ψ)
// double LJGaussian::analyticalParticleLaplacian2_lnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) {
//     double nDim = particles[0]->getNumberOfDimensions();
//     double sum = 0;
//     for (unsigned int j = 0; j < particles.size(); j++) {
//         if (j == particle_idx) continue;
//         double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
//         sum += pow(dist, -(m_parameters[2] + 2));
//     }
//     // ∇ᵢ²ln(ψ)
//     return -nDim / m_parameters[0] + 0.5 * m_parameters[2] * (nDim - m_parameters[2] - 2.0)
//         * pow(m_parameters[1], m_parameters[2]) * sum;
// }

// // ||∇ᵢln(ψ)||²
// double LJGaussian::analyticalSqNorm_ParticleGradlnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) {
//     double sqNorm_lnPsi = 0;
//     for (unsigned int d = 0; d < particles[0]->getNumberOfDimensions(); d++) {
//         double sum_d = 0;
//         for (unsigned int j = 0; j < particles.size(); j++) {
//             if (j == particle_idx) continue;
//             double dist = distance(particles[particle_idx]->getPosition(), particles[j]->getPosition());
//             sum_d += (particles[particle_idx]->getPosition()[d] - particles[j]->getPosition()[d]) * pow(dist, -(m_parameters[2] + 2));
//         }
//         sqNorm_lnPsi += sq(-particles[particle_idx]->getPosition()[d] / m_parameters[0] + 0.5
//             * m_parameters[2] * pow(m_parameters[1], m_parameters[2]) * sum_d);
//     }
//     return sqNorm_lnPsi;
// }