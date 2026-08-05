#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents an interacting (repulsive) wave function in an elliptical trap.
 * * Uses a Gaussian single-particle state combined with a Jastrow correlation factor 
 * to model the hard-core repulsion between bosons.
 * Parameters: alpha (xy width), beta (z deformation), and rep_a (hard-core radius).
 */
class LJGaussian : public WaveFunction {
public:
    LJGaussian(double alpha, double beta1, double beta2);
    double eval(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn(std::vector<std::unique_ptr<class Particle>>& particles);

    std::vector<double> lowerBounds() const override { return { 1e-2, 0.1, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1e2, 5.0, 5.0 }; }
private:
    bool hasAnalyticalDerivatives() const override { return true; }

    double analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) override;
    double analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) override;
    double analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>&) override;

    // special:
    // ∇ᵢ²ln(ψ)
    double analyticalLaplacian2_lnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);
    // ||∇ᵢln(ψ)||²
    double analyticalSqNorm_lnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);

    double evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
};
