#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents an interacting (repulsive) wave function in an elliptical trap.
 * * Uses a Gaussian single-particle state combined with a Jastrow correlation factor 
 * to model the hard-core repulsion between bosons.
 * Parameters: alpha (xy width), beta (z deformation), and rep_a (hard-core radius).
 */
class FlavoredLJGaussian : public WaveFunction {
public:
    FlavoredLJGaussian(double alpha, double beta1, double beta2);

    double eval(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn(std::vector<std::unique_ptr<class Particle>>& particles);
    
    // special:
    // ∇ᵢ²ln(ψ)
    double analyticalParticleLaplacian2_lnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);
    // ||∇ᵢln(ψ)||²
    double analyticalSqNorm_ParticleGradlnPsi(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);

    std::vector<double> lowerBounds() const override { return { 0.4, 0, 1e-4 }; }
    std::vector<double> upperBounds() const override { return { 50, 2, 10 }; }

    double get_cachedSum_Laplacian2_lnPsi() { return m_cachedSum_Laplacian2_lnPsi; }
    void set_cachedSum_Laplacian2_lnPsi(double new_val) { m_cachedSum_Laplacian2_lnPsi = new_val; }
    double get_cachedSum_SqNorm_GradlnPsi() { return m_cachedSum_SqNorm_GradlnPsi; }
    void set_cachedSum_SqNorm_GradlnPsi(double new_val) { m_cachedSum_SqNorm_GradlnPsi = new_val; }
private:
    double m_cachedSum_Laplacian2_lnPsi = 0;
    double m_cachedSum_SqNorm_GradlnPsi = 0;
    bool hasAnalyticalDerivatives() const override { return true; }

    double analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) override;
    double analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) override;
    double analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>&) override;

    double evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
};
