#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents a simple, isotropic Gaussian wave function.
 * * Used for non-interacting bosons in a perfectly spherical trap.
 * Contains a single variational parameter, alpha.
 * Equation: Psi = exp(-alpha * sum(r_i^2)).
 */
class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(double alpha);
    double eval(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn(std::vector<std::unique_ptr<class Particle>>& particles);    

    std::vector<double> lowerBounds() const override { return { 1e-3 }; }
    std::vector<double> upperBounds() const override { return { 1.0 }; }
private:
    bool hasAnalyticalDerivatives() const override { return true; }

    double analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) override;
    double analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) override;
    double analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>&) override;
};
 