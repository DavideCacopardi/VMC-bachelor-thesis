#pragma once

#include <memory>

#include "wavefunction.h"

class EllipticGaussian : public WaveFunction {
    /* assumes 3D case */
public:
    EllipticGaussian(double alpha, double beta);
    double eval(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn(std::vector<std::unique_ptr<class Particle>>& particles);
    
    std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }
private:
    const unsigned int m_NDIM = 3;
    bool hasAnalyticalDerivatives() const override { return true; }

    double analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) override;
    double analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) override;
    double analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>&) override;
};
