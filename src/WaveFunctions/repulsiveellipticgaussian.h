#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents an interacting (repulsive) wave function in an elliptical trap.
 * * Uses a Gaussian single-particle state combined with a Jastrow correlation factor 
 * to model the hard-core repulsion between bosons.
 * Parameters: alpha (xy width), beta (z deformation), and rep_a (hard-core radius).
 */
class RepEllipticGaussian : public WaveFunction {
public:
    RepEllipticGaussian(double alpha, double beta, double rep_a);
    double eval(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn(std::vector<std::unique_ptr<class Particle>>& particles);

    std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }
private:
    double m_rep_a;
    const unsigned int m_NDIM = 3;
    bool hasAnalyticalDerivatives() const override { return false; }

    double evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
    double evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
};
