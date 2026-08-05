#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class LennardJonesHO : public Hamiltonian {
public:
    LennardJonesHO(double omega, double sigma, double enEps, double alpha);

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

private:
    double m_omega;
    double m_sigma; // sigma_AA
    double m_enEps; // enEps_AA
    double m_alpha; // alpha = enEps_AB / enEps_AA

    double m_kinetic_factor;    // kinetic_factor = 0.1 * enEps_AA * sigma_AA

    const double c_eps = 1e-12; // prevents numerical errors
};

