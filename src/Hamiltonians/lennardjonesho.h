#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class LennardJonesHO : public Hamiltonian {
public:
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions = true);

    // --- Configuration ---
    static void set_loc_Ken_method(unsigned int method) { s_loc_Ken_method = method; }
    static unsigned int get_loc_Ken_method() { return s_loc_Ken_method; }

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    double computeLocalKineticEnergy(
        WaveFunction& waveFunction,
        std::vector<std::unique_ptr<Particle>>& particles,
        unsigned int method = s_loc_Ken_method,
        bool use_cached_result = false);
private:
    double m_omega;
    double m_sigma; // sigma_AA
    double m_enEps; // enEps_AA
    double m_alpha; // alpha = enEps_AB / enEps_AA
    static unsigned int s_loc_Ken_method;

    bool m_activate_interactions;

    const double c_eps = 1e-9; // prevents numerical errors
};

