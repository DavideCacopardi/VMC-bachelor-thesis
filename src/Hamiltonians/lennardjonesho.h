#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class LennardJonesHO : public Hamiltonian {
public:
    LennardJonesHO(double omega, double sigma, double enEps, double alpha);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength, double percStrength);

    // --- Configuration ---
    static void set_loc_Ken_method(unsigned int method) { s_loc_Ken_method = method; }
    static unsigned int get_loc_Ken_method() { return s_loc_Ken_method; }

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    std::vector<double> computeLocalEnergies(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    double computeLocalKineticEnergy(
        WaveFunction& waveFunction,
        std::vector<std::unique_ptr<Particle>>& particles,
        unsigned int method = s_loc_Ken_method,
        bool use_cached_result = false);

    double get_interaction_strength() override;

    void set_percStrength(double percStrength) override;
private:
    double m_omega;
    double m_sigma; // sigma_AA
    double m_enEps; // enEps_AA
    double m_alpha; // alpha = enEps_AB / enEps_AA
    bool m_activate_interactions;
    double m_maxStrength = 1;
    double m_percStrength = 1;
    static unsigned int s_loc_Ken_method;

    double localHarmonicPotentialEnergy(class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    double localLennardJonesPotentialEnergy(class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    
    double localLennardJonesAlikePotentialEnergy(class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    double localLennardJonesUnlikePotentialEnergy(class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    const double c_eps = 1e-9; // prevents numerical errors
};

