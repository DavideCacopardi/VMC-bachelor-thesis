#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

/**
 * @brief Kinetic energy + Lennard-Jones potential + Harmonic potential.
 * H = -hbar^2 / (2m) Laplacian + 1/2 m omega^2 r_i^2 + 4 enEps [alpha] ((sigma/r_{ij})^12 - (sigma/r_{ij})^6)
 * alpha is only present in terms relative to pair of particles different in flavor.
 */
class LennardJonesHO : public Hamiltonian {
public:
    /**
     * @brief Construct a new Lennard Jones H O object
     * 
     * @param omega Pulsation of the harmonic potential.
     * @param sigma Sigma parameter of the Lennard-Jones potential.
     * @param enEps Epsilon parameter of the Lennard-Jones potential.
     * @param alpha Alpha parameter of the Lennard-Jones potential.
     * @param activate_interactions Toggles the Lennard-Jones potential.
     * @param maxStrength Multiplicative factor for the Lennard-Jones potential (usually set to 1).
     * @param percStrength Percentual of the maxStrength parameter to express.
     */
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength, double percStrength);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions, double maxStrength);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha, bool activate_interactions);
    LennardJonesHO(double omega, double sigma, double enEps, double alpha);

    // --- Configuration ---
    static void set_loc_Ken_method(unsigned int method) { s_loc_Ken_method = method; }
    static unsigned int get_loc_Ken_method() { return s_loc_Ken_method; }

    /**
     * @brief Computes the total local energy.
     * 
     * @param waveFunction Wavefunction.
     * @param particles Particles.
     * @return double total local energy
     */
    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    /**
     * @brief Computes the total local energy.
     * 
     * @param waveFunction Wavefunction.
     * @param particles Particles.
     * @return std::vector<double> idx 0 contains the total local energy,
     * idx 1 contains the local kinetic energy,
     * idx 2 contains the local harmonic potential,
     * idx 3 contains the 'alike' LJ potential terms,
     * idx 4 contains the 'unlike' LJ potential terms
     */
    std::vector<double> computeLocalEnergies(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    /**
     * @brief Computes the local kinetic energy term in a selected
     * method and/or exploiting previously cached results.
     * WRN: the method selection works only with LJGaussian.
     * @param waveFunction Wavefunction
     * @param particles Particles
     * @param method 0 selects the default normalized laplacian method implemented by the wavefunction,
     * 1 selects the ∇ᵢ²ln(ψ) method, 2 selects the ||∇ᵢln(ψ)||² method.
     * @param use_cached_result Toggles the use of cached results.
     * @return double local kinetic energy
     */
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

    double localHarmonicPotentialEnergy(
        std::vector<std::unique_ptr<class Particle>>& particles);

    double localLennardJonesPotentialEnergy(
        std::vector<std::unique_ptr<class Particle>>& particles);
    
    double localLennardJonesAlikePotentialEnergy(
        std::vector<std::unique_ptr<class Particle>>& particles);
    double localLennardJonesUnlikePotentialEnergy(
        std::vector<std::unique_ptr<class Particle>>& particles);

    const double c_eps = 1e-9; // prevents numerical errors
};

