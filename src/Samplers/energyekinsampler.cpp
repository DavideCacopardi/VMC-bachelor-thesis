#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "energyekinsampler.h"
#include "Particles/particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/lennardjonesho.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/ljgaussian.h"

using namespace CommonUtils;

EnergyEkinSampler::EnergyEkinSampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    bool log_grads
) : EnergySampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps,
    log_grads
) {}

EnergyEkinSampler::EnergyEkinSampler(const std::vector<std::unique_ptr<EnergyEkinSampler>>& others)
    : EnergySampler(others[0]->m_numberOfParticles,
        others[0]->m_numberOfDimensions,
        others[0]->m_numberOfParameters,
        others[0]->m_stepLength,
        0,
        others[0]->m_log_grads)
{
    for (unsigned int i = 0; i < others.size(); i++) {
        this->mergeBaseData(others[i].get());
        m_cumulativeEkin1 += others[i]->m_cumulativeEkin1;
        m_cumulativeEkin2 += others[i]->m_cumulativeEkin2;
        m_cumulativeEkin1SQ += others[i]->m_cumulativeEkin1SQ;
        m_cumulativeEkin2SQ += others[i]->m_cumulativeEkin2SQ;
    }
    computeAverages();
}

void EnergyEkinSampler::sample(bool acceptedStep, System* system, std::vector<double>* energiesOut) {
    EnergySampler::sample(acceptedStep, system, energiesOut);   // computeLocalEnergy loads wf's cache

    auto* ljHam = dynamic_cast<LennardJonesHO*>(&system->getHamiltonian());
    if (ljHam == nullptr) {
        throw std::invalid_argument("ERR: EnergyEkinSampler requires an LJHamiltonian!");
    }
    double currEkin1 = ljHam->computeLocalKineticEnergy(
        system->getWaveFunction(),
        system->getParticles(),
        1,
        LennardJonesHO::get_loc_Ken_method() == 2 ? false : true); // if available, take advantage of cache
    double currEkin2 = ljHam->computeLocalKineticEnergy(
        system->getWaveFunction(),
        system->getParticles(),
        2,
        LennardJonesHO::get_loc_Ken_method() == 1 ? false : true); // if available, take advantage of cache
    m_cumulativeEkin1 += currEkin1;
    m_cumulativeEkin2 += currEkin2;
    m_cumulativeEkin1SQ += sq(currEkin1);
    m_cumulativeEkin2SQ += sq(currEkin2);
}

void EnergyEkinSampler::logHeader(const std::vector<double>& params, std::ofstream& outs) {
    for (unsigned int i = 0; i < params.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "]";
        print_colTitle(outs, temp, i == 0, false);
    }
    print_colTitle(outs, "energy");
    print_colTitle(outs, "variance");
    print_colTitle(outs, "error");
    print_colTitle(outs, "      -(ℏ/4)⟨∇ᵢ²ln(ψ)⟩");
    print_colTitle(outs, "  Var -(ℏ/4)⟨∇ᵢ²ln(ψ)⟩");
    print_colTitle(outs, "    (ℏ/2)⟨|∇ᵢln(ψ)|²⟩");
    print_colTitle(outs, " Var (ℏ/2)⟨|∇ᵢln(ψ)|²⟩");
    if (m_log_grads) {
        for (unsigned int i = 0; i < params.size(); i++) {
            std::string temp = "gradE[" + std::to_string(i) + "]";
            print_colTitle(outs, temp);
            temp = "gradVarE[" + std::to_string(i) + "]";
            print_colTitle(outs, temp);
        }
    }
    // outs << std::setw(width) << "      -(ℏ/4)⟨∇ᵢ²ln(ψ)⟩,"
        // << std::setw(width) << "  Var -(ℏ/4)⟨∇ᵢ²ln(ψ)⟩,"
        // << std::setw(width) << "    (ℏ/2)⟨|∇ᵢln(ψ)|²⟩,"
        // << std::setw(width) << " Var (ℏ/2)⟨|∇ᵢln(ψ)|²⟩,";
    print_colTitle(outs, "elapsed time");
    print_colTitle(outs, "accept. ratio");
    print_colTitle(outs, "MC steps", false, true);
}

void EnergyEkinSampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    const unsigned int prec = 14, width = 21;
    outs << std::scientific;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        print_colVal(outs, params[i], i==0, false);
    }
    print_colVal(outs, m_energy);
    print_colVal(outs, m_variance);
    print_colVal(outs, m_error);
    print_colVal(outs, m_Ekin1);
    print_colVal(outs, m_Ekin1_variance);
    print_colVal(outs, m_Ekin2);
    print_colVal(outs, m_Ekin2_variance);
    if (m_log_grads) {
        auto gradE = get_dEdW();
        auto gradVarE = get_dVardW();
        for (unsigned int i = 0; i < params.size(); i++) {
            print_colVal(outs, gradE[i]);
            print_colVal(outs, gradVarE[i]);
        }
    }
    print_colVal(outs, m_elapsedTime.count());
    print_colVal(outs, ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps));
    print_colVal(outs, m_numberOfMetropolisSteps, false, true);
    outs << std::flush;
}

void EnergyEkinSampler::computeAverages() {
    EnergySampler::computeAverages();
    m_Ekin1 = m_cumulativeEkin1 / (double)m_numberOfMetropolisSteps;
    m_Ekin1SQ = m_cumulativeEkin1SQ / (double)m_numberOfMetropolisSteps;
    m_Ekin2 = m_cumulativeEkin2 / (double)m_numberOfMetropolisSteps;
    m_Ekin2SQ = m_cumulativeEkin2SQ / (double)m_numberOfMetropolisSteps;
    m_Ekin1_variance = m_Ekin1SQ - sq(m_Ekin1);
    m_Ekin2_variance = m_Ekin2SQ - sq(m_Ekin2);
}