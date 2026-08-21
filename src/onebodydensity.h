#pragma once

#include <vector>
#include <fstream>
#include "WaveFunctions/wavefunction.h"
#include "Particles/particle.h"
#include "mcengine.h"
#include "Math/random.h"

std::vector<std::pair<double, double>> computeOnebodyDensity(
    MCEngine& engine,
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    bool normalize_by_nParticles,
    unsigned int numberOfParticleLogs,
    std::ofstream* densitiesOut = nullptr,
    std::ofstream* particlesOut = nullptr);