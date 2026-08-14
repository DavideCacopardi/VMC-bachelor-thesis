#pragma once

#include <memory>

#include "metropolis.h"

class SwappingMetropolis : public Metropolis {
public:
    SwappingMetropolis(std::unique_ptr<class Random> rng);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
};
