#pragma once

#include <memory>

#include "metropolishastings.h"


class SwappingMH : public MetropolisHastings {
public:
    SwappingMH(std::unique_ptr<class Random> rng, bool useUmrigarDrift = false);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    // double get_target_acceptanceRatio() const override { return 0.93; }
};
