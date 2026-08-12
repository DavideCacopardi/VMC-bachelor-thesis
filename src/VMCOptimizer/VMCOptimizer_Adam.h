#pragma once
#include <cmath>
#include <iostream>
#include <iomanip>

#include "VMCOptimizer.h"
#include "../common.h"

class VMCOptimizer_Adam : public VMCOptimizer {
public:
    VMCOptimizer_Adam(
        const runConfig& cfg,
        MCEngine& engine,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    std::vector<double> optimize(std::vector<double> initialParams, const std::vector<bool>& optimize_mask = {}) override;

private:
    double m_lr = 0.001;
    const double c_beta1 = 0.9;
    const double c_beta2 = 0.999;
    const double c_epsilon = 1e-8;

    bool checkPlateau(
        double current_val, double& best_val, unsigned int& patience_counter, double min_improvement
    );
};