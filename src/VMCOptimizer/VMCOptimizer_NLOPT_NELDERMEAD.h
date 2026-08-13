#pragma once
#include <nlopt.hpp>

#include "VMCOptimizer.h"
#include "../common.h"

class VMCOptimizer_NLOPT_NELDERMEAD : public VMCOptimizer {
public:
    VMCOptimizer_NLOPT_NELDERMEAD(
        const runConfig& cfg,
        MCEngine& engine,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    std::vector<double> optimize(std::vector<double> initialParams, const std::vector<bool>& optimize_mask = {}) override;

private:
    static double nloptObjective(const std::vector<double>& params, std::vector<double>& grad, void* data);
};