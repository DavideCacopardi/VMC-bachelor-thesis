#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <limits>

#include "mcengine.h"
#include "../config.h"

class VMCOptimizer {
public:
    VMCOptimizer(
        const runConfig& cfg,
        MCEngine& engine,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    virtual ~VMCOptimizer() = default;

    virtual std::vector<double> optimize(std::vector<double> initialParams, const std::vector<bool>& optimize_mask = {}) = 0;

protected:
    double computeMC(const std::vector<double>& params, std::vector<double>& grad);

    runConfig m_cfg;
    MCEngine& m_engine;
    std::ofstream* m_logfile;
    std::ofstream* m_outfile;
    std::ofstream* m_paramsfile;
    
    unsigned int m_mcCount = 0;
    double m_previousObjVal = 0.0;
    double m_bestObjective = std::numeric_limits<double>::infinity();
    std::vector<double> m_bestParams;
    const unsigned int c_wait = 10;
    const unsigned int c_max_improvement_tries = 5;
};