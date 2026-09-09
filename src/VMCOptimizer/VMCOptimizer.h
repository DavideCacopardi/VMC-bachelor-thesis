#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <limits>

#include "mcengine.h"
#include "../config.h"

/**
 * @brief Abstract base class for VMC Energy loss function automated optimizers. 
 */
class VMCOptimizer {
public:
    /**
     * @brief Construct a new VMCOptimizer object.
     * @param cfg Constant reference to the runConfig configuration struct to access user-tuned parameters.
     * @param engine Monte Carlo Engine to use.
     * @param logfile Log file.
     * @param outfile Output file.
     * @param paramsfile .dat file where to print the final parameters.
     */
    VMCOptimizer(
        const runConfig& cfg,
        MCEngine& engine,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );
    virtual ~VMCOptimizer() = default;

    /**
     * @brief Run the optimization starting from an initial paramter guess.
     * @param initialParams Initial parameter guess.
     * @param optimize_mask Mask to toggle the optimization of some variational parameters.
     * @return Vector containing the optimal parameters.
     */
    virtual std::vector<double> optimize(std::vector<double> initialParams, const std::vector<bool>& optimize_mask = {}) = 0;

protected:
    /**
     * @brief Wrapper of the MCEngine's run method.
     * It also implements Signal-to-Noise Ratio increment of the number of optimization MC steps.
     *
     * @param params Parameters for the wavefunction.
     * @param grad Graident of the loss function wrt parameters to load.
     * @return Evaluated loss function value. 
     */
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
    const unsigned int c_max_improvement_tries = 5;
};