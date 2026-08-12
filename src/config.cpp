#include "config.h"
#include "json.hpp" // The nlohmann/json header
#include <fstream>
#include <iostream>
#include <limits>

using json = nlohmann::json;

runConfig loadConfig(const std::string& filepath) {
    runConfig cfg;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filepath << "!\n";
        std::cerr << "Using default hardcoded parameters.\n";
        return cfg; 
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << "\n";
        return cfg;
    }

    // --- [ SYSTEM & MODELS ] ---
    if (j.contains("system")) {
        cfg.hamiltonianType       = j["system"].value("hamiltonianType", cfg.hamiltonianType);
        cfg.waveFunctionTrainType = j["system"].value("waveFunctionTrainType", cfg.waveFunctionTrainType);
        cfg.waveFunctionType      = j["system"].value("waveFunctionType", cfg.waveFunctionType);
        cfg.solverType            = j["system"].value("solverType", cfg.solverType);
        cfg.preferAnalytic        = j["system"].value("preferAnalytic", cfg.preferAnalytic);
        cfg.numberOfThreads       = j["system"].value("numberOfThreads", cfg.numberOfThreads);
    }

    // --- [ PHYSICAL PARAMETERS ] ---
    if (j.contains("physics")) {
        cfg.numberOfDimensions          = j["physics"].value("dimensions", cfg.numberOfDimensions);
        cfg.numberOfParticles           = j["physics"].value("particles", cfg.numberOfParticles);
        cfg.min_dist                    = j["physics"].value("min_dist", cfg.min_dist);
        cfg.max_radius                  = j["physics"].value("max_radius", cfg.max_radius);
        cfg.omega                       = j["physics"].value("omega", cfg.omega);
        cfg.omega_z                     = j["physics"].value("omega_z", cfg.omega_z);
        cfg.LJsigma                     = j["physics"].value("LJsigma", cfg.LJsigma);
        cfg.LJenEps                     = j["physics"].value("LJenEps", cfg.LJenEps);
        cfg.LJalpha                     = j["physics"].value("LJalpha", cfg.LJalpha);
        cfg.LJGaussian_loc_Ken_method   = j["physics"].value("LJGaussian_loc_Ken_method", cfg.LJGaussian_loc_Ken_method);
        cfg.use_jsonParams              = j["physics"].value("use_jsonParams", cfg.use_jsonParams);
        cfg.repulsive_a_factor          = j["physics"].value("repulsive_a_factor", cfg.repulsive_a_factor);
        cfg.maxStrength                 = j["physics"].value("maxStrength", cfg.maxStrength);

        // Special handling for the "inf" string
        if (j["physics"].contains("repulsive_strength")) {
            if (j["physics"]["repulsive_strength"] == "inf") {
                cfg.repulsive_strength = std::numeric_limits<double>::infinity();
            } else {
                cfg.repulsive_strength = j["physics"]["repulsive_strength"];
            }
        }

        // Parse the initial parameters vector directly
        if (j["physics"].contains("initialParams")) {
            cfg.initialParams = j["physics"]["initialParams"].get<std::vector<double>>();
        }
        if (j["physics"].contains("optimizeParams_mask")) {
            cfg.optParams_mask = j["physics"]["optimizeParams_mask"].get<std::vector<bool>>();
        }
        if (j["physics"].contains("jsonParams")) {
            cfg.jsonParams = j["physics"]["jsonParams"].get<std::vector<double>>();
        }
    }

    // --- [ MONTE CARLO ] ---
    if (j.contains("monte_carlo")) {
        cfg.timeStep           = j["monte_carlo"].value("timeStep", cfg.timeStep);
        cfg.equilibrationSteps = j["monte_carlo"].value("equilibrationSteps", cfg.equilibrationSteps);
        cfg.metropolisSteps    = j["monte_carlo"].value("metropolisSteps", cfg.metropolisSteps);
        cfg.finalMClog2steps   = j["monte_carlo"].value("finalMClog2steps", cfg.finalMClog2steps);
        cfg.BFGS_tol           = j["monte_carlo"].value("BFGS_tol", cfg.BFGS_tol);
        cfg.BFGS_VarOpt_weight = j["monte_carlo"].value("BFGS_VarOpt_weight", cfg.BFGS_VarOpt_weight);
        cfg.LJ_request_Ekin    = j["monte_carlo"].value("LJ_request_Ekin", cfg.LJ_request_Ekin);
    }

    // --- [ PARAMETER MESH ] ---
    if (j.contains("parameter_mesh")) {
        cfg.mesh_MClog2steps       = j["parameter_mesh"].value("mesh_MClog2steps", cfg.mesh_MClog2steps);
        if (j["parameter_mesh"].contains("mesh_lb")) {
            cfg.mesh_lb = j["parameter_mesh"]["mesh_lb"].get<std::vector<double>>();
        }
        if (j["parameter_mesh"].contains("mesh_ub")) {
            cfg.mesh_ub = j["parameter_mesh"]["mesh_ub"].get<std::vector<double>>();
        }
        if (j["parameter_mesh"].contains("mesh_nPoints")) {
            cfg.mesh_nPoints = j["parameter_mesh"]["mesh_nPoints"].get<std::vector<unsigned int>>();
        }
    }

    // --- [ NEURAL NETWORK ] ---
    if (j.contains("neural_network")) {
        cfg.Nhid                   = j["neural_network"].value("Nhid", cfg.Nhid);
        cfg.activationFunctionType = j["neural_network"].value("activationFunctionType", cfg.activationFunctionType);
        cfg.helpDecay              = j["neural_network"].value("helpDecay", cfg.helpDecay);
        cfg.lr                     = j["neural_network"].value("lr", cfg.lr);
        cfg.Adam_ktol              = j["neural_network"].value("Adam_ktol", cfg.Adam_ktol);
        cfg.max_patience           = j["neural_network"].value("max_patience", cfg.max_patience);
        cfg.min_improvement        = j["neural_network"].value("min_improvement", cfg.min_improvement);
        cfg.nPretrainSteps         = j["neural_network"].value("nPretrainSteps", cfg.nPretrainSteps);
        cfg.nEnergySteps           = j["neural_network"].value("nEnergySteps", cfg.nEnergySteps);
        cfg.nAdiabSteps            = j["neural_network"].value("nAdiabSteps", cfg.nAdiabSteps);
    }

    // --- [ OBSERVABLES ] ---
    if (j.contains("observables")) {
        cfg.onebodyDensitySteps  = j["observables"].value("onebodyDensitySteps", cfg.onebodyDensitySteps);
        cfg.onebodyDensity_rMax  = j["observables"].value("onebodyDensity_rMax", cfg.onebodyDensity_rMax);
        cfg.onebodyDensity_nBins = j["observables"].value("onebodyDensity_nBins", cfg.onebodyDensity_nBins);
    }

    // --- [ MISC ] ---
    if (j.contains("misc")) {
        cfg.seed = j["misc"].value("seed", cfg.seed);
    }

    return cfg;
}