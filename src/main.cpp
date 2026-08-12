#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <armadillo>
#include <nlopt.hpp>

#include "system.h"
#include "common.h"
#include "config.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/ljgaussian.h"
#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/ljgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/coulombho.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/lennardjonesho.h"
#include "Hamiltonians/repulsiveho.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "Solvers/swappingmh.h"
#include "Math/random.h"
#include "Math/blocker.h"
#include "Particles/particle.h"
#include "onebodydensity.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "VMCOptimizer/VMCOptimizer.h"
#include "VMCOptimizer/VMCOptimizer_NLOPT.h"
#include "VMCOptimizer/VMCOptimizer_Adam.h"
#include "VMCOptimizer/VMCOptimizer_NN.h"

using namespace std;
using namespace CommonUtils;

using HamiltonianFactory = function<unique_ptr<class Hamiltonian>()>;
using WaveFunctionFactory = function<unique_ptr<class WaveFunction>(const vector<double>&)>;
using SolverFactory = function<unique_ptr<MonteCarlo>(unique_ptr<Random>)>;
using ActivationFunc = std::function<torch::Tensor(const torch::Tensor&)>;
using ActivationFuncFactory = std::function<ActivationFunc()>;

void printLogHeader(const runConfig& cfg, const vector<bool>& toggles, std::ofstream& globalLog, tm* now_tm) {
    globalLog << "=========================================\n";
    globalLog << "               VMC RUN LOG               \n";
    globalLog << "=========================================\n";
    globalLog << "Date and time             : " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "\n";
    globalLog << "-----------------------------------------\n";
    globalLog << "[ SYSTEM & MODELS ]\n";
    globalLog << "Hamiltonian               : " << cfg.hamiltonianType << "\n";
    globalLog << "WF Train Type             : " << cfg.waveFunctionTrainType << "\n";
    globalLog << "WaveFunction              : " << cfg.waveFunctionType << "\n";
    globalLog << "Solver                    : " << cfg.solverType << "\n";
    globalLog << "preferAnalytic            : " << (cfg.preferAnalytic ? "true" : "false") << "\n";
    globalLog << "-----------------------------------------\n";
    globalLog << "[ PHYSICAL PARAMETERS ]\n";
    globalLog << "dimensions (D)            : " << cfg.numberOfDimensions << "\n";
    globalLog << "particles (N)             : " << cfg.numberOfParticles << "\n";
    globalLog << "min_dist                  : " << cfg.min_dist << "\n";
    globalLog << "max_radius                : " << cfg.max_radius<< "\n";
    globalLog << "omega                     : " << cfg.omega << "\n";
    globalLog << "omega_z                   : " << cfg.omega_z << "\n";
    globalLog << "LJsigma                   : " << cfg.LJsigma << "\n";
    globalLog << "LJenEps                   : " << cfg.LJenEps << "\n";
    globalLog << "LJalpha                   : " << cfg.LJalpha << "\n";
    globalLog << "LJGaussian_loc_Ken_method : " << cfg.LJGaussian_loc_Ken_method << "\n";
    globalLog << "rep_a_factor              : " << cfg.repulsive_a_factor << "\n";
    globalLog << "rep_strength              : " << cfg.repulsive_strength << "\n";
    globalLog << "maxStrength               : " << cfg.maxStrength << "\n";
    globalLog << "pre-optimization params   : [ " << setprecision(9);
    for (unsigned int i = 0; i < cfg.initialParams.size(); i++) {
        globalLog << cfg.initialParams[i];
        globalLog << ((i + 1 == cfg.initialParams.size()) ? " ]\n" : ", ");
    }
    globalLog << "param-optimization mask   : [ ";
    for (unsigned int i = 0; i < cfg.optParams_mask.size(); i++) {
        globalLog << (cfg.optParams_mask[i] ? "true" : "false");
        globalLog << ((i + 1 == cfg.optParams_mask.size()) ? " ]\n" : ", ");
    }
    globalLog << "use_jsonParams            : " << (cfg.use_jsonParams ? "true" : "false") << "\n";
    globalLog << "json params               : [ " << setprecision(9);
    for (unsigned int i = 0; i < cfg.jsonParams.size(); i++) {
        globalLog << cfg.jsonParams[i];
        globalLog << ((i + 1 == cfg.jsonParams.size()) ? " ]\n" : ", ");
    }
    globalLog << "-----------------------------------------\n";
    globalLog << "[ MONTE CARLO & OPTIMIZATION ENGINES ]\n";
    globalLog << "optimizer                 : " << cfg.optimizer << "\n";
    globalLog << "time Step                 : " << cfg.timeStep << "\n";
    globalLog << "Equilibr. Steps           : " << cfg.equilibrationSteps << "\n";
    globalLog << "optimizationMCsteps       : " << cfg.metropolisSteps << "\n";
    globalLog << "BFGS_tol                  : " << cfg.BFGS_tol << "\n";
    globalLog << "Adam_lr                   : " << cfg.Adam_lr << "\n";
    globalLog << "Adam_nSteps               : " << cfg.Adam_nSteps << "\n";
    globalLog << "Adam_min_improvement      : " << cfg.Adam_min_improvement << "\n";
    globalLog << "Adam_max_patience         : " << cfg.Adam_max_patience << "\n";
    globalLog << "varOpt_weight             : " << cfg.varOpt_weight << "\n";
    globalLog << "Final MC Steps            : 2^" << cfg.finalMClog2steps << " (" << std::pow(2, cfg.finalMClog2steps) << ")\n";
    globalLog << "-----------------------------------------\n";
    globalLog << "[ PARAMETER MESH ]\n";
    globalLog << "lower bounds              : [ " << setprecision(9);
    for (unsigned int i = 0; i < cfg.mesh_lb.size(); i++) {
        globalLog << cfg.mesh_lb[i];
        globalLog << ((i + 1 == cfg.mesh_lb.size()) ? " ]\n" : ", ");
    }
    globalLog << "upper bounds              : [ " << setprecision(9);
    for (unsigned int i = 0; i < cfg.mesh_ub.size(); i++) {
        globalLog << cfg.mesh_ub[i];
        globalLog << ((i + 1 == cfg.mesh_ub.size()) ? " ]\n" : ", ");
    }
    globalLog << "number of points          : [ ";
    for (unsigned int i = 0; i < cfg.mesh_nPoints.size(); i++) {
        globalLog << cfg.mesh_nPoints[i];
        globalLog << ((i + 1 == cfg.mesh_nPoints.size()) ? " ]\n" : ", ");
    }
    globalLog << "Mesh MC Steps             : 2^" << cfg.mesh_MClog2steps << " (" << std::pow(2, cfg.mesh_MClog2steps) << ")\n";
    globalLog << "-----------------------------------------\n";
    globalLog << "[ NEURAL NETWORK ]\n";
    globalLog << "Nhid                      : " << cfg.Nhid << "\n";
    globalLog << "Activation Func           : " << cfg.activationFunctionType << "\n";
    globalLog << "NN Learning Rate          : " << cfg.NN_lr << "\n";
    globalLog << "Pretrain Steps            : " << cfg.nPretrainSteps << "\n";
    globalLog << "Energy Steps              : " << cfg.nEnergySteps << "\n";
    globalLog << "Adiab Steps               : " << cfg.nAdiabSteps << "\n";
    globalLog << "Adam_ktol                 : " << cfg.Adam_ktol << "\n";
    globalLog << "max_patience              : " << cfg.max_patience << "\n";
    globalLog << "min_improvement           : " << cfg.min_improvement << "\n";
    globalLog << "helpDecay                 : " << cfg.helpDecay << "\n";
    globalLog << "-----------------------------------------\n";
    globalLog << "[ OBSERVABLES & MISC ]\n";
    globalLog << "1bodyDens. Steps          : " << cfg.onebodyDensitySteps << "\n";
    globalLog << "1bodyDens. rMax           : " << cfg.onebodyDensity_rMax << "\n";
    globalLog << "1bodyDens. nBins          : " << cfg.onebodyDensity_nBins << "\n";
    globalLog << "Seed                      : " << cfg.seed << "\n";
    globalLog << "=========================================\n";
    globalLog << "Called toggles            : [ ";
    for (unsigned int i = 0; i < toggles.size(); i++) {
        globalLog << (toggles[i] ? "true" : "false");
        globalLog << ((i + 1 == toggles.size()) ? " ]\n" : ", ");
    }
    globalLog << "=========================================\n\n";
    globalLog << std::flush;
}

int main(int argc, char* argv[]) {
    runConfig cfg = loadConfig("config.json");

    chrono::high_resolution_clock::time_point watch_start, watch_end;
    chrono::duration<double> elapsedTime;

    // --- Toggles based on argc, argv ---
    vector<bool> toggles(4, false);
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            int temp = atoi(argv[i]);
            if (0 < temp && temp <= (int)toggles.size()) {
                toggles[temp - 1] = true;
            }
        }
    }
    else {
        toggles.assign(toggles.size(), true);
    }

    // --- Global Log file setup ---
    string toLogStr = "";
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm* now_tm = localtime(&now_time);
    ostringstream filenameStream;
    filenameStream << "./logs/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".log";
    ofstream globalLog(filenameStream.str());
    if (!globalLog.is_open()) {
        cerr << "Error: unable to generate log file " << filenameStream.str() << endl;
        return 1;
    }
    printLogHeader(cfg, toggles, globalLog, now_tm);

    // --- Setup Classes and Factories ---
    WaveFunction::setUseAnalyticalDerivatives(cfg.preferAnalytic);
    if (!cfg.preferAnalytic && cfg.LJGaussian_loc_Ken_method != 0) {
        toLogStr = "\n WRN: the selected LJGaussian_loc_Ken_method has no numerical implementation (preferAnalytic is set to false).\n";
        globalLog << toLogStr;
        cout << toLogStr;
    }
    LennardJonesHO::set_loc_Ken_method(cfg.LJGaussian_loc_Ken_method);
    Particle::set_min_dist(cfg.repulsive_a_factor > cfg.min_dist ? cfg.repulsive_a_factor : cfg.min_dist);
    Particle::set_max_radius(Particle::s_min_dist > cfg.max_radius ? Particle::s_min_dist : cfg.max_radius);
    HamiltonianFactory hFac = [=]() -> unique_ptr<Hamiltonian> {
        if (cfg.hamiltonianType == "HarmonicOscillator") {
            return make_unique<HarmonicOscillator>(cfg.omega);
        }
        else if (cfg.hamiltonianType == "CoulombHO") {
            return make_unique<CoulombHO>(cfg.omega, cfg.omega_z, cfg.maxStrength);
        }
        else if (cfg.hamiltonianType == "LennardJonesHO") {
            return make_unique<LennardJonesHO>(cfg.omega, cfg.LJsigma, cfg.LJenEps, cfg.LJalpha);
        }
        else if (cfg.hamiltonianType == "LennardJonesHO_noInteraction") {
            return make_unique<LennardJonesHO>(cfg.omega, cfg.LJsigma, cfg.LJenEps, cfg.LJalpha, false);
        }
        else { // default to Repulsive
            return make_unique<RepulsiveHO>(cfg.omega, cfg.omega_z, cfg.repulsive_a_factor, cfg.repulsive_strength);
        }
        };
    WaveFunctionFactory wfFac = [=](const vector<double>& p) -> unique_ptr<WaveFunction> {
        if (cfg.waveFunctionType == "SimpleGaussian")
            return make_unique<SimpleGaussian>(p[0]);
        else if (cfg.waveFunctionType == "EllipticGaussian")
            return make_unique<EllipticGaussian>(p[0], p[1]);
        else if (cfg.waveFunctionType == "LJGaussian")
            return make_unique<LJGaussian>(p[0], p[1], p[2]);
        else if (cfg.waveFunctionType == "NN_envelope")
            return make_unique<NN_envelope>(cfg.numberOfParticles,
                cfg.numberOfDimensions,
                cfg.numberOfParticles * cfg.numberOfDimensions,
                cfg.Nhid,
                cfg.helpDecay,
                p
            );
        else // default to Repulsive
            return make_unique<RepEllipticGaussian>(p[0], p[1], cfg.repulsive_a_factor / sqrt(cfg.omega));
        };
    WaveFunctionFactory wfFacTrain = [=](const vector<double>& p) -> unique_ptr<WaveFunction> {
        if (cfg.waveFunctionTrainType == "SimpleGaussian")
            return make_unique<SimpleGaussian>(p[0]);
        else if (cfg.waveFunctionTrainType == "EllipticGaussian")
            return make_unique<EllipticGaussian>(p[0], p[1]);
        else if (cfg.waveFunctionType == "LJGaussian")
            return make_unique<LJGaussian>(p[0], p[1], p[2]);
        else // default to Repulsive
            return make_unique<RepEllipticGaussian>(p[0], p[1], cfg.repulsive_a_factor / sqrt(cfg.omega));
        };
    SolverFactory solverFac = [=](unique_ptr<Random> rng) -> unique_ptr<MonteCarlo> {
        if (cfg.solverType == "Metropolis") {
            return make_unique<Metropolis>(move(rng));
        }
        else if (cfg.solverType == "SwappingMH") {
            return make_unique<SwappingMH>(move(rng));
        }
        else // default to Metropolis-Hastings
            return make_unique<MetropolisHastings>(move(rng));
        };
    ActivationFuncFactory actFun = [=]() -> ActivationFunc {
        if (cfg.activationFunctionType == "gelu") {
            return [](const torch::Tensor& t) { return torch::gelu(t); };
        }
        else if (cfg.activationFunctionType == "relu") {
            return [](const torch::Tensor& t) { return torch::relu(t); };
        }
        else if (cfg.activationFunctionType == "sigmoid") {
            return [](const torch::Tensor& t) { return torch::sigmoid(t); };
        }
        else {
            // default to tanh
            return [](const torch::Tensor& t) { return torch::tanh(t); };
        }
        };

    // --- Actual Code Execution ---
    if (toggles[0] && cfg.waveFunctionType == "NN_envelope") {
        // --- 1a: Neural-Network optimization ---
        globalLog << "Initial parameters for psi_train: " << setprecision(9);
        for (unsigned int i = 0; i < cfg.initialParams.size(); i++) {
            globalLog << cfg.initialParams[i] << ", \t";
        }
        globalLog << endl;

        ostringstream filenameStream;
        filenameStream << "./logs_NN/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream logfile(filenameStream.str());
        if (!logfile.is_open()) {
            cerr << "Error: unable to generate log_NN file " << filenameStream.str() << endl;
            return 1;
        }
        ofstream outfile("./iofiles/details_results.csv");
        ofstream paramsfile("./iofiles/params.dat");
        VMCOptimizer_NN optimizer(
            cfg,
            hFac(),
            solverFac,
            actFun(),
            &logfile, &outfile, &paramsfile
        );

        watch_start = chrono::high_resolution_clock::now();
        vector<double> optimalParams = optimizer.optimize(wfFacTrain(cfg.initialParams));
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;

        cout << "Optimal parameters: " << setprecision(9);
        globalLog << "Optimal parameters: " << setprecision(9);
        for (unsigned int i = 0; i < optimalParams.size(); i++) {
            cout << optimalParams[i] << ", \t";
            globalLog << optimalParams[i] << ", \t";
        }
        cout << "\nNN_VMC optimization done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "\nNN_VMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        logfile.close(); outfile.close(); paramsfile.close();
    }

    // --- Build MC engine ---
    MCEngine engine(
        cfg,
        hFac,
        wfFac,
        solverFac
    );

    if (toggles[0] && cfg.waveFunctionType != "NN_envelope") {
        // --- 1b: Optimization ---
        globalLog << "Initial parameters: " << setprecision(9);
        for (unsigned int i = 0; i < cfg.initialParams.size(); i++) {
            globalLog << cfg.initialParams[i] << ", \t";
        }
        globalLog << endl;

        ostringstream filenameStream;
        filenameStream << "./logs_opt/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream logfile(filenameStream.str());
        if (!logfile.is_open()) {
            cerr << "Error: unable to generate log_NN file " << filenameStream.str() << endl;
            return 1;
        }
        ofstream outfile("./iofiles/detailed_nlopt_results.csv");
        ofstream paramsfile("./iofiles/params.dat");

        unique_ptr<VMCOptimizer> optimizer;
        if (cfg.optimizer == "Adam") {
            optimizer = make_unique<VMCOptimizer_Adam>(cfg, engine, &logfile, &outfile, &paramsfile);
        }
        else {  // default to NLOPT
            optimizer = make_unique<VMCOptimizer_NLOPT>(cfg, engine, &logfile, &outfile, &paramsfile);
        }
        
        watch_start = chrono::high_resolution_clock::now();
        vector<double> optimalParams = optimizer->optimize(cfg.initialParams, cfg.optParams_mask);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;

        cout << "\nOptimal parameters: " << setprecision(9);
        globalLog << "Optimal parameters: " << setprecision(9);
        for (unsigned int i = 0; i < optimalParams.size(); i++) {
            cout << optimalParams[i] << ", \t";
            globalLog << optimalParams[i] << ", \t";
        }
        cout << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        logfile.close(); outfile.close(); paramsfile.close();
    }

    if (toggles[1]) {
        // --- 2a: Final MC ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = (!toggles[0] && cfg.use_jsonParams) ? cfg.jsonParams : readVector("./iofiles/params.dat");
        std::vector<std::vector<double>> rawEnergiesData;   // later analyzed by blocking algorithm
        cout << "\rComputing Final MC..." << flush;
        unique_ptr<EnergySampler> sampler =
            engine.run(params, (unsigned int)pow(2, cfg.finalMClog2steps), &rawEnergiesData);
        cout << scientific << setprecision(9) << "\rFinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        cout << scientific << setprecision(9) << "Acceptance ratio: " << sampler->getAcceptanceRatio()
            << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "Acceptance ratio: " << sampler->getAcceptanceRatio()
            << endl << defaultfloat;
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";

        // --- 2b: Blocking ---
        if (rawEnergiesData[0].size() % 2 != 0) {   // check viability of the blocking estimate
            toLogStr = "WRN: size of data fed into the blocking algorithm must be a power of 2."
                "\n    This may be caused by the number of threads not being a power of 2 itself."
                "\n    The blocking estimate will not be evaluated.\n\n";
            cout << toLogStr;
            globalLog << toLogStr;
        }
        else {  // The number of MC cycles is a power of 2. Blocking can be calculated.
            watch_start = chrono::high_resolution_clock::now();
            double cumulative_E = 0;
            double cumulative_var = 0;
            for (unsigned int i = 0; i < rawEnergiesData.size(); i++) {
                Blocker block(rawEnergiesData[i]);
                // block.printResults("./iofiles/blocking_results.csv");
                cout << "Blocking_" << i << scientific << setprecision(9) << " energy: " << block.mean
                    << " +- " << block.stdErr << endl << defaultfloat;
                globalLog << "Blocking_" << i << scientific << setprecision(9) << " energy: " << block.mean
                    << " +- " << block.stdErr << endl << defaultfloat;
                cumulative_E += block.mean;
                cumulative_var += sq(block.stdErr);
            }
            cout << scientific << setprecision(9) << "Final blocking energy: "
                << cumulative_E / (double)rawEnergiesData.size()
                << " +- " << sqrt(cumulative_var) / (double)rawEnergiesData.size() << endl << defaultfloat;
            globalLog << scientific << setprecision(9) << "Final blocking energy: "
                << cumulative_E / (double)rawEnergiesData.size()
                << " +- " << sqrt(cumulative_var) / (double)rawEnergiesData.size() << endl << defaultfloat;
        }
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
    }

    if (toggles[2]) {
        // --- 3: One-body density ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = (!toggles[0] && cfg.use_jsonParams) ? cfg.jsonParams : readVector("./iofiles/params.dat");
        ostringstream filenameStream;
        filenameStream << "./logs_OBD/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream densityfile(filenameStream.str());
        vector<pair<double, double>> density = computeOnebodyDensity(
            engine, params, cfg.onebodyDensitySteps, cfg.onebodyDensity_rMax,
            cfg.onebodyDensity_nBins, &densityfile);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        densityfile.close();
    }

    if (toggles[3]) {
        // --- 4: Mesh ---
        watch_start = chrono::high_resolution_clock::now();
        vector<vector<double>> param_mesh = generate_mesh(cfg.mesh_lb, cfg.mesh_ub, cfg.mesh_nPoints);

        // file setup
        ostringstream filenameStream;
        filenameStream << "./parameter_mesh/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream meshfile(filenameStream.str());
        if (!meshfile.is_open()) {
            cerr << "Error: unable to generate meshfile " << filenameStream.str() << endl;
            return 1;
        }
        meshfile << "#";
        unsigned int col_width = 20, col_prec = 12; // print to file in columns
        for (unsigned int i = 0; i < param_mesh[0].size(); i++) {
            std::string temp = "p[" + std::to_string(i) + "],";
            meshfile << std::setw(col_width - (i == 0)) << temp;
        }
        meshfile << std::setw(col_width) << "energy," << std::setw(col_width) << "error" << std::endl;

        for (unsigned int par_idx = 0; par_idx < param_mesh.size(); par_idx++) {
            cout << "\rComputing mesh MC #" << par_idx + 1 << " of " << param_mesh.size() << flush;
            // --- 4a: MC run ---
            std::vector<std::vector<double>> rawEnergiesData;   // later analyzed by blocking algorithm
            unique_ptr<EnergySampler> sampler =
                engine.run(param_mesh[par_idx], (unsigned int)pow(2, cfg.mesh_MClog2steps), &rawEnergiesData);
            double energy = sampler->getEnergy();
            double err = sampler->getError();

            // --- 4b: Blocking ---
            if (rawEnergiesData[0].size() % 2 != 0) {   // check viability of the blocking estimate
                toLogStr = "WRN: size of data fed into the blocking algorithm must be a power of 2."
                    "\n    This may be caused by the number of threads not being a power of 2 itself."
                    "\n    The blocking estimate will not be evaluated.\n\n";
                cout << toLogStr;
                globalLog << toLogStr;
            }
            else {  // The number of MC cycles is a power of 2. Blocking can be calculated.
                double cumulative_var = 0;
                for (unsigned int i = 0; i < rawEnergiesData.size(); i++) {
                    Blocker block(rawEnergiesData[i]);
                    cumulative_var += sq(block.stdErr);
                }
                err = sqrt(cumulative_var) / (double)rawEnergiesData.size();

                // print row
                meshfile << scientific << setprecision(col_prec);
                for (unsigned int i = 0; i < param_mesh[par_idx].size(); i++) {
                    meshfile << setw(col_width-1) << param_mesh[par_idx][i] << ",";
                }
                meshfile << setw(col_width-1) << energy << "," << setw(col_width-1) << err << endl;
            }
        }

        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "\nMesh plot done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "Mesh plot done (in " << elapsedTime.count() << " s).\n\n";
        meshfile.close();
    }

    globalLog << "=========================================\n";
    globalLog.close();

    cout << "Exiting. (Log saved in: " << filenameStream.str() << ")\n";
    return 0;
}