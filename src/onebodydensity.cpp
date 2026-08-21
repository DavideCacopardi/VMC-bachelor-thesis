#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "onebodydensity.h"
#include "common.h"
#include "Math/random.h"
#include "Samplers/densitysampler.h"

using namespace CommonUtils;

std::vector<std::pair<double, double>> computeOnebodyDensity(
    MCEngine& engine,
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    bool normalize_by_nParticles,
    unsigned int numberOfParticleLogs,
    std::ofstream* densitiesOut,
    std::ofstream* particlesOut) {

    std::cout << "Computing one-body density..." << std::flush;

    std::unique_ptr<DensitySampler> sampler = engine.runOnebodyDensity(
        params, numberOfMetropolisSteps, rMax, nBins,
        normalize_by_nParticles, numberOfParticleLogs, particlesOut);

    const std::vector<double>& grid = sampler->getRadialGrid();
    const std::vector<double>& densityProf = sampler->getDensityProfile();
    const std::vector<std::vector<double>>& densityProfFlavor = sampler->getDensityProfileFlavor();
    const std::vector<double>& probabilityProf = sampler->getProbabilityProfile();
    const std::vector<std::vector<double>>& probabilityProfFlavor = sampler->getProbabilityProfileFlavor();
    const std::vector<double>& densityErrProf = sampler->getDensityError();
    const std::vector<std::vector<double>>& densityErrProfFlavor = sampler->getDensityErrorFlavor();
    const std::vector<double>& probabilityErrProf = sampler->getProbabilityError();
    const std::vector<std::vector<double>>& probabilityErrProfFlavor = sampler->getProbabilityErrorFlavor();

    std::vector<std::pair<double, double>> result(nBins);

    const unsigned int prec = 12, width = 19;
    if (densitiesOut) {
        *densitiesOut << "#"
            << std::setw(width-1) << "r" << ","
            << std::setw(width) << "density" << ","
            << std::setw(width) << "density error" << ",";
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            *densitiesOut << std::setw(width - 1) << "density " << (char)('A' + flav) << ","
                << std::setw(width - 1) << "density error " << (char)('A' + flav) << ",";
        }
        *densitiesOut << std::setw(width) << "radial probability"  << ","
            << std::setw(width) << "probability error" << ",";
        for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
            *densitiesOut << std::setw(width - 1) << "radial prob. " << (char)('A' + flav) << ","
                << std::setw(width - 1) << "prob. error " << (char)('A' + flav);
            *densitiesOut << ((flav + 1 == N_FLAVORS) ? "\n" : ",");
        }

        *densitiesOut << std::scientific;
        for (unsigned int i = 0; i < nBins; i++) {
            result[i] = std::make_pair(grid[i], densityProf[i]);
            if (densitiesOut) {
                *densitiesOut << std::setprecision(prec)
                    << std::setw(width) << grid[i] << ","
                    << std::setw(width) << densityProf[i] << ","
                    << std::setw(width) << densityErrProf[i] << ",";
                for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                    *densitiesOut << std::setw(width) << densityProfFlavor[i][flav] << ","
                        << std::setw(width) << densityErrProfFlavor[i][flav] << ",";
                }
                *densitiesOut << std::setw(width) << probabilityProf[i] << ","
                    << std::setw(width) << probabilityErrProf[i] << ",";
                for (unsigned int flav = 0; flav < N_FLAVORS; flav++) {
                    *densitiesOut << std::setw(width) << probabilityProfFlavor[i][flav] << ","
                        << std::setw(width) << probabilityErrProfFlavor[i][flav];
                    *densitiesOut << ((flav + 1 == N_FLAVORS) ? "\n" : ",");
                }
            }
        }
    }

    std::cout << "\r                             " << std::endl;

    return result;
}