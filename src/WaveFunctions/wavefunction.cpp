#include <memory>
#include <cmath>
#include <cassert>
#include <iostream>
#include <functional>

#include "../common.h"
#include "wavefunction.h"
#include "../system.h"
#include "../Particles/particle.h"

using namespace CommonUtils;

bool WaveFunction::s_useAnalytical = true;

double WaveFunction::spatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim) {
    if (s_useAnalytical && hasAnalyticalDerivatives()) {
        return analyticalSpatialDerivativeLn(particles, particle_idx, dim);
    }

    double h = 1e-4 * std::max(1.0, std::abs(particles[particle_idx]->getPosition()[dim]));
    
    particles[particle_idx]->adjustPosition(h, dim);
    double lnPlus = evalLn(particles);
    
    particles[particle_idx]->adjustPosition(-2.0 * h, dim);
    double lnMinus = evalLn(particles);
    
    particles[particle_idx]->adjustPosition(h, dim); // Restore
    
    return (lnPlus - lnMinus) / (2.0 * h);
}

std::vector<double> WaveFunction::spatialGradientLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) {
    unsigned int nDim = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> grad(nDim);
    for (unsigned int d = 0; d < nDim; d++) {
        grad[d] = spatialDerivativeLn(particles, particle_idx, d);
    }
    return grad;
}

double WaveFunction::spatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) {
    if (s_useAnalytical && hasAnalyticalDerivatives()) {
        return analyticalSpatialNormalizedLaplacian(particles);
    }

    double h = 1e-4 * std::max(1.0, std::abs(particles[0]->getPosition()[0]));
    double wfCurrent = eval(particles);
    double sum = 0;

    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int d = 0; d < particles[0]->getNumberOfDimensions(); d++) {
            particles[i]->adjustPosition(h, d);
            double wfPlus = eval(particles);
            
            particles[i]->adjustPosition(-2.0 * h, d);
            double wfMinus = eval(particles);
            
            particles[i]->adjustPosition(h, d); // Restore

            sum += (wfPlus - 2.0 * wfCurrent + wfMinus) / (sq(h) * wfCurrent);
        }
    }
    
    return sum;
}

double WaveFunction::paramDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx) {
    if (s_useAnalytical && hasAnalyticalDerivatives()) {
        return analyticalParamDerivativeLnAbs(particles, param_idx);
    }

    double h = 1e-4 * std::max(1.0, std::abs(m_parameters[param_idx]));
    
    m_parameters[param_idx] += h;
    double lnPlus = evalLn(particles);
    
    m_parameters[param_idx] -= 2.0 * h;
    double lnMinus = evalLn(particles);
    
    m_parameters[param_idx] += h; // Restore
    
    return (lnPlus - lnMinus) / (2.0 * h);
}

std::vector<double> WaveFunction::paramGradientLnAbs(std::vector<std::unique_ptr<Particle>>& particles) {
    std::vector<double> grad(m_numberOfParameters);
    for (int i = 0; i < m_numberOfParameters; i++) {
        grad[i] = paramDerivativeLnAbs(particles, i);
    }
    return grad;
}