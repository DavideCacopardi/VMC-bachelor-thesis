#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "../Particles/particle.h"

/**
 * @brief Abstract base class for all trial wave functions.
 * * This class defines the interface that any specific wave function must implement.
 * It also provides numerical fallback methods for derivatives and quantum forces
 * in case analytical expressions are too complex to derive.
 */
class WaveFunction {
public:
    virtual ~WaveFunction() = default;

    // --- Configuration ---
    static void setUseAnalyticalDerivatives(bool useAnalytical) { s_useAnalytical = useAnalytical; }
    
    int getNumberOfParameters() const { return m_numberOfParameters; }
    virtual const std::vector<double>& getParameters() { return m_parameters; }
    // for optimization
    virtual std::vector<double> lowerBounds() const { return {}; }
    virtual std::vector<double> upperBounds() const { return {}; }

    // --- 0th Order ---
    virtual double eval(std::vector<std::unique_ptr<Particle>>& particles) = 0;
    virtual double evalLn(std::vector<std::unique_ptr<Particle>>& particles) { return log(eval(particles)); }

    // --- 1st Order ---
    virtual double spatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim);
    virtual double paramDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx);

    // --- 2nd Order ---
    // (∇²ψ)/ψ
    virtual double spatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles);

    // Vectorized versions (they loop over the above)
    // ∇ᵢln(ψ)
    virtual std::vector<double> spatialGradientLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);
    // ∇ᵢln|ψ|
    virtual std::vector<double> paramGradientLnAbs(std::vector<std::unique_ptr<Particle>>& particles);

    virtual bool hasJastrow() const { return false; }
    virtual void deactivateJastrow() {
        throw std::logic_error("WaveFunction has no Jastrow parameters");
    }
protected:
    int m_numberOfParameters = 0;
    std::vector<double> m_parameters;
    
    static bool s_useAnalytical; 

    WaveFunction(int numberOfParameters, std::vector<double> parameters)
        : m_numberOfParameters(numberOfParameters), m_parameters(std::move(parameters)) {}

    virtual bool hasAnalyticalDerivatives() const { return false; }
    
    virtual double analyticalSpatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) {
        throw std::logic_error("Analytical derivative not implemented");
    }
    virtual double analyticalParamDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) {
        throw std::logic_error("Analytical derivative not implemented");
    }
    virtual double analyticalSpatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>&) {
        throw std::logic_error("Analytical derivative not implemented");
    }
};