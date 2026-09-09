#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "../Particles/particle.h"

/**
 * @brief Abstract base class for all trial wave functions.
 * It also provides numerical fallback methods for derivatives
 * in case analytical expressions are too complex to derive.
 */
class WaveFunction {
public:
    virtual ~WaveFunction() = default;

    // --- Configuration ---
    /**
     * @brief Toggles the preferencial use of analytical derivatives.
     * @param useAnalytical New boolean value.
     */
    static void setUseAnalyticalDerivatives(bool useAnalytical) { s_useAnalytical = useAnalytical; }

    /**
     * @brief Gets the number of variational parameters.
     * @return Number of variational parameters.
     */
    int getNumberOfParameters() const { return m_numberOfParameters; }
    /**
     * @brief Gets the wavefunction's parameters.
     * @return Constant reference to the wavefunction's parameters
     */
    virtual const std::vector<double>& getParameters() { return m_parameters; }

    // for optimization
    /**
     * @brief Gets lower bounds used by NLOPT VMCOptimizers.
     * @return Lower bounds.
     */
    virtual std::vector<double> lowerBounds() const { return {}; }
    /**
     * @brief Gets upper bounds used by NLOPT VMCOptimizers.
     * @return Upper bounds.
     */
    virtual std::vector<double> upperBounds() const { return {}; }

    // --- 0th Order ---
    /**
     * @brief Evaluates the wavefunction at given particles.
     * @param particles Particles.
     * @return ψ(particles)
     */
    virtual double eval(std::vector<std::unique_ptr<Particle>>& particles) = 0;
    /**
     * @brief Evaluates the natural logarithm of the wavefunction at given particles.
     * @param particles Particles.
     * @return ln(ψ(particles))
     */
    virtual double evalLn(std::vector<std::unique_ptr<Particle>>& particles) { return log(eval(particles)); }

    // --- 1st Order ---
    /**
     * @brief Evaluates a component of a particle-wise gradient of the logarithm of the wavefunction.
     * @param particles Particles.
     * @param particle_idx Index of the particle with respect to which the gradient should be evaluated.
     * @param dim Dimensional index.
     * @return [∇ᵢln(ψ)]_dim
     */
    virtual double spatialDerivativeLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx, unsigned int dim);
    /**
     * @brief Evaluates the derivative of the absolute value of the logarithm of the wavefunction wrt a variational parameter.
     * @param particles Particles.
     * @param param_idx Index of the variational parameter.
     * @param dim Dimensional index.
     * @return d(ln|ψ|)/dW_i
     */
    virtual double paramDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>& particles, unsigned int param_idx);

    // --- 2nd Order ---
    /**
     * @brief Evaluates the normalized laplacian of the wavefunction wrt particle positions.
     * @param particles Particles
     * @return (∇²ψ)/ψ
     */
    virtual double spatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles);

    // Vectorized versions (they loop over the above)
    /**
     * @brief Evaluates a particle-wise gradient of the logarithm of the wavefunction.
     * @param particles Particles.
     * @param particle_idx Index of the particle with respect to which the gradient should be evaluated.
     * @return ∇ᵢln(ψ)
     */
    virtual std::vector<double> spatialGradientLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx);
    /**
     * @brief Evaluates the derivative of the absolute value of the logarithm of the wavefunction wrt its variational parameters.
     * @param particles Particles.
     * @return d(ln|ψ|)/dW
     */
    virtual std::vector<double> paramGradientLnAbs(std::vector<std::unique_ptr<Particle>>& particles);

    /**
     * @brief Says whether the wavefunction comprises Jastrow parameters.
     * @return true or false depending on the form of the wavefunction.
     */
    virtual bool hasJastrow() const { return false; }

    /**
     * @brief Sets the Jastrow terms to a value which renders the rest of the wavefunction immune to them.
     */
    virtual void deactivateJastrow() {
        throw std::logic_error("WaveFunction has no Jastrow parameters");
    }
protected:
    int m_numberOfParameters = 0;
    std::vector<double> m_parameters;
    
    static bool s_useAnalytical; 

    /**
     * @brief Construct a new Wave Function object.
     * @param numberOfParameters Number of variational parameters.
     * @param parameters Variational parameters' values.
     */
    WaveFunction(int numberOfParameters, std::vector<double> parameters)
        : m_numberOfParameters(numberOfParameters), m_parameters(std::move(parameters)) {}

    /**
     * @brief Says whether the wavefunction has implemented analytical derivatives.
     * @return true or false based on the implementation.
     */
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