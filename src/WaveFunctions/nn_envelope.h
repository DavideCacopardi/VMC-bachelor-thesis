#pragma once
#include <memory>
#include <vector>

#include "wavefunction.h"
#include "neuralnetwork.h"

class NN_envelope : public WaveFunction {
public:
    NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay);
    NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay, ActivationFunc actFunc);
    NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay, const std::vector<double>& params);
    NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay, ActivationFunc actFunc, const std::vector<double>& params);

    const std::vector<double>& getParameters() override;

    double eval(std::vector<std::unique_ptr<class Particle>>& particles) override;
    torch::Tensor encode(std::vector<std::unique_ptr<class Particle>>& particles);
    // std::vector<double> QFac(std::vector<std::unique_ptr<class Particle>>& particles);
    // double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    // std::vector<double> computeQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particle_idx) override;

    // std::vector<double> computeLogParDer_vect(std::vector<std::unique_ptr<class Particle>>& particles) override;

    NeuralNetwork& net() { return m_net; }

    std::vector<double> spatialGradientLn(std::vector<std::unique_ptr<Particle>>& particles, unsigned int particle_idx) override;
    std::vector<double> paramGradientLnAbs(std::vector<std::unique_ptr<Particle>>& particles) override;
    double spatialNormalizedLaplacian(std::vector<std::unique_ptr<Particle>>& particles) override;

    // block the scalar methods from being used directly on the NN
    double spatialDerivativeLn(std::vector<std::unique_ptr<Particle>>&, unsigned int, unsigned int) override {
        throw std::logic_error("ERR: Do not call scalar derivatives on NN_envelope. Use spatialGradientLn for performance.");
    }
    double paramDerivativeLnAbs(std::vector<std::unique_ptr<Particle>>&, unsigned int) override {
        throw std::logic_error("ERR: Do not call scalar derivatives on NN_envelope. Use paramGradientLnAbs for performance.");
    }

private:
    int m_N;
    int m_D;
    int m_Nin;
    NeuralNetwork m_net;

    bool hasAnalyticalDerivatives() const override { return true; }
};