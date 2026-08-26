#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "neuralnetwork.h"
#include "../Particles/particle.h"

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid, double helpDecay)
    : NeuralNetwork(Nin, Nhid, helpDecay, [](const torch::Tensor& t) { return torch::tanh(t); }) {}

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid, double helpDecay, ActivationFunc actFun)
    : m_Nin(Nin), m_Nhid(Nhid), m_nParams((Nin + 2) * Nhid + 1), m_actFun(actFun) {
    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    // Xavier/Glorot scaling: keeps variance of activations stable
    double scale1 = std::sqrt(2.0 / (Nin + Nhid)) * 0.01;
    double scale2 = std::sqrt(2.0 / Nhid) * 0.01;
    scale2 = scale1 = 1;
    m_helpDecay = register_parameter("helpDecay", torch::tensor({ helpDecay }, opts));
    m_W1 = register_parameter("W1", torch::randn({ Nin, Nhid }, opts) * scale1);
    m_b = register_parameter("b", torch::randn(Nhid, opts) * scale2);
    m_W2 = register_parameter("W2", torch::randn(Nhid, opts) * scale2);
}

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid, const std::vector<double>& params)
    : NeuralNetwork(Nin, Nhid, [](const torch::Tensor& t) { return torch::tanh(t); }, params) {}

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid, ActivationFunc actFun, const std::vector<double>& params)
    : m_Nin(Nin), m_Nhid(Nhid), m_nParams((Nin + 2) * Nhid + 1), m_actFun(actFun) {
    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    const int64_t nW1 = Nin * Nhid;
    const int64_t nb = Nhid;
    const int64_t nW2 = Nhid;

    if ((int64_t)params.size() != m_nParams)
        throw std::invalid_argument("params size " + std::to_string(params.size()) +
            " does not match (Nin+2)*Nhid + 1 = " + std::to_string(m_nParams));

    auto it = params.begin();

    auto alpha_tensor = torch::tensor(std::vector<double>(it, it + 1), opts);
    it++;
    auto W1_tensor = torch::tensor(
        std::vector<double>(it, it + nW1), opts).reshape({ Nin, Nhid }
        );
    it += nW1;
    auto b_tensor = torch::tensor(
        std::vector<double>(it, it + nb), opts
    );
    it += nb;
    auto W2_tensor = torch::tensor(
        std::vector<double>(it, it + nW2), opts
    );

    m_helpDecay = register_parameter("helpDecay", alpha_tensor);
    m_W1 = register_parameter("W1", W1_tensor);
    m_b = register_parameter("b", b_tensor);
    m_W2 = register_parameter("W2", W2_tensor);
}

torch::Tensor NeuralNetwork::ln_forward(torch::Tensor input) {
    // input is a [ 1 x Nin ] matrix
    torch::Tensor hidden = torch::addmm(m_b, input, m_W1);

    hidden = m_actFun(hidden);
    torch::Tensor u_out = torch::mv(hidden, m_W2);
    u_out = torch::clamp(u_out, -20.0, 20.0);    // should be safe for double

    // Gaussian Envelope (-m_helpDecay * r^2)
    torch::Tensor r_squared = input.pow(2).sum(-1, true);
    torch::Tensor gauss_envelope = -m_helpDecay.abs() * r_squared;

    return u_out + gauss_envelope;
}

torch::Tensor NeuralNetwork::forward(torch::Tensor input) {
    return torch::exp(ln_forward(input));
}

void NeuralNetwork::setGrads(const std::vector<double>& grads) {
    int idx = 0;
    for (auto& param : this->parameters()) {
        auto grad_tensor = torch::empty_like(param);

        // Grab the raw memory pointer for safe, direct C++ assignment
        double* grad_data = grad_tensor.data_ptr<double>();

        for (int i = 0; i < param.numel(); i++) {
            grad_data[i] = grads[idx++];
        }

        param.mutable_grad() = grad_tensor;
    }
}

std::vector<double> NeuralNetwork::getParams() const {
    std::vector<double> out;
    out.reserve(m_nParams);
    for (const auto& p : parameters()) {
        auto flat = p.detach().flatten();
        for (int i = 0; i < flat.numel(); ++i)
            out.push_back(flat[i].item<double>());
    }
    return out;
}