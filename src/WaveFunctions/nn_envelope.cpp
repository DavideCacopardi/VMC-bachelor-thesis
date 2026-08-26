#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "nn_envelope.h"
#include "../Particles/particle.h"

NN_envelope::NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay)
    : WaveFunction(Nhid * (2 + Nin) + 1, {}), m_N(N), m_D(D), m_Nin(Nin), m_net(Nin, Nhid, helpDecay) {}

NN_envelope::NN_envelope(int N, int D, int Nin, int Nhid, double helpDecay, ActivationFunc actFunc)
    : WaveFunction(Nhid * (2 + Nin) + 1, {}), m_N(N), m_D(D), m_Nin(Nin), m_net(Nin, Nhid, helpDecay, actFunc) {}

NN_envelope::NN_envelope(int N, int D, int Nin, int Nhid, const std::vector<double>& params)
    : WaveFunction(Nhid * (2 + Nin) + 1, params), m_N(N), m_D(D), m_Nin(Nin), m_net(Nin, Nhid, params) {}

NN_envelope::NN_envelope(int N, int D, int Nin, int Nhid, ActivationFunc actFunc, const std::vector<double>& params)
    : WaveFunction(Nhid * (2 + Nin) + 1, params), m_N(N), m_D(D), m_Nin(Nin), m_net(Nin, Nhid, actFunc, params) {}

torch::Tensor NN_envelope::encode(std::vector<std::unique_ptr<class Particle>>& particles) {
    std::vector<double> xi;
    xi.reserve(m_N * m_D);
    for (auto& p : particles)
        for (double x : p->getPosition())
            xi.push_back(x);
    // unsqueeze makes the [ m_N * m_D ] vector a [ 1 x m_N * m_D ] matrix
    return torch::tensor(xi, torch::kDouble).unsqueeze(0);
}

double NN_envelope::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
    torch::NoGradGuard no_grad;
    auto input = encode(particles);
    return m_net.forward(input).item<double>();
}

std::vector<double> NN_envelope::spatialGradientLn(
    std::vector<std::unique_ptr<Particle>>& particles, 
    unsigned int particle_idx) 
{
    auto pos = encode(particles).squeeze(0).requires_grad_(true);
    auto log_psi = m_net.ln_forward(pos.unsqueeze(0)).squeeze(0);

    // ∂(log ψ)/∂pos_i for all coordinates, shape [Nin]
    auto grad = torch::autograd::grad(
        { log_psi }, { pos },
        { torch::ones_like(log_psi) }, // =grad_outputs
        false, // =retain_graph
        false // =create_graph
    )[0];

    // Slice out the D elements belonging to particle_idx
    // encode() flattens as [x0,y0, x1,y1, ...] so particle k starts at k*D
    const int start = static_cast<int>(particle_idx) * m_D;
    std::vector<double> gradient(m_D);
    for (int d = 0; d < m_D; ++d) {
        gradient[d] = grad[start + d].item<double>();
    }

    return gradient;
}

// ── ∇²ψ / ψ =  ∇²(ln ψ)  +  |∇(ln ψ)|²  via autograd ────────────────────────────
double NN_envelope::spatialNormalizedLaplacian(
    std::vector<std::unique_ptr<class Particle>>& particles) 
{
    // Build position tensor with gradient tracking
    auto pos = encode(particles).squeeze(0).to(torch::kDouble).requires_grad_(true);
    auto log_psi = m_net.ln_forward(pos.unsqueeze(0)).squeeze(0);

    // First derivatives ∂(ln ψ)/∂pos_i,  shape [Nin]
    auto grad1 = torch::autograd::grad(
        { log_psi }, { pos },
        { torch::ones_like(log_psi) }, // =grad_outputs
        true, // =retain_graph
        true  // =create_graph, needed so we can differentiate again
    )[0];

    // Second derivatives: diagonal of Hessian, shape [Nin]
    // Loop over each coordinate and extract its own second derivative
    double laplacian_log_psi = 0.0;
    for (int i = 0; i < m_Nin; ++i) {
        auto grad2 = torch::autograd::grad(
            { grad1[i] }, { pos },
            {},   // =grad_outputs
            true, // =retain_graph
            false // =create_graph
        )[0];
        laplacian_log_psi += grad2[i].item<double>();
    }

    // |∇ ln ψ|²
    double sq_grad = grad1.pow(2).sum().item<double>(); 

    // ∇²ψ / ψ = ∇²(ln ψ) + |∇(ln ψ)|²
    return laplacian_log_psi + sq_grad; 
}


std::vector<double> NN_envelope::paramGradientLnAbs(std::vector<std::unique_ptr<Particle>>& particles) {
    m_net.zero_grad();
    auto input = encode(particles); 
    auto log_psi = m_net.ln_forward(input); 

    // Compute all parameter gradients
    log_psi.backward();

    // order: helpDecay, W1, b, W2
    std::vector<double> OW(m_numberOfParameters);
    int idx = 0;

    for (const auto& param : m_net.parameters()) {
        auto grad = param.grad().flatten();
        for (int i = 0; i < grad.numel(); ++i) {
            OW[idx++] = grad[i].item<double>();
        }
    }
    return OW;
}

const std::vector<double>& NN_envelope::getParameters() {
    m_parameters = m_net.getParams();
    return m_parameters;
}