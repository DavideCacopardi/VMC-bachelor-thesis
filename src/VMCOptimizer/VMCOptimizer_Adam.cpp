#include "VMCOptimizer_Adam.h"

using namespace CommonUtils;

VMCOptimizer_Adam::VMCOptimizer_Adam(
    const runConfig& cfg,
    MCEngine& engine,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : VMCOptimizer(cfg, engine, logfile, outfile, paramsfile), m_lr(cfg.Adam_lr) {}

std::vector<double> VMCOptimizer_Adam::optimize(std::vector<double> params, const std::vector<bool>& optimize_mask) {
    m_mcCount = 0;
    
    std::vector<double> m(params.size(), 0.0);  // first moment
    std::vector<double> v(params.size(), 0.0);  // second moment
    std::vector<double> grad(params.size(), 0.0);

    double min_improvement = m_cfg.Adam_min_improvement;
    unsigned int patience_counter = 0;


    try {
        for (unsigned int t = 1; t <= m_cfg.Adam_nSteps; t++) {
            double res = computeMC(params, grad);

            if (checkPlateau(res, m_bestObjective, patience_counter, min_improvement)) {
                min_improvement *= 0.1;
                m_lr *= 0.5;
                std::cout << " Plateau detected; setting lr = "
                    << std::scientific << std::setprecision(8) << m_lr << std::endl;
            }

            // Adam step
            for (unsigned int i = 0; i < params.size(); i++) {
                // std::cout << " DEBUG: " << grad[i];
                // Skip froken parameters
                if (optimize_mask.size() == params.size() && !optimize_mask[i]) continue;

                // Adam update
                m[i] = c_beta1 * m[i] + (1 - c_beta1) * grad[i];
                v[i] = c_beta2 * v[i] + (1 - c_beta2) * sq(grad[i]);

                double m_hat = m[i] / (1 - pow(c_beta1, t));
                double v_hat = v[i] / (1 - pow(c_beta2, t));

                params[i] -= m_lr * m_hat / (sqrt(v_hat) + c_epsilon);
            }
            // std::cout << std::endl;
        }
    }
    catch (std::runtime_error& e) {
        std::cout << "\nAdam optimization interrupted: " << e.what() << std::endl;
        std::cout << "  WRN: Falling back to the lowest energy parameters observed (m_bestParams).\n";
        params = m_bestParams;
    }


    // Output finale
    if (m_outfile) {
        *m_outfile << "# Optimal parameters (Adam): " << std::setprecision(8);
        for (auto p : params) *m_outfile << p << ", \t";
        *m_outfile << std::endl;
    }
    if (m_paramsfile) {
        for (auto p : params) *m_paramsfile << p << std::endl;
    }

    return params;
}

bool VMCOptimizer_Adam::checkPlateau(
    double current_val, double& best_val, unsigned int& patience_counter, double min_improvement
) {
    if (current_val < best_val - min_improvement) {
        // Significant improvement in E found
        best_val = current_val;
        patience_counter = 0;
        return false;
    }
    else {
        // No significant improvement
        patience_counter++;
        if (patience_counter >= m_cfg.Adam_max_patience) {
            patience_counter = 0;
            return true;
        }
        return false;
    }
}