#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <limits>

#include "../common.h"
#include "repulsiveellipticgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../Particles/particle.h"

using namespace CommonUtils;

RepEllipticGaussian::RepEllipticGaussian(double alpha, double beta, double rep_a)
    : WaveFunction(2, { alpha, beta }), m_rep_a(rep_a) {
    if (alpha < 0) throw std::invalid_argument("ERR: alpha must be non-negative");
    if (beta < 0) throw std::invalid_argument("ERR: beta must be non-negative");
    if (rep_a < 0) throw std::invalid_argument("ERR: rep_a must be non-negative");
}

// same as evalLn in EllipticGaussian
double RepEllipticGaussian::evalLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
    long double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            if (j == 2)
                sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
            else
                sum += sq(particles[i]->getPosition()[j]);
        }
    }
    return -m_parameters[0] * sum;
}

double RepEllipticGaussian::evalLn_onlyInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
    if (m_rep_a == 0) return 0;

    long double interaction_terms = 0; // sum of ln f
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = i + 1; j < particles.size(); j++) {
            double dist = distance(particles[i]->getPosition(), particles[j]->getPosition());
            if (dist <= m_rep_a)
                return -std::numeric_limits<double>::infinity();
            interaction_terms += log(1 - m_rep_a / dist);
        }
    }
    return interaction_terms;
}

double RepEllipticGaussian::evalLn(std::vector<std::unique_ptr<class Particle>>& particles) {
    return evalLn_noInteraction(particles) + evalLn_onlyInteraction(particles);
}

double RepEllipticGaussian::eval(std::vector<std::unique_ptr<class Particle>>& particles) {
    return exp(evalLn(particles));
}