#include "particle.h"
#include <cassert>

double Particle::s_max_radius = 100;
double Particle::s_min_dist = 0;

Particle::Particle(const std::vector<double>& position) {
    m_numberOfDimensions = position.size();
    m_position = position;
}

Particle::Particle(const std::vector<double>& position, const Flavor flav) {
    m_numberOfDimensions = position.size();
    m_position = position;
    m_flavor = flav;
}

void Particle::adjustPosition(double change, unsigned int dimension) {
    m_position[dimension] += change;
}

void Particle::setPosition(double value, unsigned int dimension) {
    m_position[dimension] = value;
}

void Particle::setPosition(const std::vector<double> loc) {
    m_position = loc;
}

void Particle::setFlavor(const Flavor flav) {
    m_flavor = flav;
}