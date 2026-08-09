#pragma once
#include <vector>

enum Flavor {
    A, B
};

/**
 * @brief Constructs a particle with an initial position.
 * @param position Vector containing the initial spatial coordinates.
 */
class Particle {
public:
    /**
     * @brief Constructs a particle with an initial position.
     * @param position Vector containing the initial spatial coordinates.
     */
    Particle(const std::vector<double>& position);

    /**
     * @brief Constructs a particle with an initial position and an initial flavor.
     * @param position Vector containing the initial spatial coordinates.
     * @param flav Flavor of the particle (A or B).
     */
    Particle(const std::vector<double>& position, const Flavor flav);

    /**
     * @brief Shifts the particle along a specific dimension.
     * @param change The displacement (delta) to add to the current coordinate.
     * @param dimension The dimension index (e.g., 0 for x, 1 for y, 2 for z).
     */
    void adjustPosition(double change, unsigned int dimension);

    /**
     * @brief Absolutely sets the position along a single dimension.
     * @param value The new coordinate value.
     * @param dimension The dimension index.
     */
    void setPosition(double value, unsigned int dimension);

    /**
     * @brief Sets the entire position of the particle in space.
     * @param loc Vector containing the new coordinates.
     */
    void setPosition(const std::vector<double> loc);

    /**
     * @brief Sets the flavor of the particle.
     * @param flav New flavor.
     */
    void setFlavor(const Flavor flav);

    /**
     * @brief Gets the current position of the particle.
     * @return Reference to the coordinate vector.
     */
    std::vector<double>& getPosition() { return m_position; }

    /**
     * @brief Gets the flavor of the particle.
     * @return Flavor of the particle.
     */
    Flavor getFlavor() { return m_flavor; }

    /**
     * @brief Gets the number of dimensions the particle moves in.
     * @return Number of dimensions.
     */
    unsigned int getNumberOfDimensions() { return m_numberOfDimensions; }

    // to setup initial state
    static double s_min_dist;
    static void set_min_dist(bool min_dist) { s_min_dist = min_dist; }
    static double s_max_radius;
    static void set_max_radius(bool max_radius) { s_max_radius = max_radius; }

private:
    unsigned int m_numberOfDimensions = 0;
    std::vector<double> m_position = std::vector<double>();
    Flavor m_flavor = A;
};