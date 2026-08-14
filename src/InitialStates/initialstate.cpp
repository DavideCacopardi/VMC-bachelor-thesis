#include <memory>
#include <iostream>
#include <cassert>
#include <string>

#include "initialstate.h"
#include "../Particles/particle.h"
#include "Math/random.h"
#include "../common.h"

using namespace CommonUtils;

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random& rng,
    double min_dist,
    double max_radius
) {
    assert(numberOfDimensions > 0 && numberOfParticles > 0);
    const unsigned int c_max_nIteration = 1000000000;

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (unsigned int i = 0; i < numberOfParticles; i++) {
        std::vector<double> position(numberOfDimensions);
        bool tooClose = true;

        unsigned int nIteration = 0;
        do {
            nIteration++;
            for (unsigned int j = 0; j < numberOfDimensions; j++) {
                position[j] = 2 * (rng.nextDouble() - 0.5) * max_radius;
            }
            tooClose = false;
            for (unsigned int k = 0; !tooClose && k < i; k++) {
                double dist = distance(position, particles[k]->getPosition());
                if (dist <= min_dist) {
                    // std::cout << " DEBUG: dist = " << dist << " is less than " << min_dist << std::endl;
                    tooClose = true;
                }
            }
        } while (tooClose && nIteration < c_max_nIteration);
        if (tooClose) {
            std::string errStr = "ERR: Particle " + std::to_string(i);
            errStr += " didn't manage to be randomly setup satisfying the requested conditions in far too many iterations.";
            throw std::runtime_error(errStr);
        }

        
        // Flavor flav = rng.nextInt(0, 1) ? A : B;
        // particles.push_back(std::make_unique<Particle>(position, flav));

        // TEMP
        if (i % 2 == 0)
            particles.push_back(std::make_unique<Particle>(position, A));
        else
            particles.push_back(std::make_unique<Particle>(position, B));
    }

    return particles;
}