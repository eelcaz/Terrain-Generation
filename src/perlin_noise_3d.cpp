#include "perlin_noise_3d.h"

#include <math.h>
#include <random>
#include <functional>

#include <iostream>

PerlinNoise3D::PerlinNoise3D() {
    std::mt19937 generator(0);
    std::uniform_real_distribution<> distribution;
    auto dice = std::bind(distribution, generator);
    for (unsigned i = 0; i < tableSize; ++i) {
        double theta = acos(2 * dice() - 1);
        double phi = 2 * dice() * M_PI;

        double x = cos(phi) * sin(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(theta);
        gradients[i] = {y, z, x};
    }
}

PerlinNoise3D::PerlinNoise3D(unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<> distribution;
    auto dice = std::bind(distribution, generator);
    for (unsigned i = 0; i < tableSize; ++i) {
        double theta = acos(2 * dice() - 1);
        double phi = 2 * dice() * M_PI;

        double x = cos(phi) * sin(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(theta);
        gradients[i] = {y, z, x};
    }
}

double PerlinNoise3D::interpolate(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

Vector3 PerlinNoise3D::randomGradient(int y, int z, int x) {
    int randInd = PERMUTATION[(PERMUTATION[(PERMUTATION[abs(y) % 256] + abs(z)) % 256] + abs(x)) % 256];
    // test that these values are within range of gradients' length
    return gradients[randInd];
};

double PerlinNoise3D::dotProduct(int GridY, int GridZ, int GridX,
                                 double py, double pz, double px) {
    Vector3 gradient = randomGradient(GridY, GridZ, GridX);
    // get the offset vector from the grid point to the target point
    Vector3 offsetVector = {
        py - (double)GridY,     // y
        pz - (double)GridZ,     // z
        px - (double)GridX      // x
    };
    return gradient.y * offsetVector.y +
           gradient.z * offsetVector.z +
           gradient.x * offsetVector.x;
};

double PerlinNoise3D::noise(double y, double z, double x) {
    int yGrid0 = (int)floor(y);
    int zGrid0 = (int)floor(z);
    int xGrid0 = (int)floor(x);
    int yGrid1 = yGrid0 + 1;
    int zGrid1 = zGrid0 + 1;
    int xGrid1 = xGrid0 + 1;

    // calculate weights
    double wy = y - (double)yGrid0;
    double wz = z - (double)zGrid0;
    double wx = x - (double)xGrid0;

    double dot1, dot2;
    double interp1, interp2, interp3, interp4, interp5, interp6;
    dot1 = dotProduct(yGrid0, zGrid0, xGrid0, y, z, x);
    dot2 = dotProduct(yGrid1, zGrid0, xGrid0, y, z, x);
    interp1 = interpolate(dot1, dot2, wy);

    dot1 = dotProduct(yGrid0, zGrid1, xGrid0, y, z, x);
    dot2 = dotProduct(yGrid1, zGrid1, xGrid0, y, z, x);
    interp2 = interpolate(dot1, dot2, wy);

    dot1 = dotProduct(yGrid0, zGrid0, xGrid1, y, z, x);
    dot2 = dotProduct(yGrid1, zGrid0, xGrid1, y, z, x);
    interp3 = interpolate(dot1, dot2, wy);

    dot1 = dotProduct(yGrid0, zGrid1, xGrid1, y, z, x);
    dot2 = dotProduct(yGrid1, zGrid1, xGrid1, y, z, x);
    interp4 = interpolate(dot1, dot2, wy);

    interp5 = interpolate(interp1, interp2, wz);
    interp6 = interpolate(interp3, interp4, wz);

    return interpolate(interp5, interp6, wx);
}
