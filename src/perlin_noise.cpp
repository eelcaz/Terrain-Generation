#include "perlin_noise.h"

#include <math.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <functional>

PerlinNoise::PerlinNoise() {
    std::mt19937 generator(0);
    std::uniform_real_distribution<> distribution(0, tableSize);
    auto dice = std::bind(distribution, generator);
    for (unsigned i = 0; i < tableSize; ++i) {
        permutation[i] = dice();
    }
}

PerlinNoise::PerlinNoise(unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<> distribution(0, tableSize);
    auto dice = std::bind(distribution, generator);
    for (unsigned i = 0; i < tableSize; ++i) {
        permutation[i] = dice();
    }
};

double PerlinNoise::interpolate(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

Vector2 PerlinNoise::randomGradient(int z, int x) {
    int randDir = permutation[(permutation[abs(z) % 256] + abs(x)) % 256];
    Vector2 v = {
        (float)cos(randDir),    // z
        (float)sin(randDir)     // x
    };
    return v;
};

double PerlinNoise::dotProduct(int GridZ, int GridX, double pz, double px) {
    Vector2 gradient = randomGradient(GridZ, GridX);
    // get the offset vector from the grid point to the target point
    Vector2 offsetVector = {
        pz - (double)GridZ,      // z
        px - (double)GridX       // x
    };
    return gradient.z * offsetVector.z + gradient.x * offsetVector.x;
};

double PerlinNoise::noise(double z, double x) {
    int zGrid0 = (int)floor(z);
    int xGrid0 = (int)floor(x);
    int zGrid1 = zGrid0 + 1;
    int xGrid1 = xGrid0 + 1;

    // calculate weights
    double wz = z - (double)zGrid0;
    double wx = x - (double)xGrid0;

    double dot1, dot2, interp1, interp2;
    dot1 = dotProduct(zGrid0, xGrid0, z, x);
    dot2 = dotProduct(zGrid1, xGrid0, z, x);
    interp1 = interpolate(dot1, dot2, wz);

    dot1 = dotProduct(zGrid0, xGrid1, z, x);
    dot2 = dotProduct(zGrid1, xGrid1, z, x);
    interp2 = interpolate(dot1, dot2, wz);

    return interpolate(interp1, interp2, wx);
}

// int main(int argc, char *argv[]) {
//     float n1, n2, n3;
//     n1 = PerlinNoise::noise(1.3, 1.6);
//     n2 = PerlinNoise::noise(85.24, 46.22);
//     n3 = PerlinNoise::noise(0.11, 39.9);
//     std::cout << "noise values:" << "\n";
//     std::cout << n1 << "\n";
//     std::cout << n2 << "\n";
//     std::cout << n3 << "\n";

//     for (double z = 0.05; z < 1; z += 0.1) {
//         for (double x = 0.05; x < 1; x += 0.1) {
//             double val = PerlinNoise::noise(z, x);
//             val = (val + 1)/2;
//             val = (int)(val * 100);
//             std::cout << std::left << std::setw(12) << val << " ";
//             // std::cout << std::left << std::setw(12) << val << " ";
//         }
//         std::cout << "\n";
//     }
// }

// int main(int argc, char *argv[]) {
//     int chunkWidth = 16;
//     int terrainZoom = 16;
//     int chunkZ = 0;
//     int chunkX = 0;
//     double offset = (double)1/(2*chunkWidth);
//     for (int z = 0; z < chunkWidth; ++z) {
//         for (int x = 0; x < chunkWidth; ++x) {
//             double noiseZ = (chunkZ + offset + (double)z/chunkWidth)/terrainZoom;
//             double noiseX = (chunkX + offset + (double)x/chunkWidth)/terrainZoom;
//             double val = PerlinNoise::noise(noiseZ, noiseX);
//             std::cout << std::left << std::setw(12) << val << " ";
//         }
//         std::cout << "\n";
//     }
// }
