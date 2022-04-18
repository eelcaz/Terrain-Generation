#include "perlin_noise.h"

#include <math.h>
#include <iostream>
#include <iomanip>


double PerlinNoise::interpolate(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

Vector2 PerlinNoise::randomGradient(int x, int y) {
    int randDir = PERMUTATION[(PERMUTATION[x] + y) % 256];
    Vector2 v = {
        (float)cos(randDir),    // x
        (float)sin(randDir)     // y
    };
    return v;
};

double PerlinNoise::dotProduct(int xGrid, int yGrid, double px, double py) {
    Vector2 gradient = randomGradient(xGrid, yGrid);
    // get the offset vector from the grid point to the target point
    Vector2 offsetVector = {
        px - (double)xGrid,      // x
        py - (double)yGrid       // y
    };
    return gradient.x * offsetVector.x + gradient.y * offsetVector.y;
};

double PerlinNoise::noise(double x, double y) {
    int xGrid0 = (int)floor(x);
    int yGrid0 = (int)floor(y);
    int xGrid1 = xGrid0 + 1;
    int yGrid1 = yGrid0 + 1;

    // calculate weights
    double wx = x - (double)xGrid0;
    double wy = y - (double)yGrid0;

    double dot1, dot2, interp1, interp2;
    dot1 = dotProduct(xGrid0, yGrid0, x, y);
    dot2 = dotProduct(xGrid1, yGrid0, x, y);
    interp1 = interpolate(dot1, dot2, wx);

    dot1 = dotProduct(xGrid0, yGrid1, x, y);
    dot2 = dotProduct(xGrid1, yGrid1, x, y);
    interp2 = interpolate(dot1, dot2, wx);

    return interpolate(interp1, interp2, wy);
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

//     for (double i = 0.05; i < 1; i += 0.1) {
//         for (double j = 0.05; j < 1; j += 0.1) {
//             double val = PerlinNoise::noise(i, j);
//             val = (val + 1)/2;
//             val = (int)(val * 100);
//             std::cout << std::left << std::setw(12) << val << " ";
//             // std::cout << std::left << std::setw(12) << val << " ";
//         }
//         std::cout << "\n";
//     }
// }
