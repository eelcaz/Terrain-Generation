#include "perlin_noise.h"

#include <math.h>
#include <iostream>
#include <iomanip>


float PerlinNoise::interpolate(float a, float b, float weight) {
    // the simplest method of interpolation
    // might need a more complex method if in need of better results
    if (weight < 0) return a;
    if (weight > 1) return b;
    return a * (1 - weight) + (b * weight);
};

Vector2 PerlinNoise::randomGradient(int x, int y) {
    int randDir = PERMUTATION[(PERMUTATION[x] + y) % 256];
    Vector2 v = {
        (float)cos(randDir),    // x
        (float)sin(randDir)     // y
    };
    return v;
};

float PerlinNoise::dotProduct(int xGrid, int yGrid, float px, float py) {
    Vector2 gradient = randomGradient(xGrid, yGrid);
    // get the offset vector from the grid point to the target point
    Vector2 offsetVector = {
        px - (float)xGrid,      // x
        py - (float)yGrid       // y
    };
    return gradient.x * offsetVector.x + gradient.y * offsetVector.y;
};

float PerlinNoise::noise(float x, float y) {
    int xGrid0 = (int)floor(x);
    int yGrid0 = (int)floor(y);
    int xGrid1 = xGrid0 + 1;
    int yGrid1 = yGrid0 + 1;

    // calculate weights
    float wx = x - (float)xGrid0;
    float wy = y - (float)yGrid0;

    float dot1, dot2, interp1, interp2;
    dot1 = dotProduct(xGrid0, yGrid0, x, y);
    dot2 = dotProduct(xGrid1, yGrid0, x, y);
    interp1 = interpolate(dot1, dot2, wx);

    dot1 = dotProduct(xGrid0, yGrid1, x, y);
    dot2 = dotProduct(xGrid1, yGrid1, x, y);
    interp2 = interpolate(dot1, dot2, wx);

    return interpolate(interp1, interp2, wy);
}

int main(int argc, char* argv[]) {
    float n1, n2, n3;
    n1 = PerlinNoise::noise(1.3, 1.6);
    n2 = PerlinNoise::noise(85.24, 46.22);
    n3 = PerlinNoise::noise(0.11, 39.9);
    std::cout << "noise values:" << "\n";
    std::cout << n1 << "\n";
    std::cout << n2 << "\n";
    std::cout << n3 << "\n";

    for (float i = 5; i < 8; i += 0.2) {
        for (float j = 5; j < 8; j += 0.2) {
            float val = PerlinNoise::noise(i, j);
            std::cout << std::left << std::setw(12) << val << " ";
        }
        std::cout << "\n";
    }

}