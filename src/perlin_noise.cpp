#include <math.h>
#include <iostream>
#include <iomanip>

struct Vector2 {
    float x;
    float y;
};

class PerlinNoise {
private:
    // random values from 0-255, used in ken perlins original code
    static constexpr int PERMUTATION[] = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,
        103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,
        26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,
        87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,
        46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
        187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,
        198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,
        255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,
        170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
        172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,
        104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
        241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,
        157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,
        93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };
public:
    static float interpolate(float a, float b, float weight);
    static float dotProduct(int gridX, int gridY, float x, float y);
    static Vector2 randomGradient(int x, int y);
    static float noise(float x, float y);
};


float PerlinNoise::interpolate(float a, float b, float weight) {
    // the simplest method of interpolation
    // might need a more complex method if in need of better results
    if (weight < 0) return a;
    if (weight > 1) return b;
    return a*(1-weight) + (b*weight);
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
    return gradient.x*offsetVector.x + gradient.y*offsetVector.y;
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

int main(int argc, char *argv[]) {
    return 0;
}
