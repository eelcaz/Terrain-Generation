struct Vector2 {
    double z;
    double x;
};

class PerlinNoise {
private:
    // random values from 0-255, used in ken perlins original code
    static const int tableSize = 256;
    int permutation[tableSize];
public:
    PerlinNoise();
    PerlinNoise(unsigned int seed);
    static double interpolate(double a, double b, double weight);
    double dotProduct(int gridZ, int gridX, double z, double x);
    Vector2 randomGradient(int z, int x);
    double noise(double z, double x);
};
