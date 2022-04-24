struct Vector3 {
    double y;
    double z;
    double x;
};

class PerlinNoise3D {
private:
    static const int tableSize = 256;
    static constexpr double M_PI = 3.14159265358979323846;
    Vector3 gradients[tableSize];
    unsigned permutationTable[tableSize];
public:
    PerlinNoise3D(unsigned int seed);
    double interpolate(double a, double b, double weight);
    Vector3 randomGradient(int y, int z, int x);
    double dotProduct(int GridY, int GridZ, int GridX, double py, double pz, double px);
    double noise(double y, double z, double x);
};
