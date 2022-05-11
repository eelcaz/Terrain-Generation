#include "perlin_noise_3d.h"
#include "perlin_noise.h"

class Terrain {
private:
    double fbmNoise(double z, double x, int octaves);
public:
    PerlinNoise noise2D;
    PerlinNoise3D noise3D;
    // constants
    // chunk[y][z][x] z=x=CHUNK_WIDTH, y=CHUNK_HEIGHT
    static const int CHUNK_WIDTH = 16;
    static const int CHUNK_HEIGHT = 256;
    static const int NUM_CHUNKS_SIDE = 4;
    // good zoom values seem to be 256/CHUNK_WIDTH
    static const int TERRAIN_ZOOM = 17;
    static const int TERRAIN_AMPLITUDE = 200;
    static const int CAVE_ZOOM = 1;
    static constexpr double CAVE_INTENSITY = 0; // anything greater than this value is a solid
    Terrain(unsigned int seed);
    int** generateChunkHeightMap(int chunkZ, int chunkX);
    int* generateChunkHeightMapGpu(int chunkZ, int chunkX);
    float* generateChunkData(int chunkZ, int chunkX);
    float* generateChunkDataGpu(int chunkZ, int chunkX);
    float* generateChunkDataGpuOpt(int chunkZ, int chunkX);
};
