#include "perlin_noise_3d.h"
#include "perlin_noise.h"

class Terrain {
private:
    PerlinNoise noise2D;
    PerlinNoise3D noise3D;
    double fbmNoise(double z, double x, int octaves);
public:
    // constants
    // chunk[y][z][x] z=x=CHUNK_WIDTH, y=CHUNK_HEIGHT
    static const int CHUNK_WIDTH = 16;
    static const int CHUNK_HEIGHT = 256;
    static const int NUM_CHUNKS_SIDE = 16;
    // good zoom values seem to be 256/CHUNK_WIDTH
    static const int TERRAIN_ZOOM = 17;
    static const int TERRAIN_AMPLITUDE = 200;
    static const int CAVE_ZOOM = 1;
    static constexpr double CAVE_INTENSITY = -0.25;
    Terrain(unsigned int seed);
    int** generateChunkHeightMap(int chunkZ, int chunkX);
    int* generateChunkHeightMapGpu(int chunkZ, int chunkX);
    int*** generateChunkData(int chunkZ, int chunkX);
    int* generateChunkDataGpu(int chunkZ, int chunkX);
    int* generateChunkDataGpuOpt(int chunkZ, int chunkX);
    int*** createEmptyChunkCpu();
    static void deallocateChunk(int*** &chunk);
};
