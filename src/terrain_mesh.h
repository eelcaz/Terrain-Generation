#include <array>

static const int CHUNK_WIDTH = 256; // chunk[z][x] z=x=CHUNK_WIDTH

class Terrain {
private:
    static float fbmNoise(float x, float y, int octaves);
    static float fbmNoise2(float x, float y, int octaves);
public:
    static std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMapOld(int x, int y);
    static std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMapInt(int chunkX, int chunkY);
    static std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMapInt2(int chunkX, int chunkY);
    static int** generateChunkHeightMap(int z, int x);
};
