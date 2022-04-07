#include <array>
#include "perlin_noise.h"
#include <iostream>
#include <iomanip>


const int CHUNK_WIDTH = 16; // 16 * 16,

struct chunk {
    int width;
};

class Terrain {
private:
    static float fbmNoise(float x, float y);
public:
    static std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMap(int x, int y);
};

float Terrain::fbmNoise(float x, float y) {
    // placeholder values, will probable want this customizable in the future
    int octaves = 4;
    float total = 0;
    float frequency = 1;
    float amplitude = 1;
    float persistence = 4;
    float maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        total += PerlinNoise::noise(x*frequency, y*frequency) * amplitude;
        maxVal += amplitude;
        amplitude *= persistence;
        frequency *= 2;
    }
    return total/maxVal;
};

std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH>
Terrain::generateChunkHeightMap(int chunkX, int chunkY) {
    std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH> heightMap;

    for (int x = 0; x < CHUNK_WIDTH; ++x) {
        for (int y = 0; y < CHUNK_WIDTH; ++y) {
            float offset = 0.5/CHUNK_WIDTH;
            float val = fbmNoise(
                chunkX + ((float)x/CHUNK_WIDTH) + offset,
                chunkY + ((float)y/CHUNK_WIDTH) + offset
            );
            heightMap[x][y] = (val + 1.0f)/2.0f; // make positive [0-1]
        }
    }

    return heightMap;
};

int main(int argc, char *argv[]) {
    auto c0_0 = Terrain::generateChunkHeightMap(5, 0);
    auto c0_1 = Terrain::generateChunkHeightMap(6, 0);
    for (auto row : c0_0) {
        for (auto block : row) {
            std::cout << std::left << std::setw(12) << block << " ";
        }
        std::cout << "\n";
    }
    for (auto row : c0_1) {
        for (auto block : row) {
            std::cout << std::left << std::setw(12) << block << " ";
        }
        std::cout << "\n";
    }
}
