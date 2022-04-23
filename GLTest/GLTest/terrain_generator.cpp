#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "terrain_generator.h"
#include "perlin_noise.h"

double Terrain::fbmNoise(double z, double x, int octaves) {
    float total = 0.0;
    float maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        float amplitude = pow(0.58f, i);
        float frequency = pow(2.0f, i);
        total += PerlinNoise::noise(z*frequency, x*frequency) * amplitude;
        maxVal += amplitude;
    }
    return total/maxVal;
};

int** Terrain::generateChunkHeightMap(int chunkZ, int chunkX) {
    // allocate new chunk heightmap
    int **heightMap = new int*[CHUNK_WIDTH];
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        heightMap[z] = new int[CHUNK_WIDTH];
    }

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            double offset = (double)1/(2*CHUNK_WIDTH);
            double val = fbmNoise(
                (chunkZ + offset + (double)z/CHUNK_WIDTH)/TERRAIN_ZOOM,
                (chunkX + offset + (double)x/CHUNK_WIDTH)/TERRAIN_ZOOM,
                6
            );
            val = (val + 1) / 2;
            val = (int)floor(val * TERRAIN_AMPLITUDE);
            heightMap[z][x] = val;
        }
    }

    return heightMap;
};

int*** Terrain::generateChunkData(int chunkZ, int chunkX) {
    // allocate data for whole chunk, zero initialized
    int ***chunk = new int**[CHUNK_HEIGHT];
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        chunk[y] = new int*[CHUNK_WIDTH];
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            chunk[y][z] = new int[CHUNK_WIDTH]{0};
        }
    }

    auto heightMap = generateChunkHeightMap(chunkZ, chunkX);
    // solid parts of chunk will have value of 1, otherwise 0
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            for (int y = 0; y < heightMap[z][x]; ++y) {
                chunk[y][z][x] = 1;
            }
        }
    }

    return chunk;
};

void Terrain::deallocateChunk(int*** chunk) {
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            delete[] chunk[y][z];
        }
        delete[] chunk[y];
    }
    delete[] chunk;
}
