#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "terrain_generator.h"
#include "heightMapGen.h"
#include "perlin_noise.h"
#include "perlin_noise_3d.h"

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

int* Terrain::generateChunkHeightMapGpu(int chunkZ, int chunkX) {
    return chunkHeightMapKernel(chunkZ, chunkX);
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

    // this is here just to test that the 3D perlin noise works
    PerlinNoise3D noise3D(999);
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                double offset = (double)1/(2*CHUNK_WIDTH);
                double fy = ((int)floor(y/CHUNK_WIDTH))+ offset + ((double)(y%CHUNK_WIDTH))/CHUNK_WIDTH;
                double fz = (chunkZ + offset + (double)z/CHUNK_WIDTH);
                double fx = (chunkX + offset + (double)x/CHUNK_WIDTH);
                double val = noise3D.noise(fy*Terrain::CAVE_ZOOM,
                                           fz*Terrain::CAVE_ZOOM,
                                           fx*Terrain::CAVE_ZOOM);
                if (chunk[y][z][x] == 1) {
                    chunk[y][z][x] = val <= CAVE_INTENSITY ? 0 : 1;
                }
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
