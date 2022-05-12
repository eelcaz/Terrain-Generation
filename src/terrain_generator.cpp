#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "terrain_generator.h"
#include "heightMapGen.h"
#include "heightMapGenOpt.h"
#include "chunkGen.h"
#include "chunkGenOpt.h"

double Terrain::fbmNoise(double z, double x, int octaves) {
    double total = 0.0;
    double maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        double amplitude = pow(0.58f, i);
        double frequency = pow(2.0f, i);
        total += noise2D.noise(z*frequency, x*frequency) * amplitude;
        maxVal += amplitude;
    }
    return total/maxVal;
};

Terrain::Terrain(unsigned int seed)
    : noise2D(PerlinNoise(seed)), noise3D(PerlinNoise3D(seed)) {
    setConstantGradients(noise3D.gradientsGPU);
    // setConstantPermutation(noise2D.permutation);
};

int** Terrain::generateChunkHeightMap(int chunkZ, int chunkX) {
    // allocate new chunk heightmap
    int **heightMap = new int*[CHUNK_WIDTH];
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        heightMap[z] = new int[CHUNK_WIDTH];
    }

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            double offset = (double)1/(2*(CHUNK_WIDTH-1));
            double val = fbmNoise(
                (chunkZ - offset + (double)z/(CHUNK_WIDTH-1))/TERRAIN_ZOOM,
                (chunkX - offset + (double)x/(CHUNK_WIDTH-1))/TERRAIN_ZOOM,
                6
            );
            val = (val + 1) / 2;
            heightMap[z][x] = (int)floor(val * TERRAIN_AMPLITUDE);
        }
    }

    return heightMap;
};

int* Terrain::generateChunkHeightMapGpu(int chunkZ, int chunkX) {
    return chunkHeightMapKernel(chunkZ, chunkX, noise2D.permutation);
};

float* Terrain::generateChunkData(int chunkZ, int chunkX) {
    // allocate data for whole chunk, zero initialized
    float *chunk = new float[CHUNK_HEIGHT*CHUNK_WIDTH*CHUNK_WIDTH]{0};

    auto heightMap = generateChunkHeightMap(chunkZ, chunkX);

    // gen caves
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                double offset = (double)1/(2*(CHUNK_WIDTH-1));
                double fy = ((int)floor(y/CHUNK_WIDTH)) + offset + ((double)(y%CHUNK_WIDTH))/CHUNK_WIDTH;
                double fz = (chunkZ + offset + (double)z/(CHUNK_WIDTH-1));
                double fx = (chunkX + offset + (double)x/(CHUNK_WIDTH-1));
                double val = noise3D.noise(fy*Terrain::CAVE_ZOOM,
                                           fz*Terrain::CAVE_ZOOM,
                                           fx*Terrain::CAVE_ZOOM);
                int index = y*CHUNK_WIDTH*CHUNK_WIDTH + x*CHUNK_WIDTH + z;

                chunk[index] = (float)val;
            }
        }
    }

    // surface
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                int index = y*CHUNK_WIDTH*CHUNK_WIDTH + x*CHUNK_WIDTH + z;
                if (y > heightMap[z][x]) {
                    chunk[index] = CAVE_INTENSITY - 0.1f;
                }
                if (y > heightMap[z][x] - 5
                    && y < heightMap[z][x]
                    && chunk[index] > CAVE_INTENSITY) {

                    chunk[index] = chunk[index] + abs(CAVE_INTENSITY);
                }
            }
        }
    }
    return chunk;
};

float* Terrain::generateChunkDataGpu(int chunkZ, int chunkX) {
    auto heights = chunkHeightMapKernel(chunkZ, chunkX, noise2D.permutation);
    return chunkDataKernel(chunkZ, chunkX, heights, noise3D.gradientsGPU);
};


float* Terrain::generateChunkDataGpuOpt(int chunkZ, int chunkX) {
    auto heights = chunkHeightMapKernelOpt(chunkZ, chunkX, noise2D.permutation);
    return chunkDataKernelOptWrapper(chunkZ, chunkX, heights);
};
