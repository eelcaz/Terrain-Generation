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

int*** Terrain::generateChunkData(int chunkZ, int chunkX) {
    // allocate data for whole chunk, zero initialized
    int ***chunk = Terrain::createEmptyChunkCpu();

    auto heightMap = generateChunkHeightMap(chunkZ, chunkX);
    // solid parts of chunk will have value of 1, otherwise 0
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            for (int y = 0; y < heightMap[z][x]; ++y) {
                chunk[y][z][x] = 1;
            }
        }
    }

    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                double offset = (double)1/(2*(CHUNK_WIDTH-1));
                double fy = ((int)floor(y/CHUNK_WIDTH))+ offset + ((double)(y%CHUNK_WIDTH))/CHUNK_WIDTH;
                double fz = (chunkZ - offset + (double)z/(CHUNK_WIDTH-1));
                double fx = (chunkX - offset + (double)x/(CHUNK_WIDTH-1));
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

int* Terrain::generateChunkDataGpu(int chunkZ, int chunkX) {
    auto heights = chunkHeightMapKernel(chunkZ, chunkX, noise2D.permutation);
    return chunkDataKernel(chunkZ, chunkX, heights, noise3D.gradientsGPU);
};


int* Terrain::generateChunkDataGpuOpt(int chunkZ, int chunkX) {
    auto heights = chunkHeightMapKernelOpt(chunkZ, chunkX, noise2D.permutation);
    return chunkDataKernelOptWrapper(chunkZ, chunkX, heights);
};

int*** Terrain::createEmptyChunkCpu() {
    // allocate data for whole chunk, zero initialized
    int ***chunk = new int**[CHUNK_HEIGHT];
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        chunk[y] = new int*[CHUNK_WIDTH];
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            chunk[y][z] = new int[CHUNK_WIDTH]{0};
        }
    }
    return chunk;
};

void Terrain::deallocateChunk(int*** &chunk) {
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            delete[] chunk[y][z];
        }
        delete[] chunk[y];
    }
    delete[] chunk;
}
