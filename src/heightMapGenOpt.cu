#include <iostream>
#include <iomanip>
#include "terrain_generator.h"


__device__ double interpolateOpt(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

__device__ double dotProductOpt(int GridZ, int GridX, double pz, double px, int* permutation) {
    // get the random vector on the gridPoint
    int randDir = permutation[(permutation[abs(GridZ) % 256] + abs(GridX)) % 256];
    double gradZ = cos((double)randDir);
    double gradX = sin((double)randDir);
    // get the offset vector from the grid point to the target point
    double offsetZ = pz-(double)GridZ;
    double offsetX = px-(double)GridX;

    return gradZ * offsetZ + gradX * offsetX;
};

__global__ void chunkHeightMapKernelOpt(int chunkZ, int chunkX, int* heightMap, int* permutation) {
    __shared__ float s_totals[64];
    int sectionSize = 64;
    int id = (threadIdx.x % sectionSize) + sectionSize*blockIdx.x;
    double offset = (double)1/(2*(Terrain::CHUNK_WIDTH-1));
    int _z = id / Terrain::CHUNK_WIDTH;
    int _x = id % Terrain::CHUNK_WIDTH;
    double z = (chunkZ + offset + (double)_z/(Terrain::CHUNK_WIDTH-1))/Terrain::TERRAIN_ZOOM;
    double x = (chunkX + offset + (double)_x/(Terrain::CHUNK_WIDTH-1))/Terrain::TERRAIN_ZOOM;
    double noiseZ, noiseX;

    // fbm iterations

    int i = threadIdx.x / sectionSize;

    double total = 0.0;
    double maxVal = 0;

    double amplitude = pow(0.58, (double) i);
    double frequency = pow(2.0, (double) i);

    noiseZ = z * frequency;
    noiseX = x * frequency;

    // noise calculations
    int zGrid0 = (int)floor(noiseZ);
    int xGrid0 = (int)floor(noiseX);
    int zGrid1 = zGrid0 + 1;
    int xGrid1 = xGrid0 + 1;

    // calculate weights
    double wz = noiseZ - (double)zGrid0;
    double wx = noiseX - (double)xGrid0;

    double dot1, dot2, interp1, interp2;
    dot1 = dotProductOpt(zGrid0, xGrid0, noiseZ, noiseX, permutation);
    dot2 = dotProductOpt(zGrid1, xGrid0, noiseZ, noiseX, permutation);
    interp1 = interpolateOpt(dot1, dot2, wz);

    dot1 = dotProductOpt(zGrid0, xGrid1, noiseZ, noiseX, permutation);
    dot2 = dotProductOpt(zGrid1, xGrid1, noiseZ, noiseX, permutation);
    interp2 = interpolateOpt(dot1, dot2, wz);

    double noiseVal = interpolateOpt(interp1, interp2, wx);

    double localTotal = noiseVal * amplitude;

    // add to final total
    // if (id >= 64) {
    //     printf("threadIdx: %d, blockIdx: %d id: %d\n", threadIdx.x, blockIdx.x, id);
    // }

    atomicAdd(&s_totals[id % 64], (float)localTotal);
    // printf("s_totals[%d]: %f\n", id%64, s_totals[id%64]);

    __syncthreads();
    // divide by maxVal
    if (i == 0) {
        for (int j = 0; j < 6; ++j) {
            maxVal += pow(0.58, (double) j);
        }

        total = s_totals[id % 64];
        // printf("id: %d, total: %f, s_totals: %f\n", id, total, s_totals[id % 64]);
        total /= maxVal;
        // printf("id: %d, total: %f, maxVal: %f\n", id, total, maxVal);
        total = (total + 1)/2;

        heightMap[id] = (int)floor((double)total * Terrain::TERRAIN_AMPLITUDE);
    }

    return;
};

// void setConstantPermutation(int* permutation) {
//     size_t permutationSize = sizeof(int)*256;
//     cudaMemcpyToSymbol(c_permutation, permutation, permutationSize);
// };

int* chunkHeightMapKernelOpt(int chunkZ, int chunkX, int* permutation) {
    int* d_heightMap;
    int* d_permutation;

    size_t heightMapSize = sizeof(int)*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
    int* heightMap = new int[Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH];
    cudaMalloc(&d_heightMap, heightMapSize);

    size_t permutationSize = sizeof(int)*256;
    cudaMalloc(&d_permutation, permutationSize);
    cudaMemcpy(d_permutation, permutation, permutationSize, cudaMemcpyHostToDevice);

    int block_width = Terrain::CHUNK_WIDTH*(Terrain::CHUNK_WIDTH/4)*6;
    dim3 dimBlock(block_width, 1, 1);
    dim3 dimGrid(4, 1, 1);
    chunkHeightMapKernelOpt<<<dimGrid, dimBlock>>>(chunkZ, chunkX, d_heightMap, d_permutation);
    cudaMemcpy(heightMap, d_heightMap, heightMapSize, cudaMemcpyDeviceToHost);
    return heightMap;
}


int main(int argc, char *argv[]) {
    Terrain terrain(2022);

    int* d_heightMap;
    size_t heightMapSize = sizeof(int) * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH;
    int* heightMap = new int[Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH];

    int* d_permutation;
    size_t permutationSize = sizeof(int)*256;

    int block_width = Terrain::CHUNK_WIDTH*(Terrain::CHUNK_WIDTH/4)*6;
    dim3 dimBlock(block_width, 1, 1);
    dim3 dimGrid(4, 1, 1);

    // setup gpu timers
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // using constant memory for gradients and heightMap & shared memory
    for (int i = 0; i < 1000 ; ++i) {
        cudaMalloc(&d_heightMap, heightMapSize);
        cudaMalloc(&d_permutation, permutationSize);
        cudaMemcpy(d_permutation, terrain.noise2D.permutation, permutationSize, cudaMemcpyHostToDevice);
        chunkHeightMapKernelOpt<<<dimGrid, dimBlock>>>(0, 0, d_heightMap, d_permutation);
        cudaFree(d_heightMap);
    }

    // stop gpu timers
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // after cudaEventRecord
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("chunkHeightMapKernelOpt time elapsed after 1000 kernel executions: %fms\n", time);
    delete[] heightMap;
    return 0;
}
