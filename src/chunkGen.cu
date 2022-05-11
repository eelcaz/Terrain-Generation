#include <iostream>
#include <stdio.h>
#include "terrain_generator.h"

__constant__ static const int PERMUTATION[] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,
    103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,
    26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,
    87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,
    46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
    187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,
    198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,
    255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,
    170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,
    104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
    241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,
    157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,
    93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ double interpolate3D(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

__device__ double dotProduct3D(int GridY, int GridZ, int GridX,
                      double py, double pz, double px,
                      double* gradients) {
    int ind = PERMUTATION[(PERMUTATION[(PERMUTATION[abs(GridY) % 256] + abs(GridZ)) % 256] + abs(GridX)) % 256];

    double offsetY = py - (double)GridY;
    double offsetZ = pz - (double)GridZ;
    double offsetX = px - (double)GridX;

    return gradients[ind*3] * offsetY +
        gradients[ind*3+1] * offsetZ +
        gradients[ind*3+2] * offsetX;
};


__global__ void chunkDataKernel(int chunkZ, int chunkX, int* heightMap, double* gradients, int* chunk) {

    int finalVal = 0;

    // map thread to coordinates in chunk
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int _y = id / (Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH);
    int _z = (id % (Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH)) / Terrain::CHUNK_WIDTH;
    int _x = (id % (Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH)) % Terrain::CHUNK_WIDTH;

    // noise coordinates
    double offset = (double)1/(2*(Terrain::CHUNK_WIDTH-1));
    double y = (_y/Terrain::CHUNK_WIDTH)
        + offset
        + ((double)(_y%Terrain::CHUNK_WIDTH))/Terrain::CHUNK_WIDTH;
    double z = (chunkZ + offset + (double)_z/(Terrain::CHUNK_WIDTH-1));
    double x = (chunkX + offset + (double)_x/(Terrain::CHUNK_WIDTH-1));

    int yGrid0 = (int)floor(y);
    int zGrid0 = (int)floor(z);
    int xGrid0 = (int)floor(x);
    int yGrid1 = yGrid0 + 1;
    int zGrid1 = zGrid0 + 1;
    int xGrid1 = xGrid0 + 1;

    // calculate weights
    double wy = y - (double)yGrid0;
    double wz = z - (double)zGrid0;
    double wx = x - (double)xGrid0;

    double dot1, dot2;
    double interp1, interp2, interp3, interp4, interp5, interp6;
    dot1 = dotProduct3D(yGrid0, zGrid0, xGrid0, y, z, x, gradients);
    dot2 = dotProduct3D(yGrid1, zGrid0, xGrid0, y, z, x, gradients);
    interp1 = interpolate3D(dot1, dot2, wy);

    dot1 = dotProduct3D(yGrid0, zGrid1, xGrid0, y, z, x, gradients);
    dot2 = dotProduct3D(yGrid1, zGrid1, xGrid0, y, z, x, gradients);
    interp2 = interpolate3D(dot1, dot2, wy);

    dot1 = dotProduct3D(yGrid0, zGrid0, xGrid1, y, z, x, gradients);
    dot2 = dotProduct3D(yGrid1, zGrid0, xGrid1, y, z, x, gradients);
    interp3 = interpolate3D(dot1, dot2, wy);

    dot1 = dotProduct3D(yGrid0, zGrid1, xGrid1, y, z, x, gradients);
    dot2 = dotProduct3D(yGrid1, zGrid1, xGrid1, y, z, x, gradients);
    interp4 = interpolate3D(dot1, dot2, wy);

    interp5 = interpolate3D(interp1, interp2, wz);
    interp6 = interpolate3D(interp3, interp4, wz);

    // dig out caves
    if (y <= heightMap[_z*Terrain::CHUNK_WIDTH + _x]) {
      finalVal = interpolate3D(interp5, interp6, wx);
    }

    chunk[id] = finalVal;
};

float* chunkDataKernel(int chunkZ, int chunkX, int* heightMap, double* gradients) {
    // call kernel and return
    float* d_chunk;
    int* d_heightMap;
    double* d_gradients;
    size_t chunkSize = sizeof(float)*Terrain::CHUNK_HEIGHT*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
    float* chunk = new float[Terrain::CHUNK_HEIGHT*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH];
    cudaMalloc(&d_chunk, chunkSize);
    size_t heightMapSize = sizeof(int)*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
    cudaMalloc(&d_heightMap, heightMapSize);
    cudaMemcpy(d_heightMap, heightMap, heightMapSize, cudaMemcpyHostToDevice);
    size_t gradientsSize = sizeof(double)*256*3;
    cudaMalloc(&d_gradients, gradientsSize);
    cudaMemcpy(d_gradients, gradients, gradientsSize, cudaMemcpyHostToDevice);

    // assuming chunk_width is 16, block_width will be 1024
    int block_width = Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH*2;
    dim3 dimBlock(block_width, 1, 1);
    int grid_size = (Terrain::CHUNK_HEIGHT*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH)/block_width;
    dim3 dimGrid(grid_size, 1, 1);
    // std::cout << "block_width: " << block_width << "\n";
    // std::cout << "grid_size: " << grid_size << std::endl;
    cudaMemcpyToSymbol("PERMUTATION", &PERMUTATION, sizeof(PERMUTATION));
    chunkDataKernel<<<dimGrid, dimBlock>>>(chunkZ, chunkX, d_heightMap, d_gradients, d_chunk);
    // printf("Device call:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(chunk, d_chunk, chunkSize, cudaMemcpyDeviceToHost);
    // for (int y = 0; y < Terrain::CHUNK_HEIGHT; ++y) {
    //     for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
    //         for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
    //             std::cout << chunk[y*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH + z*Terrain::CHUNK_WIDTH + x] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "------------------" << std::endl;
    // }
    return chunk;
};


// int main(int argc, char** argv) {
//     Terrain terrain(2022);
//     auto heightMap = terrain.generateChunkHeightMapGpu(0,0);
//     int* d_chunk;
//     size_t chunkSize = sizeof(int)*Terrain::CHUNK_HEIGHT*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
//     size_t heightMapSize = sizeof(int)*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
//     size_t gradientsSize = sizeof(double)*256*3;
//     cudaMemcpyToSymbol(c_gradients, terrain.noise3D.gradientsGPU, gradientsSize);

//     int block_width = Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH*2;
//     dim3 dimBlock(block_width, 1, 1);
//     int grid_size = (Terrain::CHUNK_HEIGHT*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH)/block_width;
//     dim3 dimGrid(grid_size, 1, 1);


//     // setup gpu timers
//     cudaEvent_t start, stop;
//     float time;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);

//     // using constant memory for gradients and heightMap & shared memory
//     for (int i = 0; i < 1000 ; ++i) {
//         cudaMalloc(&d_chunk, chunkSize);
//         cudaMemcpyToSymbol(c_heightMap, heightMap, heightMapSize);
//         chunkDataKernelOpt<<<dimGrid, dimBlock>>>(i, i, d_chunk);
//         cudaFree(d_chunk);
//     }
//     // stop gpu timers
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop); // after cudaEventRecord
//     cudaEventElapsedTime(&time, start, stop);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     printf("time elapsed after 100 kernel executions: %fms\n", time);
// }
