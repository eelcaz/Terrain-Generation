#include <iostream>
#include <iomanip>
#include "terrain_generator.h"

__device__ double interpolate(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

__device__ double dotProduct(int GridZ, int GridX, double pz, double px, int* permutation) {
    // get the random vector on the gridPoint
    int randDir = permutation[(permutation[abs(GridZ) % 256] + abs(GridX)) % 256];
    double gradZ = cos((double)randDir);
    double gradX = sin((double)randDir);
    // get the offset vector from the grid point to the target point
    double offsetZ = pz-(double)GridZ;
    double offsetX = px-(double)GridX;

    return gradZ * offsetZ + gradX * offsetX;
};

__global__ void chunkHeightMapKernel(int chunkZ, int chunkX, int* chunk, int* permutation) {
    double offset = (double)1/(2*(Terrain::CHUNK_WIDTH-1));
    int _z = threadIdx.x / Terrain::CHUNK_WIDTH;
    int _x = threadIdx.x % Terrain::CHUNK_WIDTH;
    double z = (chunkZ + offset + (double)_z/(Terrain::CHUNK_WIDTH-1))/Terrain::TERRAIN_ZOOM;
    double x = (chunkX + offset + (double)_x/(Terrain::CHUNK_WIDTH-1))/Terrain::TERRAIN_ZOOM;
    double noiseZ, noiseX;

    // fbm iterations
    int octaves = 6;
    double total = 0.0;
    double maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
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
        dot1 = dotProduct(zGrid0, xGrid0, noiseZ, noiseX, permutation);
        dot2 = dotProduct(zGrid1, xGrid0, noiseZ, noiseX, permutation);
        interp1 = interpolate(dot1, dot2, wz);

        dot1 = dotProduct(zGrid0, xGrid1, noiseZ, noiseX, permutation);
        dot2 = dotProduct(zGrid1, xGrid1, noiseZ, noiseX, permutation);
        interp2 = interpolate(dot1, dot2, wz);

        double noiseVal = interpolate(interp1, interp2, wx);
        total += noiseVal * amplitude;
        maxVal += amplitude;
    }
    total = total/maxVal;
    // apply terrain calcs
    total = (total + 1)/2;
    total = (int)floor(total * Terrain::TERRAIN_AMPLITUDE);
    chunk[threadIdx.x] = total;
    return;
};

int* chunkHeightMapKernel(int chunkZ, int chunkX, int* permutation) {
    int* d_chunk;
    int* d_permutation;
    size_t chunkSize = sizeof(int)*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
    int* chunk = new int[Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH];
    cudaMalloc(&d_chunk, chunkSize);
    size_t permutationSize = sizeof(int)*256;
    cudaMalloc(&d_permutation, permutationSize);
    cudaMemcpy(d_permutation, permutation, permutationSize, cudaMemcpyHostToDevice);
    int block_width = Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
    dim3 dimBlock(block_width, 1, 1);
    dim3 dimGrid(1, 1, 1);
    chunkHeightMapKernel<<<dimGrid, dimBlock>>>(chunkZ, chunkX, d_chunk, d_permutation);
    cudaMemcpy(chunk, d_chunk, chunkSize, cudaMemcpyDeviceToHost);
    return chunk;
}


// int main(int argc, char *argv[]) {
//     int* d_chunk;
//     size_t chunkSize = sizeof(int) * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH;
//     chunk* chunk = new int[Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH];
//     cudaMalloc(&d_chunk, chunkSize);
//     int block_width = Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
//     dim3 dimBlock(block_width, 1, 1);
//     dim3 dimGrid(1, 1, 1);
//     cudaMemcpyToSymbol("PERMUTATION", &PERMUTATION, sizeof(PERMUTATION));
//     chunkHeightMapKernel<<<dimGrid, dimBlock>>>(0, 0, d_chunk);
//     cudaMemcpy(chunk, d_chunk, chunkSize, cudaMemcpyDeviceToHost);

//     for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
//         for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
//             std::cout << std::left << std::setw(12)
//                       << chunk[z*Terrain::CHUNK_WIDTH + x] << " ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << std::endl;
//     delete[] chunk;
//     return 0;
// }
