#include <iostream>
#include <iomanip>
#include "terrain_generator.h"

__constant__ static constexpr int PERMUTATION[] = {
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

__device__ double interpolate(double a, double b, double weight) {
    if (weight < 0) return a;
    if (weight > 1) return b;
    return (b - a) * ((weight * (weight * 6.0 - 15.0) + 10.0) * weight * weight * weight) + a;
};

__device__ double dotProduct(int GridZ, int GridX, double pz, double px) {
    // get the random vector on the gridPoint
    int randDir = PERMUTATION[(PERMUTATION[GridZ] + GridX) % 256];
    double gradZ = cos((double)randDir);
    double gradX = sin((double)randDir);
    // get the offset vector from the grid point to the target point
    double offsetZ = pz-(double)GridZ;
    double offsetX = px-(double)GridX;

    return gradZ * offsetZ + gradX * offsetX;
};

__global__ void chunkHeightMapKernel(int chunkZ, int chunkX, double *chunk) {
    double offset = (double)1/(2*CHUNK_WIDTH);
    int _z = threadIdx.x / CHUNK_WIDTH;
    int _x = threadIdx.x % CHUNK_WIDTH;
    double z = (chunkZ + offset + (double)_z/CHUNK_WIDTH)/TERRAIN_ZOOM;
    double x = (chunkX + offset + (double)_x/CHUNK_WIDTH)/TERRAIN_ZOOM;
    double noiseZ, noiseX;

    // fbm iterations
    int octaves = 6;
    double total = 0.0;
    double maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        float amplitude = pow(0.58f, i);
        float frequency = pow(2.0f, i);

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
        dot1 = dotProduct(zGrid0, xGrid0, noiseZ, noiseX);
        dot2 = dotProduct(zGrid1, xGrid0, noiseZ, noiseX);
        interp1 = interpolate(dot1, dot2, wz);

        dot1 = dotProduct(zGrid0, xGrid1, noiseZ, noiseX);
        dot2 = dotProduct(zGrid1, xGrid1, noiseZ, noiseX);
        interp2 = interpolate(dot1, dot2, wz);

        double noiseVal = interpolate(interp1, interp2, wx);
        total += noiseVal * amplitude;
        maxVal += amplitude;
    }
    total = total/maxVal;
    // apply terrain calcs
    total = (total + 1)/2;
    total = (int)floor(total * TERRAIN_AMPLITUDE);
    chunk[threadIdx.x] = total;
    // chunk[threadIdx.x] = CHUNK_WIDTH;
    return;
};


int main(int argc, char *argv[]) {
    double* d_chunk;
    size_t chunkSize = sizeof(double) * CHUNK_WIDTH * CHUNK_WIDTH;
    double* chunk = new double[CHUNK_WIDTH*CHUNK_WIDTH];
    cudaMalloc(&d_chunk, chunkSize);
    int block_width = CHUNK_WIDTH*CHUNK_WIDTH;
    dim3 dimBlock(block_width, 1, 1);
    dim3 dimGrid(1, 1, 1);
    cudaMemcpyToSymbol("PERMUTATION", &PERMUTATION, sizeof(PERMUTATION));
    chunkHeightMapKernel<<<dimGrid, dimBlock>>>(0, 0, d_chunk);
    cudaMemcpy(chunk, d_chunk, chunkSize, cudaMemcpyDeviceToHost);

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            std::cout << std::left << std::setw(12)
                      << chunk[z*CHUNK_WIDTH + x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] chunk;
    return 0;
}
