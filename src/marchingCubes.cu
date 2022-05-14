#include <GL/glew.h>
#include <device_launch_parameters.h>
#include <vector>
#include <glm/glm.hpp>
#include <terrain_generator.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <chrono>

__constant__ unsigned int tris[256][16];
__constant__ unsigned int c2np[256];

#define CH 256
#define NCS 4
#define CW 16

__global__ void marchingCubesGPU(size_t* slices, GLfloat* vertices, float* chunks) {
    int k = threadIdx.x;
    int l = blockIdx.y;
    int m = blockIdx.x;
    int chunkA = l * 2 * NCS + m;
    //printf("%d\n", chunkA);
    int chunk = chunkA * CW * CW * CH;
    size_t index = slices[(CH) * chunkA + k];
    if (slices[(CH) * chunkA + k + 1] - index == 0) return;
    //printf("%d\n", index);
    //printf("%d\n", l);
    //printf("%ld\n", index);
    //printf("%d\n", chunks[0]);
    /*if (k == 0 && m == 0 && l == 0) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%d ", tris[i][j]);
            }
            printf("\n");
        }
        printf("im alive\n");
    }*/
    for (int i = 0; i < CW - 1; i++) {
        for (int j = 0; j < CW - 1; j++) {
            //size_t curVoxel = 90 * (k * w * w + i * w + j);
            int b = 0;

            // flat array indexing for gpu returned chunks
            int k_ = k * CW * CW;
            int k_1 = k_ + CW * CW;
            int i_ = i * CW;
            int i_1 = i_ + CW;
            int j_ = j;
            int j_1 = j_ + 1;
            b += chunks[chunk+ k_ + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v7
            b <<= 1;
            b += chunks[chunk+k_1 + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v6
            b <<= 1;
            b += chunks[chunk+k_1 + i_ + j_1] > Terrain::CAVE_INTENSITY; // v5
            b <<= 1;
            b += chunks[chunk+k_ + i_ + j_1] > Terrain::CAVE_INTENSITY; // v4
            b <<= 1;
            b += chunks[chunk+k_ + i_1 + j_] > Terrain::CAVE_INTENSITY; // v3
            b <<= 1;
            b += chunks[chunk+k_1 + i_1 + j_] > Terrain::CAVE_INTENSITY; // v2
            b <<= 1;
            b += chunks[chunk+k_1 + i_ + j_] > Terrain::CAVE_INTENSITY; // v1
            b <<= 1;
            b += chunks[chunk+k_ + i_ + j_] > Terrain::CAVE_INTENSITY; // v0
            //printf("%d\n", b);
            unsigned int numTriangles = c2np[b];
            GLfloat edges[12][3] = {
                {i + 0.0f, j + 0.0f, k + 0.5f},    // e0
                {i + 0.5f, j + 0.0f, k + 1.0f},    // e1
                {i + 1.0f, j + 0.0f, k + 0.5f},    // e2
                {i + 0.5f, j + 0.0f, k + 0.0f},    // e3
                {i + 0.0f, j + 1.0f, k + 0.5f},    // e4
                {i + 0.5f, j + 1.0f, k + 1.0f},    // e5
                {i + 1.0f, j + 1.0f, k + 0.5f},    // e6
                {i + 0.5f, j + 1.0f, k + 0.0f},    // e7
                {i + 0.0f, j + 0.5f, k + 0.0f},    // e8
                {i + 0.0f, j + 0.5f, k + 1.0f},    // e9
                {i + 1.0f, j + 0.5f, k + 1.0f},    // e10
                {i + 1.0f, j + 0.5f, k + 0.0f}     // e11
            };
            int d = 1;

            //glm::vec3 grad(0, 0, 0);
            int k_d = k_ + d * CW * CW;
            int i_d = i_ + d * CW;
            int j_d = j_ + d;
            float gx = (float)chunks[chunk+k_ + i_ + j_d] - chunks[chunk+k_ + i_ + j_];
            float gy = (float)chunks[chunk+k_d + i_ + j_] - chunks[chunk+k_ + i_ + j_];
            float gz = (float)chunks[chunk+k_ + i_d + j_] - chunks[chunk+k_ + i_ + j_];
            //grad = -normalize(grad);
            //printf("%d\n", numTriangles);
            for (unsigned int iterate = 0; iterate < numTriangles; iterate++) {
                for (int ij = 0; ij < 3; ij++) {
                    auto curEdge = tris[b][iterate * 3 + ij];
                    auto px = edges[curEdge][1] + (CW - 1) * (m-NCS);
                    auto py = edges[curEdge][2];
                    auto pz = edges[curEdge][0] + (CW - 1) * (l-NCS);
                    vertices[index++] = px*2;
                    vertices[index++] = py*2;
                    vertices[index++] = pz*2;
                    vertices[index++] = -gx;
                    vertices[index++] = -gy;
                    vertices[index++] = -gz;
                }
            }
        }

        //__syncthreads();
    }
}

__global__ void getSlicesGPU(size_t* slices, float* chunks) {
    // represents the current chunk
    int k = threadIdx.x;
    int l = blockIdx.y;
    int m = blockIdx.x;
    //input1[tx] = input[tx];
    int i, j;
    int chunkA = l * 2 * NCS + m;
    //printf("%d\n", chunkA);
    int chunk = chunkA * CW * CW * CH;
    //slices[curSlice++] = num;
    int num = 0;
    for (i = 0; i < Terrain::CHUNK_WIDTH - 1; i++) {
        for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
            int b = 0;

            // flat array indexing for gpu returned chunks
            int k_ = k * CW * CW;
            int k_1 = k_ + CW * CW;
            int i_ = i * CW;
            int i_1 = i_ + CW;
            int j_ = j;
            int j_1 = j_ + 1;
            b += chunks[chunk + k_ + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v7
            b <<= 1;
            b += chunks[chunk + k_1 + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v6
            b <<= 1;
            b += chunks[chunk + k_1 + i_ + j_1] > Terrain::CAVE_INTENSITY; // v5
            b <<= 1;
            b += chunks[chunk + k_ + i_ + j_1] > Terrain::CAVE_INTENSITY; // v4
            b <<= 1;
            b += chunks[chunk + k_ + i_1 + j_] > Terrain::CAVE_INTENSITY; // v3
            b <<= 1;
            b += chunks[chunk + k_1 + i_1 + j_] > Terrain::CAVE_INTENSITY; // v2
            b <<= 1;
            b += chunks[chunk + k_1 + i_ + j_] > Terrain::CAVE_INTENSITY; // v1
            b <<= 1;
            b += chunks[chunk + k_ + i_ + j_] > Terrain::CAVE_INTENSITY; // v0
            //printf("%d\n", b);
            unsigned int numTriangles = c2np[b];
            num += numTriangles * 18;
        }
    }
    if (k == 256) num = 0;
    __syncthreads();
    slices[chunkA * CH + k+1] = num;

}

void slicesKernel(size_t* slices, std::vector<float*> chunks, unsigned int triTable[256][16], unsigned int case_to_numpolys[256]) {
    float* chunks_host = (float*)calloc(chunks.size() * CW * CW * CH, sizeof(float));
    size_t index = 0;
    //thrust::device_vector<int> dummy_vec(chunks.size() * (CH)+1);
    for (int i = 0; i < chunks.size(); i++) {
        for (int j = 0; j < CW * CW * CH; j++) {
            chunks_host[index++] = chunks[i][j];
        }
    }

    float* chunks_device;
    size_t* slices_device;

    size_t chunks_size = chunks.size() * CW * CW * CH * sizeof(float);
    size_t slices_size = (chunks.size() * (CH)+1) * sizeof(size_t);

    cudaMalloc((void**)&chunks_device, chunks_size);
    cudaMalloc((void**)&slices_device, slices_size);

    //thrust::host_vector<size_t> h_data(chunks.size() * (CH)+1);
    //thrust::device_vector<size_t> d_data(chunks.size() * CH + 1);

    cudaMemcpy(chunks_device, chunks_host, chunks_size, cudaMemcpyHostToDevice);
    cudaMemcpy(slices_device, slices, slices_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c2np, case_to_numpolys, 256 * sizeof(unsigned int));
    cudaMemcpyToSymbol(tris, triTable, 256 * 16 * sizeof(unsigned int));

    dim3 blockDim, gridDim;
    gridDim = dim3(2 * NCS, 2 * NCS);
    blockDim = dim3(CH);
    //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
      //  gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //thrust::fill(h_data.begin(), h_data.end(), 0);
    //thrust::copy(h_data.begin(), h_data.end(), d_data.begin());

    //std::cout << "warm up iteration " << d_data[0] << std::endl;
    //thrust::fill(d_data.begin(), d_data.end(), 0);
    //thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    //std::cout << "warm up iteration " << h_data[0] << std::endl;
    //thrust::fill(h_data.begin(), h_data.end(), 0);

    cudaEventRecord(start);

    getSlicesGPU << <gridDim, blockDim >> > (slices_device, chunks_device);
    //printf("Device call:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Marching Cubes Pt 1 (GPU): %f milliseconds\n", time);
    cudaMemcpy(slices, slices_device, slices_size, cudaMemcpyDeviceToHost);

    
    //thrust::device_vector<size_t> d_data(slices, slices + chunks.size() * (CH)+1);

    //for (int i = 0; i < 512; i++) {
      //  printf("%d, %d\n", i, slices[i]);
    //}
    /*for (int i = 0; i < (chunks.size() * (CH)+1); i++) {
        d_data[i] = slices[i];
    }*/

    //cudaEventRecord(start);

    //thrust::inclusive_scan(d_data.begin(), d_data.end(), d_data.begin());

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&time, start, stop);

    //printf("Marching Cubes Pt 1 Thrust (GPU): %f milliseconds\n", time);

    //thrust::copy(d_data.begin(), d_data.end(), slices);
    
    /*for (int i = 0; i < h_data.size(); i++) {
        slices[i] = h_data[i];
    }*/
    /*for (int i = 0; i < 512; i++) {
        printf("%d, %d\n", i, slices[i]);
    }*/
    //clock_t begin = clock();
    //cudaEventRecord(start);
    std::chrono::time_point<std::chrono::system_clock> start2, end2;

    start2 = std::chrono::system_clock::now();

    size_t tracker = 0;
    for (int i = 0; i < (chunks.size() * (CH)+1); i++) {
        size_t temp = slices[i];
        slices[i] += tracker;
        tracker += temp;
    }
    end2 = std::chrono::system_clock::now();
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&time, start, stop);
    //clock_t end = clock();
    //double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //time_spent *= 1000;
    std::chrono::duration<double> elapsed_seconds = end2 - start2;
    printf("Marching Cubes Pt 1 Non-Thrust (CPU): %f milliseconds\n", elapsed_seconds.count() * 1000);
    /*for (int i = 0; i < 512; i++) {
        printf("%d, %d\n", i, slices[i]);
    }*/
    free(chunks_host);
    cudaFree(slices_device);
    cudaFree(chunks_device);
}

void marchingCubesKernel(size_t* slices, GLfloat* vertices_3D, std::vector<float*> chunks, size_t num, unsigned int triTable[256][16], unsigned int case_to_numpolys[256]) {
    float* chunks_host = (float*)calloc(chunks.size() * CW * CW * CH, sizeof(float));
    size_t index = 0;

    for (int i = 0; i < chunks.size(); i++) {
        for (int j = 0; j < CW * CW * CH; j++) {
            chunks_host[index++] = chunks[i][j];
        }
    }

    float* chunks_device;
    GLfloat* vertices_device;
    size_t* slices_device;

    size_t chunks_size = chunks.size() * CW * CW * CH * sizeof(float);
    size_t vertices_size = num * sizeof(GLfloat);
    size_t slices_size = (chunks.size() * (CH) +1)* sizeof(size_t);

    cudaMalloc((void**)&chunks_device, chunks_size);
    cudaMalloc((void**)&vertices_device, vertices_size);
    cudaMalloc((void**)&slices_device, slices_size);

    cudaMemcpy(chunks_device, chunks_host, chunks_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vertices_device, vertices_3D, vertices_size, cudaMemcpyHostToDevice);
    cudaMemcpy(slices_device, slices, slices_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c2np, case_to_numpolys, 256 * sizeof(unsigned int));
    cudaMemcpyToSymbol(tris, triTable, 256 * 16 * sizeof(unsigned int));

    dim3 blockDim, gridDim;
    gridDim = dim3(2 * NCS, 2 * NCS);
    blockDim = dim3(CH - 1);
    //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
      //  gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    
    cudaEvent_t start2, stop2;
    float time2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    //printf("%d\n", NCS);
    marchingCubesGPU << <gridDim, blockDim >> > (slices_device, vertices_device, chunks_device);
    //printf("Device call:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    //cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time2, start2, stop2);
    cudaMemcpy(vertices_3D, vertices_device, vertices_size, cudaMemcpyDeviceToHost);

    printf("Marching Cubes Pt 2 (GPU): %f milliseconds\n", time2);
    free(chunks_host);
    //for (int i = 0; i < 512; i++) {
      //  printf("%d, %d\n", i, slices[i]);
    //}
    cudaFree(vertices_device);
    cudaFree(slices_device);
    cudaFree(chunks_device);
}