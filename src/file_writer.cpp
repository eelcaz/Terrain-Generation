#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "terrain_generator.h"

void imageWriteChunk(std::string filename, int** chunk) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n" << Terrain::CHUNK_WIDTH << " ";
    fp << Terrain::CHUNK_WIDTH << "\n" << 200 << "\n";

    for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
        for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
            unsigned char uc;
            uc = chunk[z][x];
            fp << uc << uc << uc;
        }
    }
    fp << std::endl;
    fp.close();
};

void imageWriteChunkList(std::string filename, int*** chunkList, int len) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n" << Terrain::CHUNK_WIDTH << " ";
    fp << Terrain::CHUNK_WIDTH*len << "\n" << 200 << "\n";

    for (int i = 0; i < len; ++i) {
        for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
            for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
                unsigned char uc = (unsigned char)chunkList[i][z][x];
                fp << uc << uc << uc;
            }
        }
    }
    fp << std::endl;
    fp.close();
};

void imageWrite3DChunk(std::string filename, int*** chunk) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n" << Terrain::CHUNK_WIDTH << " ";
    fp << Terrain::CHUNK_WIDTH*Terrain::CHUNK_HEIGHT;
    fp << "\n" << 1 << "\n";

    for (int y = 0; y < Terrain::CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
            for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
                unsigned char uc = (unsigned char)chunk[y][z][x];
                fp << uc << uc << uc;
            }
        }
    }
    fp << std::endl;
    fp.close();
};

void plainTextWriteChunk(std::string filename, int** chunk) {
    std::ofstream fp;
    fp.open(filename);

    for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
        for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
            fp << (int)floor(chunk[z][x]) << " ";
        }
        if (z != Terrain::CHUNK_WIDTH - 1) {
            fp << std::endl;
        }
    }
    fp << std::endl;
    fp.close();
};

void plainTextWriteChunkList(std::string filename, int*** chunkList, int len) {
    std::ofstream fp;
    fp.open(filename);
    for (int i = 0; i < len; ++i) {
        for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
            for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
                fp << (int)floor(chunkList[i][z][x]) << " ";
            }
            if (z != Terrain::CHUNK_WIDTH - 1 || i != len - 1) {
                fp << std::endl;
            }
        }
    }
    fp.close();
}

void plainTextWrite3DChunk(std::string filename, int*** chunk, int chunkWidth, int chunkHeight) {
    std::ofstream fp;
    fp.open(filename);
    for (int y = 0; y < chunkHeight; ++y) {
        for (int z = 0; z < chunkWidth; ++z) {
            for (int x = 0; x < chunkWidth; ++x) {
                fp << chunk[y][z][x] << " ";
            }
            fp << std::endl;
        }
        fp << "--------------------------------\n";
    }
    fp << std::endl;
    fp.close();
}

// int main(int argc, char *argv[]) {
//     int numChunks = 16;
//     auto chunkList = new int **[numChunks];
//     for (int i = 0; i < numChunks; ++i) {
//         chunkList[i] = Terrain::generateChunkHeightMap(i-numChunks/2, 0);
//     }
//     imageWriteChunk("chunk.ppm", chunkList[0]);
//     imageWriteChunkList("chunks.ppm", chunkList, numChunks);
//     plainTextWriteChunk("chunk.txt", chunkList[0]);
//     plainTextWriteChunkList("chunks.txt", chunkList, numChunks);
//     // deallocate
//     for (int i = 0; i < numChunks; ++i) {
//         delete[] chunkList[i];
//     }
//     delete[] chunkList;

//     // create 3D chunk data
//     auto chunk3D = Terrain::generateChunkData(0, 0);
//     plainTextWrite3DChunk("chunk3D.txt", chunk3D, 16, 256);
//     imageWrite3DChunk("chunk3D.ppm", chunk3D);
//     Terrain::deallocateChunk(chunk3D);
//     return 0;
// }
