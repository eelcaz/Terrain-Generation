#include <iostream>
#include <fstream>
#include <iomanip>

#include "terrain_mesh.h"

void imageWriteChunk(std::string filename, int** chunk) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH << "\n" << 200 << "\n";

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
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

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH*len << "\n" << 200 << "\n";
    for (int i = 0; i < len; ++i) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                unsigned char uc = (unsigned char)chunkList[i][z][x];
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

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            fp << (int)chunk[z][x] << " ";
        }
        if (z != CHUNK_WIDTH - 1) {
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
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                fp << (int)chunkList[i][z][x] << " ";
            }
            if (z != CHUNK_WIDTH - 1 || i != len - 1) {
                fp << std::endl;
            }
        }
    }
    fp.close();
}

int main(int argc, char *argv[]) {
    auto chunkList = new int **[16];
    for (int i = 0; i < 16; ++i) {
        chunkList[i] = Terrain::generateChunkHeightMap(i, 0);
    }
    imageWriteChunk("1chunk.ppm", chunkList[0]);
    imageWriteChunkList("16chunks.ppm", chunkList, 16);
    plainTextWriteChunk("1chunk.txt", chunkList[0]);
    plainTextWriteChunkList("16chunks.txt", chunkList, 16);
    // deallocate
    for (int i = 0; i < 16; ++i) {
        delete[] chunkList[i];
    }
    delete[] chunkList;
    return 0;
}
