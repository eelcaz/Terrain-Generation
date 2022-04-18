#include <iostream>
#include <fstream>
#include <iomanip>

#include "terrain_mesh.h"
#include "perlin_noise.h"

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
            val = (int)(val * TERRAIN_AMPLITUDE);
            heightMap[z][x] = val;
        }
    }

    return heightMap;
};

void write_ppm(std::string filename, std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> pic) {

    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH << "\n" << 30 << "\n";

    for (int i=0; i<CHUNK_WIDTH; i++) {
        for (int j=0; j<CHUNK_WIDTH; j++) {
            unsigned char uc = pic[i][j];
            fp << uc << uc << uc;
        }
    }
    fp << std::endl;
    fp.close();
};

void chunkImageWriter(std::string filename,
                      std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c1,
                      std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c2) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH*2 << " " << CHUNK_WIDTH << "\n" << 150 << "\n";

    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH*2; ++j) {
            unsigned char uc;
            if (j < CHUNK_WIDTH) {
                uc = c1[i][j];
            } else {
                uc = c2[i][j];
            }
            fp << uc << uc << uc;
        }
    }
    fp << std::endl;
    fp.close();
};

void chunkImageWriterVert2(std::string filename, int** c1, int** c2) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH*2 << "\n" << 100 << "\n";

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            unsigned char uc = (unsigned char)c1[z][x];
            fp << uc << uc << uc;
        }
    }
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            unsigned char uc = (unsigned char)c2[z][x];
            fp << uc << uc << uc;
        }
    }
    fp << std::endl;
    fp.close();
};

void chunkImageWriterList(std::string filename, int*** chunkList) {
    int chunkListLen = sizeof(chunkList);
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH*chunkListLen << "\n" << 100 << "\n";
    for (int i = 0; i < chunkListLen; ++i) {
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

void chunkWriter(std::string filename,
                 std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c1) {
    std::ofstream fp;
    fp.open(filename);

    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH; ++j) {
            fp << (int)c1[i][j] << " ";
        }
        fp << "\n";
    }
    fp << std::endl;
    fp.close();
};

void chunkWriterVert2(std::string filename, int** c1, int** c2) {
    std::ofstream fp;
    fp.open(filename);

    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            fp << (int)c1[z][x] << " ";
        }
        fp << "\n";
    }
    for (int z = 0; z < CHUNK_WIDTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            fp << (int)c2[z][x] << " ";
        }
        if (z != CHUNK_WIDTH - 1) {
            fp << "\n";
        }
    }
    fp.close();
};

void chunkWriterList(std::string filename, int*** chunkList) {
    int chunkListLen = sizeof(chunkList);
    std::ofstream fp;
    fp.open(filename);
    for (int i = 0; i < chunkListLen; ++i) {
        for (int z = 0; z < CHUNK_WIDTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                fp << (int)chunkList[i][z][x] << " ";
            }
            if (z != CHUNK_WIDTH - 1 || i != chunkListLen - 1) {
                fp << "\n";
            }
        }
    }
    fp.close();
}
int main(int argc, char *argv[]) {
    // auto c0_0 = Terrain::generateChunkHeightMap(5, 0);
    // auto c0_1 = Terrain::generateChunkHeightMap(6, 0);
    auto c1 = Terrain::generateChunkHeightMapInt(5, 57);
    auto c2 = Terrain::generateChunkHeightMapInt(6, 57);
    std::string str = "perlin.ppm";
    // std::cout << CHUNK_WIDTH << "\n" << std::endl;
    std::cout << std::endl;
    chunkImageWriterVert(str, c1, c2);
    std::string mname = "perlin.txt";
    // chunkWriter(mname, c1);
    chunkWriterVert(mname, c1, c2);
}
