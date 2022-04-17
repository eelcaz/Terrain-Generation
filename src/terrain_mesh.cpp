#include <array>
#include "perlin_noise.h"
#include <iostream>
#include <fstream>
#include <iomanip>


const int CHUNK_WIDTH = 256; // 16 * 16,

struct chunk {
    int width;
};

class Terrain {
private:
    static float fbmNoise(float x, float y, int octaves);
    static float fbmNoise2(float x, float y, int octaves);
public:
    static std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMap(int x, int y);
    static std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMapInt(int chunkX, int chunkY);
    static std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH>
    generateChunkHeightMapInt2(int chunkX, int chunkY);
};

float Terrain::fbmNoise(float x, float y, int octaves) {
    // placeholder values, will probable want this customizable in the future
    float total = 0;
    float frequency = 1;
    float amplitude = 1;
    float gain = 4;
    float maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        total += PerlinNoise::noise(x*frequency, y*frequency) * amplitude;
        maxVal += amplitude;
        amplitude = pow(amplitude, 0.58);
        frequency *= 2;
    }
    // for (i = 0; i < octaves; ++i) {
    //     total += noise((float)x * frequency,
    //                    (float)y * frequency) * amplitude;
    //     frequency *= lacunarity;
    //     amplitude *= gain;
    // }
    // map[x][y]=total;
    return total/maxVal;
};

float Terrain::fbmNoise2(float x, float y, int octaves) {
    // placeholder values, will probable want this customizable in the future
    float total = 0.0;
    // float amplitude = 0.5;
    // float gain = 4;
    // float frequency = 1;
    float maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        float amplitude = pow(0.58f, i);
        float frequency = pow(2.0f, i);
        total += PerlinNoise::noise(x*frequency, y*frequency) * amplitude;
        maxVal += amplitude;
    }
    // for (i = 0; i < octaves; ++i) {
    //     total += noise((float)x * frequency,
    //                    (float)y * frequency) * amplitude;
    //     frequency *= lacunarity;
    //     amplitude *= gain;
    // }
    // map[x][y]=total;
    return total/maxVal;
};

std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH>
Terrain::generateChunkHeightMap(int chunkX, int chunkY) {
    std::array<std::array<float, CHUNK_WIDTH>, CHUNK_WIDTH> heightMap;

    for (int x = 0; x < CHUNK_WIDTH; ++x) {
        for (int y = 0; y < CHUNK_WIDTH; ++y) {
            float offset = 0.5/CHUNK_WIDTH;
            // float val = fbmNoise(
            //     chunkX + ((float)x/CHUNK_WIDTH) + offset,
            //     chunkY + ((float)y/CHUNK_WIDTH) + offset,
            //     4
            // );
            // heightMap[x][y] = (val + 1.0f)/2.0f; // make positive [0-1]
            float val = PerlinNoise::noise(
                chunkX + ((float)x/CHUNK_WIDTH) + offset,
                chunkY + ((float)y/CHUNK_WIDTH) + offset
            );
            val = (val + 1.0f)/2.0f;
            heightMap[x][y] = (val * 105) - 18;
        }
    }

    // change values into a more usable scale
    return heightMap;
};

std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH>
Terrain::generateChunkHeightMapInt(int chunkX, int chunkY) {
    std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> heightMap;

    for (int y = 0; y < CHUNK_WIDTH; ++y) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            float offset = 0.5/CHUNK_WIDTH;
            float val = fbmNoise2(
                chunkX + ((float)x/CHUNK_WIDTH) + offset,
                chunkY + ((float)y/CHUNK_WIDTH) + offset,
                4
            );
            val = (val + 1.0f)/2.0f;
            heightMap[x][y] = static_cast<unsigned char>((val * 105) - 36);
        }
    }

    // change values into a more usable scale
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

void chunkImageWriterVert(std::string filename,
                          std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c1,
                          std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c2) {
    std::ofstream fp;
    fp.open(filename);

    fp << "P6\n";
    fp << CHUNK_WIDTH << " " << CHUNK_WIDTH*2 << "\n" << 50 << "\n";

    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH; ++j) {
            unsigned char uc = c1[i][j];
            fp << uc << uc << uc;
        }
    }
    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH; ++j) {
            unsigned char uc = c2[i][j];
            fp << uc << uc << uc;
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

void chunkWriterVert(std::string filename,
                     std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c1,
                     std::array<std::array<unsigned char, CHUNK_WIDTH>, CHUNK_WIDTH> c2) {
    std::ofstream fp;
    fp.open(filename);

    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH; ++j) {
            fp << (int)c1[i][j] << " ";
        }
        fp << "\n";
    }
    for (int i = 0; i < CHUNK_WIDTH; ++i) {
        for (int j = 0; j < CHUNK_WIDTH; ++j) {
            fp << (int)c2[i][j] << " ";
        }
        fp << "\n";
    }
    fp << std::endl;
    fp.close();
};
                

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
