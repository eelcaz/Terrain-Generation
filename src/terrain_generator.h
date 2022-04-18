#include <array>

static const int CHUNK_WIDTH = 16; // chunk[z][x] z=x=CHUNK_WIDTH

class Terrain {
private:
    static double fbmNoise(double z, double x, int octaves);
    static const int TERRAIN_AMPLITUDE = 200;
    /* good zoom values seem to be 256/CHUNK_WIDTH */
    static const int TERRAIN_ZOOM = 16;
public:;
    static int** generateChunkHeightMap(int z, int x);
    static int*** generateChunkData(int z, int x);
};
