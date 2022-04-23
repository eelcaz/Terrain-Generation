class Terrain {
private:
    static double fbmNoise(double z, double x, int octaves);
public:
    // constants
    // chunk[y][z][x] z=x=CHUNK_WIDTH, y=CHUNK_HEIGHT
    static const int CHUNK_WIDTH = 16;
    static const int CHUNK_HEIGHT = 256;
    // good zoom values seem to be 256/CHUNK_WIDTH
    static const int TERRAIN_ZOOM = 16;
    static const int TERRAIN_AMPLITUDE = 200;

    static int** generateChunkHeightMap(int chunkZ, int chunkX);
    static int*** generateChunkData(int chunkZ, int chunkX);
    static void deallocateChunk(int*** chunk);
};
