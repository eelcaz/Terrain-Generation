#include <GL/glew.h>
#include <vector>
void marchingCubesKernel(size_t* slices, GLfloat* vertices_3D, std::vector<int*> chunks, size_t num, unsigned int triTable[256][16], unsigned int case_to_numpolys[256]);
void slicesKernel(size_t* slices, std::vector<int*> chunks, unsigned int triTable[256][16], unsigned int case_to_numpolys[256]);