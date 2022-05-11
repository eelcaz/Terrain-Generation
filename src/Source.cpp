#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <windows.h>
#include <GL/glew.h> // include this one first
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>
#include <terrain_generator.h>
#include <shaders.h>
#include <camera.h>
#include <tables.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <marchingCubes.h>

#define SEED 2022
// VERSION: 0 = CPU, 1 = GPU, 2 = GPU Optimized
#define VERSION 0

/*
double thing(int x, int y, int z, int l, int m, Terrain terrain) {
    double offset = (double)1 / (2 * Terrain::CHUNK_WIDTH);
    double fy = ((int)floor(y / Terrain::CHUNK_WIDTH)) + offset + ((double)(y % Terrain::CHUNK_WIDTH)) / Terrain::CHUNK_WIDTH;
    double fz = (l + offset + (double)z / Terrain::CHUNK_WIDTH);
    double fx = (m + offset + (double)x / Terrain::CHUNK_WIDTH);
    double val = terrain.noise3D.noise(fy * Terrain::CAVE_ZOOM,
        fz * Terrain::CAVE_ZOOM,
        fx * Terrain::CAVE_ZOOM);
    return val;
}
*/

// location = 0 bc of attrib pointer



int main(int argc, char** argv) {
    int i = 0;
    int j = 0;
    int k = 0;

#if VERSION == 1
    std::cout << "In GPU implementation" << std::endl;
#elif VERSION == 2
    std::cout << "In Optimized GPU implementation" << std::endl;
#else
    std::cout << "In CPU implementation" << std::endl;
#endif

    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "woo", NULL, NULL);

    if (window == NULL) {
        const char* error;
        glfwGetError(&error);
        std::cout << "Failed to create GLFW window" << error << std::endl;
    }

    // you need a context before you initialize
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;

    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW " << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    std::cout << glGetString(GL_VERSION) << std::endl;


    // load shaders
    int res;
    if ((res = loadShaders()) == -1) return res;
    
    // link shaders and start it up
    if((res = linkShaders()) == -1) return res;

    // vertex data loading -- posx, posy, posz
    int l;
    int m;
    Terrain terrain(SEED);
    //std::vector<GLfloat> new_vertices_3D(0);
    clock_t begin = clock();
#if VERSION == 1 || VERSION == 2
    std::vector<float*> chunks(0);
    std::cout << "Chunk Generation (GPU): ";
#else
    std::vector<float*> chunks(0);
    std::cout << "Chunk Generation (CPU): ";
#endif
    for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
        for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {

#if VERSION == 1
            auto chunk = terrain.generateChunkDataGpu(l, m);
#elif VERSION == 2
            auto chunk = terrain.generateChunkDataGpuOpt(l, m);
#else
            auto chunk = terrain.generateChunkData(l, m);
#endif
            chunks.push_back(chunk);
            /*
            int tracker = 0;
            for (int a = 0; a < Terrain::CHUNK_HEIGHT; a++) {
                for (int b = 0; b < Terrain::CHUNK_WIDTH; b++) {
                    for (int c = 0; c < Terrain::CHUNK_WIDTH; c++) {
                        printf("%d ", chunk[tracker++]);
                    }
                }
                printf("\n");
            }*/

        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //time_spent *= 1000.0; // seconds to milliseconds
    std::cout << time_spent << std::endl;

    size_t* slices = (size_t*)calloc(chunks.size() * (Terrain::CHUNK_HEIGHT)+1, sizeof(size_t));
    begin = clock();
    size_t num = 0;
    int curSlice = 0;
//#if VERSION == 2
  //  slicesKernel(slices, chunks, triTable, case_to_numpolys);
//#else
    for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
            //size_t curChunk = chunkSize * 2 * Terrain::NUM_CHUNKS_SIDE * (l + Terrain::NUM_CHUNKS_SIDE) + chunkSize * (m + Terrain::NUM_CHUNKS_SIDE);
            for (k = 0; k < Terrain::CHUNK_HEIGHT - 1; k++) {
                slices[curSlice++] = num;
                for (i = 0; i < Terrain::CHUNK_WIDTH - 1; i++) {
                    for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
                        int b = 0;

                        float* chunk = chunks[(m + Terrain::NUM_CHUNKS_SIDE) * 2 * Terrain::NUM_CHUNKS_SIDE + l + Terrain::NUM_CHUNKS_SIDE];
                        // flat array indexing for gpu returned chunks
                        int k_ = k * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH;
                        int k_1 = k_ + Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH;
                        int i_ = i * Terrain::CHUNK_WIDTH;
                        int i_1 = i_ + Terrain::CHUNK_WIDTH;
                        int j_ = j;
                        int j_1 = j_ + 1;
                        b += chunk[k_ + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v7
                        b <<= 1;
                        b += chunk[k_1 + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v6
                        b <<= 1;
                        b += chunk[k_1 + i_ + j_1] > Terrain::CAVE_INTENSITY; // v5
                        b <<= 1;
                        b += chunk[k_ + i_ + j_1] > Terrain::CAVE_INTENSITY; // v4
                        b <<= 1;
                        b += chunk[k_ + i_1 + j_] > Terrain::CAVE_INTENSITY; // v3
                        b <<= 1;
                        b += chunk[k_1 + i_1 + j_] > Terrain::CAVE_INTENSITY; // v2
                        b <<= 1;
                        b += chunk[k_1 + i_ + j_] > Terrain::CAVE_INTENSITY; // v1
                        b <<= 1;
                        b += chunk[k_ + i_ + j_] > Terrain::CAVE_INTENSITY; // v0


                        unsigned int numTriangles = case_to_numpolys[b];
                        num += numTriangles * 18;
                        
                    }
                }
                
            }
            slices[curSlice++] = num;
        }
    }
    slices[curSlice++] = num;
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //time_spent *= 1000.0; // seconds to milliseconds
    std::cout << "Marching Cubes Pt 1 (CPU): " << time_spent << std::endl;
//#endif
    int w = Terrain::CHUNK_WIDTH-1;
    int h = Terrain::CHUNK_HEIGHT-1;
    size_t chunkSize = 90 * w * w * h;
    size_t numChunks = 4 * Terrain::NUM_CHUNKS_SIDE * Terrain::NUM_CHUNKS_SIDE;
    size_t size = chunkSize * numChunks;
    GLfloat* vertices_3D = (GLfloat*)calloc(num, sizeof(GLfloat));

    if (!vertices_3D) {
        std::cerr << "calloc failed; valid sizes: " << std::endl;
        for (int i = 0; i < 16; i++) {
            numChunks = 4 * i * i;
            size = chunkSize * numChunks;
            std::cout << size*sizeof(GLfloat) << std::endl;
        }
        std::cout << SIZE_MAX << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t index = 0;
    begin = clock();
#if VERSION == 1 || VERSION == 2
    // need to cudaMalloc for slices, vertices, and chunks
    //unsigned int** new_triTable = (unsigned int**)calloc(256 * 16, sizeof(unsigned int));
    //unsigned int* new_c2np = (unsigned int*)calloc(256, sizeof(unsigned int));
    marchingCubesKernel(slices, vertices_3D, chunks, num, triTable, case_to_numpolys);
    /*for (int i = 0; i < num * sizeof(GLfloat); i++) {
        std::cout << stuff[i] << " " << std::endl;
    }*/

#else
    for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
            //size_t curChunk = chunkSize * 2 * Terrain::NUM_CHUNKS_SIDE * (l+Terrain::NUM_CHUNKS_SIDE) + chunkSize * (m+Terrain::NUM_CHUNKS_SIDE);
            for (k = 0; k < Terrain::CHUNK_HEIGHT-1; k++) {
                for (i = 0; i < Terrain::CHUNK_WIDTH - 1; i++) {
                    for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
                        //size_t curVoxel = 90 * (k * w * w + i * w + j);
                        int b = 0;

                        float* chunk = chunks[(m + Terrain::NUM_CHUNKS_SIDE) * 2 * Terrain::NUM_CHUNKS_SIDE + l + Terrain::NUM_CHUNKS_SIDE];
                        // flat array indexing for gpu returned chunks
                        int k_ = k*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
                        int k_1 = k_ + Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
                        int i_ = i*Terrain::CHUNK_WIDTH;
                        int i_1 = i_ + Terrain::CHUNK_WIDTH;
                        int j_ = j;
                        int j_1 = j_ + 1;
                        b += chunk[k_  + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v7
                        b <<= 1;
                        b += chunk[k_1 + i_1 + j_1] > Terrain::CAVE_INTENSITY; // v6
                        b <<= 1;
                        b += chunk[k_1 + i_  + j_1] > Terrain::CAVE_INTENSITY; // v5
                        b <<= 1;
                        b += chunk[k_  + i_  + j_1] > Terrain::CAVE_INTENSITY; // v4
                        b <<= 1;
                        b += chunk[k_  + i_1 + j_ ] > Terrain::CAVE_INTENSITY; // v3
                        b <<= 1;
                        b += chunk[k_1 + i_1 + j_ ] > Terrain::CAVE_INTENSITY; // v2
                        b <<= 1;
                        b += chunk[k_1 + i_  + j_ ] > Terrain::CAVE_INTENSITY; // v1
                        b <<= 1;
                        b += chunk[k_ + i_ + j_] > Terrain::CAVE_INTENSITY; // v0

                        unsigned int numTriangles = case_to_numpolys[b];
                        unsigned int triangles[16];
                        for (int iterate = 0; iterate < 16; iterate++) {
                            triangles[iterate] = triTable[b][iterate];
                        }
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
                        //std::cout << numVerts;
                        int d = 1;
                        glm::vec3 grad(0.0f, 0.0f, 0.0f);
                        //auto abc = terrain.noise3D.randomGradient(k, i, j);
                        //grad = -normalize(glm::vec3(abc.x, abc.y, abc.z));
                        //grad.x = thing(j + 1, k, i, l, m, terrain) - thing(j - 1, k, i, l, m, terrain);
                        //grad.y = thing(j, k + 1, i, l, m, terrain) - thing(j, k - 1, i, l, m, terrain);
                        //grad.z = thing(j, k, i + 1, l, m, terrain) - thing(j, k, i - 1, l, m, terrain);
                        //grad = -normalize(grad);
                        int k_d = k_ + d*Terrain::CHUNK_WIDTH*Terrain::CHUNK_WIDTH;
                        int i_d = i_ + d*Terrain::CHUNK_WIDTH;
                        int j_d = j_ + d;
                        grad.x = (float) chunk[k_  + i_  + j_d] - chunk[k_ + i_ + j_];
                        grad.y = (float) chunk[k_d + i_  + j_ ] - chunk[k_ + i_ + j_];
                        grad.z = (float) chunk[k_  + i_d + j_ ] - chunk[k_ + i_ + j_];
                        grad = -normalize(grad);

                        for (unsigned int iterate = 0; iterate < numTriangles; iterate++) {
                            int curTriangle = 18 * iterate;
                            for (int ij = 0; ij < 3; ij++) {
                                auto curEdge = triangles[iterate * 3 + ij];
                                auto px = edges[curEdge][1] + (Terrain::CHUNK_WIDTH-1) * m;
                                auto py = edges[curEdge][2];
                                auto pz = edges[curEdge][0] + (Terrain::CHUNK_WIDTH-1) * l;
                                //new_vertices_3D.push_back(px);
                                //new_vertices_3D.push_back(py);
                                //new_vertices_3D.push_back(pz);
                                //new_vertices_3D.push_back(grad.x);
                                //new_vertices_3D.push_back(grad.y);
                                //new_vertices_3D.push_back(grad.z);
                                //size_t index = curChunk + curVoxel + curTriangle + 6 * ij;
                                vertices_3D[index++] = px*2;
                                vertices_3D[index++] = py*2;
                                vertices_3D[index++] = pz*2;
                                vertices_3D[index++] = grad.x;
                                vertices_3D[index++] = grad.y;
                                vertices_3D[index++] = grad.z;
                            }
                        }
                    }
                }
            }
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Marching Cubes Pt 2 (CPU): " << time_spent << std::endl;
#endif

    // VBO for Land Points
    GLuint VBOP;
    glGenBuffers(1, &VBOP);
    GLuint VAOP;
    glGenVertexArrays(1, &VAOP);
    glBindVertexArray(VAOP);
    glBindBuffer(GL_ARRAY_BUFFER, VBOP);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(GLfloat), vertices_3D, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3*sizeof(GLfloat)));
    glEnableVertexAttribArray(1);


    glm::mat4 Projection = glm::perspective(glm::radians(90.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 200.0f);
    //glm::mat4 Projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 100.0f);

    //lookAt takes in camera position, target position, and up in the world space

    glm::mat4 View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 Model = glm::mat4(1.0f);
    glm::mat4 mvp = Projection * View * Model;

    GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");
    GLuint ModelMatrixID = glGetUniformLocation(shaderProgram, "model");
    GLuint ColorID = glGetUniformLocation(shaderProgram, "u_color");
    GLuint LightID = glGetUniformLocation(shaderProgram, "u_reverseLightDirection");


    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    glm::vec3 sky = glm::vec3(140.0f, 189.0f, 214.0f) / 255.0f;
    glm::vec3 dirt = glm::vec3(155.0f, 118.0f, 83.0f) / 255.0f;
    glm::vec3 grass = glm::vec3(86.0f, 125.0f, 70.0f) / 255.0f;
    auto thing = glm::normalize(glm::vec3(0.5, 1.0, 0.25));
    

    glClearColor(sky.x, sky.y, sky.z, 0.0f);
    glPointSize(10);

    glUseProgram(shaderProgram);

    glUniform3f(ColorID, grass.x, grass.y, grass.z);
    glUniform3f(LightID, thing.x, thing.y, thing.z);

    do {
        processInput(window);
        float currentFrame = (float) glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // startup
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &Model[0][0]);
        
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei) num);
        
        View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        mvp = Projection * View * Model;

        // double buffering setup
        glfwSwapBuffers(window);
        // tells OS to wait for events to occur etc
        glfwPollEvents();

    } while (!glfwWindowShouldClose(window));
    
    glDeleteBuffers(1, &VAOP);
    glDeleteBuffers(1, &VBOP);
    free(vertices_3D);
    free(slices);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
