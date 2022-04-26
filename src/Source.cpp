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


#define NUM_CHUNKS 8
#define SEED 2022
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
int main(int argc, char** argv) {
    int i = 0;
    int j = 0;
    int k = 0;

    auto heights = new int *[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        heights[i] = Terrain::generateChunkHeightMapGpu(i-NUM_CHUNKS/2, 0);
    }


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
    std::vector<GLfloat> new_vertices_3D(0);
    std::vector<int***> chunks(0);
    for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
            chunks.push_back(terrain.generateChunkData(l, m));
        }
    }
    unsigned long int totalChunkCount = 4 * Terrain::NUM_CHUNKS_SIDE * Terrain::NUM_CHUNKS_SIDE;
    unsigned long int pointCount = (Terrain::CHUNK_HEIGHT - 1) * (Terrain::CHUNK_WIDTH - 1) * (Terrain::CHUNK_WIDTH - 1);
    unsigned long int maxTriangleCount = 5;
    unsigned long int pointsPerTriangle = 6 * 3; // vx, vy, vz, nx, ny, nz for 3 vertices
    unsigned long int size = totalChunkCount * pointCount * maxTriangleCount * pointsPerTriangle;
    unsigned long int sizeOfChunk = pointCount * maxTriangleCount * pointsPerTriangle;
    unsigned long int sizeOfPlane = (Terrain::CHUNK_WIDTH - 1) * (Terrain::CHUNK_WIDTH - 1);
    GLfloat* vertices_3D = (GLfloat *) calloc(size, sizeof(GLfloat));
    for (m = 0; m < 2 * Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = 0; l < 2 * Terrain::NUM_CHUNKS_SIDE; l++) {
            auto chunk = terrain.generateChunkData(l - Terrain::NUM_CHUNKS_SIDE, m - Terrain::NUM_CHUNKS_SIDE);
            unsigned int curChunk = m * sizeOfChunk * sizeOfChunk + l * sizeOfChunk;
            for (k = 0; k < Terrain::CHUNK_HEIGHT - 1; k++) {
                for (i = 0; i < Terrain::CHUNK_WIDTH - 1; i++) {
                    for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
                        unsigned int curVoxel = k * sizeOfPlane + i*(Terrain::CHUNK_WIDTH-1)+j;
                        unsigned int curIndex = curChunk + curVoxel;

                        int b = 0;
                        b += chunk[k][i + 1][j + 1];    // v7
                        b <<= 1;
                        b += chunk[k + 1][i + 1][j + 1];    // v6
                        b <<= 1;
                        b += chunk[k + 1][i][j + 1];    // v5
                        b <<= 1;
                        b += chunk[k][i][j + 1];    // v4
                        b <<= 1;
                        b += chunk[k][i + 1][j];    // v3
                        b <<= 1;
                        b += chunk[k + 1][i + 1][j];    // v2
                        b <<= 1;
                        b += chunk[k + 1][i][j];    // v1
                        b <<= 1;
                        b += chunk[k][i][j];    // v0

                        unsigned int e = edgeTable[b];
                        unsigned int numTriangles = case_to_numpolys[b];
                        unsigned int triangles[16];
                        for (int iterate = 0; iterate < 16; iterate++) {
                            triangles[iterate] = triTable[b][iterate];
                        }
                        GLfloat edges[12][3] = {
                            {i + 0.0, j + 0.0, k + 0.5},    // e0
                            {i + 0.5, j + 0.0, k + 1.0},    // e1
                            {i + 1.0, j + 0.0, k + 0.5},    // e2
                            {i + 0.5, j + 0.0, k + 0.0},    // e3
                            {i + 0.0, j + 1.0, k + 0.5},    // e4
                            {i + 0.5, j + 1.0, k + 1.0},    // e5
                            {i + 1.0, j + 1.0, k + 0.5},    // e6
                            {i + 0.5, j + 1.0, k + 0.0},    // e7
                            {i + 0.0, j + 0.5, k + 0.0},    // e8
                            {i + 0.0, j + 0.5, k + 1.0},    // e9
                            {i + 1.0, j + 0.5, k + 1.0},    // e10
                            {i + 1.0, j + 0.5, k + 0.0}     // e11
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
                        grad.x = chunk[k][i][j + d] - chunk[k][i][j];
                        grad.y = chunk[k + d][i][j] - chunk[k][i][j];
                        grad.z = chunk[k][i + d][j] - chunk[k][i][j];
                        grad = -normalize(grad);

                        for (int iterate = 0; iterate < numTriangles; iterate++) {
                            
                            for (int ij = 0; ij < 3; ij++) {
                                auto curEdge = triangles[iterate * 3 + ij];
                                GLfloat px = edges[curEdge][1] + Terrain::CHUNK_WIDTH * (m-Terrain::NUM_CHUNKS_SIDE);
                                GLfloat py = edges[curEdge][2];
                                GLfloat pz = edges[curEdge][0] + Terrain::CHUNK_WIDTH * (l-Terrain::NUM_CHUNKS_SIDE);
                                vertices_3D[curIndex * 6] = px;
                                vertices_3D[curIndex * 6 + 1] = py;
                                vertices_3D[curIndex * 6 + 2] = pz;
                                vertices_3D[curIndex * 6 + 3] = grad.x;
                                vertices_3D[curIndex * 6 + 4] = grad.y;
                                vertices_3D[curIndex * 6 + 5] = grad.z;
                                //new_vertices_3D.push_back(px);
                                //new_vertices_3D.push_back(py);
                                //new_vertices_3D.push_back(pz);
                                //new_vertices_3D.push_back(grad.x);
                                //new_vertices_3D.push_back(grad.y);
                                //new_vertices_3D.push_back(grad.z);
                            }
                            curIndex++;
                        }
                    }
                }
            }
        }
    }
    if (new_vertices_3D.size() == 0) {
        std::cerr << "UH OH NO VERTICES" << std::endl;
        return -1;
    }
    // VBO for Land Points
    GLuint VBOP;
    glGenBuffers(1, &VBOP);
    GLuint VAOP;
    glGenVertexArrays(1, &VAOP);
    glBindVertexArray(VAOP);
    glBindBuffer(GL_ARRAY_BUFFER, VBOP);
    glBufferData(GL_ARRAY_BUFFER, new_vertices_3D.size() * sizeof(GLfloat), &new_vertices_3D[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    

    glm::mat4 Projection = glm::perspective(glm::radians(90.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
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
    auto thing = glm::normalize(glm::vec3(0.0, 1.0, 0.0));
    

    glClearColor(sky.x, sky.y, sky.z, 0.0f);
    glPointSize(10);
    
    do {
        processInput(window);
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // startup
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &Model[0][0]);
        glUniform3f(ColorID, grass.x, grass.y, grass.z);
        glUniform3f(LightID, thing.x, thing.y, thing.z);
        
        glBindVertexArray(VAOP);
        glPointSize(10);
        glDrawArrays(GL_TRIANGLES, 0, new_vertices_3D.size()/2);
        glBindVertexArray(0);


        View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        mvp = Projection * View * Model;

        // double buffering setup
        glfwSwapBuffers(window);
        // tells OS to wait for events to occur etc
        glfwPollEvents();

    } while (!glfwWindowShouldClose(window));
    
    glDeleteBuffers(1, &VAOP);
    glDeleteBuffers(1, &VBOP);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
