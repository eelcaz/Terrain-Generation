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

// location = 0 bc of attrib pointer



int main(int argc, char** argv) {
    int i = 0;
    int j = 0;
    int k = 0;

    // // generate NUM_CHUNKS heightmaps
    // auto heights = new int **[NUM_CHUNKS];
    // for (int i = 0; i < NUM_CHUNKS; ++i) {
    //     heights[i] = Terrain::generateChunkHeightMap(i-NUM_CHUNKS/2, 0);
    // }

    // call kernel
    // auto chunk = chunkHeightMapKernel(0, 0);
    // for (int z = 0; z < Terrain::CHUNK_WIDTH; ++z) {
    //     for (int x = 0; x < Terrain::CHUNK_WIDTH; ++x) {
    //         std::cout << chunk[z*Terrain::CHUNK_WIDTH + x] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // delete[] chunk;
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

    // vertex data loading -- posx, posy, posz, r, g, b
    std::vector<GLfloat> vertices_3D(0);
    std::vector<GLfloat> vertices_3D_caves(0);
    int l;
    int m;
    /*
    // k = y, i = z, j = x; pushed as (x, y, z)
    for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
            auto D_chunks = Terrain::generateChunkData(l, m);
            for (k = 0; k < Terrain::CHUNK_HEIGHT; k++) {
                for (i = 0; i < Terrain::CHUNK_WIDTH; i++) {
                    for (j = 0; j < Terrain::CHUNK_WIDTH; j++) {
                        if (D_chunks[k][i][j] == 1) {
                            vertices_3D.push_back((j + Terrain::CHUNK_WIDTH * m));
                            vertices_3D.push_back(k/2.0);
                            vertices_3D.push_back((i + Terrain::CHUNK_WIDTH * l));
                        }
                        else {
                            vertices_3D_caves.push_back((j + Terrain::CHUNK_WIDTH * m));
                            vertices_3D_caves.push_back(k/2.0);
                            vertices_3D_caves.push_back((i + Terrain::CHUNK_WIDTH * l));
                        }
                    }
                }
            }
        }
    }
    */
    
    std::vector<GLfloat> new_vertices_3D(0);
    std::vector<glm::vec3> normals(0);
    for (m = -Terrain::NUM_CHUNKS_SIDE; m < Terrain::NUM_CHUNKS_SIDE; m++) {
        for (l = -Terrain::NUM_CHUNKS_SIDE; l < Terrain::NUM_CHUNKS_SIDE; l++) {
            auto chunk = Terrain::generateChunkData(l, m);
            for (k = 0; k < Terrain::CHUNK_HEIGHT - 1; k++) {
                for (i = 0; i < Terrain::CHUNK_WIDTH - 1; i++) {
                    for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
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
                        float edges[12][3] = {
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
                        for (int iterate = 0; iterate < numTriangles; iterate++) {
                            std::vector<glm::vec3> points(0);
                            for (int ij = 0; ij < 3; ij++) {
                                auto curEdge = triangles[iterate * 3 + ij];
                                auto px = edges[curEdge][1] + Terrain::CHUNK_WIDTH * m;
                                auto py = edges[curEdge][2];
                                auto pz = edges[curEdge][0] + Terrain::CHUNK_WIDTH * l;
                                new_vertices_3D.push_back(px);
                                new_vertices_3D.push_back(py);
                                new_vertices_3D.push_back(pz);
                                if (ij < 2) {
                                    points.push_back(glm::vec3(px, py, pz));
                                }
                            }
                            normals.push_back(glm::cross(points[0], points[1]));
                        }
                    }
                }
            }
        }
    }

    /*
    std::vector<GLfloat> vertices(Terrain::CHUNK_WIDTH* Terrain::CHUNK_WIDTH* NUM_CHUNKS * 3);
    std::vector<GLuint> elements((Terrain::CHUNK_WIDTH)*(Terrain::CHUNK_WIDTH-1)*NUM_CHUNKS*3*2);

    for (k = 0; k < NUM_CHUNKS; k++) {
        for (i = 0; i < Terrain::CHUNK_WIDTH; i++) {
            for (j = 0; j < Terrain::CHUNK_WIDTH; j++) {
                unsigned int cur = k * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH * 3 + i * Terrain::CHUNK_WIDTH * 3 + j * 3;
                vertices[cur + 0] = i - Terrain::CHUNK_WIDTH / 2.0f + k * Terrain::CHUNK_WIDTH;
                vertices[cur + 2] = -j;
                vertices[cur + 1] = heights[k][i*Terrain::CHUNK_WIDTH + j];
            }
        }
    }

    l = 0;
    for (k = 0; k < NUM_CHUNKS; k++) {
        for (j = 0; j < Terrain::CHUNK_WIDTH - 1; j++) {
            for (i = 0; i < Terrain::CHUNK_WIDTH-1; i++) {
                int vertIndex = k * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH + i * Terrain::CHUNK_WIDTH + j;

                elements[l++] = vertIndex;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH+1;
                elements[l++] = vertIndex;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH+1;
                elements[l++] = vertIndex + 1;
            }
            if (k < NUM_CHUNKS - 1) {
                int vertIndex = k * Terrain::CHUNK_WIDTH * Terrain::CHUNK_WIDTH + i * Terrain::CHUNK_WIDTH + j;

                elements[l++] = vertIndex;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH + 1;
                elements[l++] = vertIndex;
                elements[l++] = vertIndex + Terrain::CHUNK_WIDTH + 1;
                elements[l++] = vertIndex + 1;
            }
        }
    }

    GLfloat second_verts[] = {
        -50.0f, 0.0f, -50.0f,
        50.0f, 0.0f, -50.0f,
        50.0f, 0.0f, 50.0f,
        -50.0f, 0.0f, 50.0f
    };
    GLuint second_elements[] = {
        0, 1, 2,
        0, 3, 2
    };
    */

    // VBO for Land Points
    GLuint VBOP;
    glGenBuffers(1, &VBOP);
    GLuint VAOP;
    glGenVertexArrays(1, &VAOP);
    glBindVertexArray(VAOP);
    glBindBuffer(GL_ARRAY_BUFFER, VBOP);
    glBufferData(GL_ARRAY_BUFFER, new_vertices_3D.size() * sizeof(GLfloat), &new_vertices_3D[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    /*
    // VBO for Cave Points
    GLuint VBOP2;
    glGenBuffers(1, &VBOP2);
    GLuint VAOP2;
    glGenVertexArrays(1, &VAOP2);
    glBindVertexArray(VAOP2);
    glBindBuffer(GL_ARRAY_BUFFER, VBOP2);
    glBufferData(GL_ARRAY_BUFFER, vertices_3D_caves.size() * sizeof(GLfloat), &vertices_3D_caves[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    */

    /*
    // VBO for the "chunk strip"
    GLuint VBO;
    glGenBuffers(1, &VBO);

    GLuint EBO;
    glGenBuffers(1, &EBO);

    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(GLuint), &elements[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // VBO for "ground"
    GLuint VBO2;
    glGenBuffers(1, &VBO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(second_verts), second_verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // unbinding is just binding to 0 ; vao handles this
    //glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint EBO2;
    glGenBuffers(1, &EBO2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(second_elements), second_elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    GLuint VAO2;
    glGenVertexArrays(1, &VAO2);
    glBindVertexArray(VAO2);

    // getting ready to send to shader
    // first param is index, second param is number of indices
    // third is type, fourth is for normalization
    // fifth is stride of bytes to move between each vertex
    // sixth is pointer to the first element ; in some cases u can ignore void, dunno why
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBindVertexArray(0);
    */

    glUseProgram(shaderProgram);
    glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), 1.0f, 0.5f, 0.5f);
    glUseProgram(0);

    glm::mat4 Projection = glm::perspective(glm::radians(90.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
    //glm::mat4 Projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 100.0f);

    //lookAt takes in camera position, target position, and up in the world space

    glm::mat4 View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
    glm::mat4 mvp = Projection * View * Model;

    GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");



    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    glm::vec3 sky = glm::vec3(140.0f, 189.0f, 214.0f) / 255.0f;
    glm::vec3 dirt = glm::vec3(155.0f, 118.0f, 83.0f) / 255.0f;
    glm::vec3 grass = glm::vec3(86.0f, 125.0f, 70.0f) / 255.0f;
    glClearColor(sky.x, sky.y, sky.z, 0.0f);
    glPointSize(10);
    do {
        processInput(window);
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        //float red = sin(glfwGetTime());
        //float green = sin(glfwGetTime()/2);
        //float blue = sin(glfwGetTime()*0.9);
        //glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), red, green, blue);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // startup
        glUseProgram(shaderProgram);
        
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
        // draw the triangle, index 0, 3 vertices
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        //glDrawArrays(GL_TRIANGLES, 0, 3); // could to GL_POINTS, GL_LINE_STRIP
        /*
        glBindVertexArray(VAO);
        glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), grass.x, grass.y, grass.z);
        glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
        //glDrawArrays(GL_TRIANGLE_STRIP, 0, 16*16*2);
        glBindVertexArray(0);
        glBindVertexArray(VAO2);
        //glDrawElements(GL_TRIANGLES, sizeof(second_elements) / sizeof(elements[0]), GL_UNSIGNED_INT, 0);
        glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), dirt.x, dirt.y, dirt.z);
        glDrawElements(GL_TRIANGLES, sizeof(second_elements) / sizeof(second_elements[0]), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        */
        glBindVertexArray(VAOP);
        glPointSize(10);
        glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), grass.x, grass.y, grass.z);
        glDrawArrays(GL_TRIANGLES, 0, new_vertices_3D.size());
        glBindVertexArray(0);

        /*
        glBindVertexArray(VAOP2);
        glPointSize(50);
        glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), dirt.x, dirt.y, dirt.z);
        glDrawArrays(GL_POINTS, 0, vertices_3D_caves.size());
        glBindVertexArray(0);
        */

        View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        mvp = Projection * View * Model;
        //glBindVertexArray(0);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // double buffering setup
        glfwSwapBuffers(window);
        // tells OS to wait for events to occur etc
        glfwPollEvents();

    } while (!glfwWindowShouldClose(window));
    /*
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteShader(planeVS);
    glDeleteShader(planeFS);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);
    */
    glDeleteBuffers(1, &VBOP);
    //glDeleteBuffers(1, &VBOP2);

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
