#include <GL/glew.h>

GLuint planeVS;
GLuint planeFS;
GLuint shaderProgram;
GLint success;
GLchar info_log[512];

const GLchar* planeVSSource = R"glsl(
#version 440 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 aNormal;


uniform mat4 MVP;
uniform mat4 model;

varying vec3 Normal;

void main()
{
    gl_Position = MVP * vec4(position, 1.0);
    Normal = aNormal;
}
)glsl";

const GLchar* planeFSSource = R"glsl(
#version 440 core

varying vec3 Normal;

uniform vec3 u_color;
uniform vec3 u_reverseLightDirection;

void main()
{
    vec3 normal = normalize(Normal);
    float light = dot(normal, normalize(u_reverseLightDirection));
    gl_FragColor = vec4(u_color.xyz, 1.0);
    gl_FragColor.rgb *= light;
}
)glsl";

int loadShaders() {
    planeVS = glCreateShader(GL_VERTEX_SHADER);
    // which shader, number of shaders, source glsl shader code, length done internally
    glShaderSource(planeVS, 1, &planeVSSource, NULL);
    // always keep an eye on the compilation process
    glCompileShader(planeVS);

    // get shader inventory ; which shader, which inventory you want (also maybe link),
    // and then where ur writing success to
    glGetShaderiv(planeVS, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(planeVS, 512, NULL, info_log);
        std::cout << "ERORR::SHADER::VERTEX::COMPILATION_FAILED\n" << info_log << std::endl;
        return -1;
    }


    // fragment shader
    planeFS = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(planeFS, 1, &planeFSSource, NULL);
    glCompileShader(planeFS);

    glGetShaderiv(planeFS, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(planeFS, 512, NULL, info_log);
        std::cout << "ERORR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << info_log << std::endl;
        return -1;
    }
    return 0;
}

int linkShaders() {
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, planeVS);
    glAttachShader(shaderProgram, planeFS);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, info_log);
        std::cout << "ERROR::PROGRAM::LINKER_FAILED\n" << info_log << std::endl;
        return -1;
    }
}