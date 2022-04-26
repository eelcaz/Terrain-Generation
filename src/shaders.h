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
uniform mat4 view;

out vec3 Normal;
out vec3 FragPos;

void main()
{
gl_Position = MVP * vec4(position, 1.0);
FragPos = vec3(model * vec4(position, 1.0));
Normal = aNormal;
}
)glsl";

const GLchar* planeFSSource = R"glsl(
#version 440 core

out vec4 color;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 lightDir;
uniform vec3 u_color;

void main()
{
vec3 lightColor = vec3(1.0, 1.0, 1.0);

// ambient
float ambientStrength = 0.1;
vec3 ambient = ambientStrength * lightColor;

// diffuse
vec3 norm = normalize(Normal);
vec3 lightDir = normalize(lightPos - FragPos);
float diff = max(dot(norm, lightDir), 0.0);
vec3 diffuse = diff*lightColor;

// specular
float specularStrength = 0.5;
vec3 viewDir = normalize(lightPos - FragPos);
vec3 reflectDir = reflect(-lightDir, norm);
float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
vec3 specular = specularStrength * spec * lightColor;

vec3 result = (ambient + diffuse + specular) * u_color;
color = vec4(result, 1.0);
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