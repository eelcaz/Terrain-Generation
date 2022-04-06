#include <iostream>
#include <windows.h>
#include <GL/glew.h> // include this one first
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#define WIDTH 1024
#define HEIGHT 768

// location = 0 bc of attrib pointer
const GLchar* vertexShaderSource = R"glsl(
#version 440 core

layout(location = 0) in vec3 position;

uniform mat4 MVP;

out vec4 out_color;

void main()
{
gl_Position = MVP * vec4(position, 1.0);
out_color = vec4(gl_Position.xyz, 1.0);
}
)glsl";

const GLchar* fragmentShaderSource = R"glsl(
#version 440 core

in vec4 out_color;
out vec4 color;

uniform vec3 u_color;

void main()
{
color = out_color;
//color = vec4(u_color.xyz, 1.0);
}
)glsl";

int main(int argc, char** argv) {

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

	// coordinates are between -1 and 1, center is 0 0
	// this defines the new center as the center
	//glViewport(0, 0, WIDTH, HEIGHT);

	std::cout << glGetString(GL_VERSION) << std::endl;

	

	// load shaders
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// which shader, number of shaders, source glsl shader code, length done internally
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// always keep an eye on the compilation process
	glCompileShader(vertexShader);

	// keeping an eye on it!
	GLint success;
	GLchar info_log[512];
	// get shader inventory ; which shader, which inventory you want (also maybe link),
	// and then where ur writing success to
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(vertexShader, 512, NULL, info_log);
		std::cout << "ERORR::SHADER::VERTEX::COMPILATION_FAILED\n" << info_log << std::endl;
		return -1;
	}


	// fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertexShader, 512, NULL, info_log);
		std::cout << "ERORR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << info_log << std::endl;
		return -1;
	}


	// link shaders and start it up
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, info_log);
		std::cout << "ERROR::PROGRAM::LINKER_FAILED\n" << info_log << std::endl;
		return -1;
	}


	// vertex data loading -- posx, posy, posz, r, g, b
	GLfloat vertices[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f
	};

	GLuint elements[] = {
		0, 1, 2, // first
		0, 3, 2
	};

	GLuint VBO;
	// first arg is number of buffers, second arg is buffer
	glGenBuffers(1, &VBO);
	// everything below will be on this buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// static draw: the data won't change very much
	// dynamic draw: data is likely to change
	// stream draw: data iwll change every frame
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// getting ready to send to shader
	// first param is index, second param is number of indices
	// third is type, fourth is for normalization
	// fifth is stride of bytes to move between each vertex
	// sixth is pointer to the first element ; in some cases u can ignore void, dunno why
	// 6 * sizeof(GLfloat) for sending position AND color, 3 for position
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);

	// need attrib pointer to actually tell the gpu how to process the data
	//glEnableVertexAttribArray(1);
	//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));

	// unbinding is just binding to 0 ; vao handles this
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLuint EBO;
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); vao handles
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GLuint VAO; // like a container for the vertex buffer and element buffer
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBindVertexArray(0);

	glUseProgram(shaderProgram);
	glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), 1.0f, 0.5f, 0.5f);
	glUseProgram(0);

	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);

	//glm::mat4 Projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 100.0f);

	glm::mat4 View = glm::lookAt(
		glm::vec3(4, 3, 3),
		glm::vec3(0, 0, 0),
		glm::vec3(0, 1, 0)
	);
	
	glm::mat4 Model = glm::mat4(1.0f);
	glm::mat4 mvp = Projection * View * Model;

	GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");
	

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glClearColor(0.0f, 0.4f, 0.4f, 0.0f);
	do {
		

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
		glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, sizeof(elements)/sizeof(elements[0]), GL_UNSIGNED_INT, 0);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		//glBindVertexArray(0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// double buffering setup
		glfwSwapBuffers(window);
		// tells OS to wait for events to occur etc
		glfwPollEvents();

	} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		!glfwWindowShouldClose(window));

	glDeleteBuffers(1, &VBO);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteBuffers(1, &EBO);

	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}