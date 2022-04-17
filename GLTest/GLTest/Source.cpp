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

#define WIDTH 1920
#define HEIGHT 1080

#define NUM_CHUNKS 2
#define CHUNK_SIZE 256

// location = 0 bc of attrib pointer
const GLchar* vertexShaderSource = R"glsl(
#version 440 core

layout(location = 0) in vec3 position;

uniform mat4 MVP;

out vec4 out_color;

void main()
{
gl_Position = MVP * vec4(position, 1.0);
out_color = vec4(gl_Position.y, gl_Position.y, gl_Position.y, 1.0);
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
color = vec4(u_color.xyz, 1.0);
}
)glsl";

glm::vec3 cameraPos = glm::vec3(0.0f, 70.0f, 40.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

float pitch = 0.0f;
float yaw = -90.0f;

bool firstMouse = true;
float lastX = WIDTH / 2.0f, lastY = HEIGHT / 2.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main(int argc, char** argv) {

	std::string line;
	std::ifstream myfile("../perlin_fbm_good_2chunks.txt");
	float heights[NUM_CHUNKS][CHUNK_SIZE][CHUNK_SIZE];
	int i = 0;
	int j = 0;
	int k = 0;
	int trackerS = 0;
	int trackerE = 0;
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			//std::cout << line << '\n';
			//std::cout << line << std::endl;
			for (j = 0; j < CHUNK_SIZE; j++) {
				while (trackerS < line.length() && line[trackerS] == ' ') {
					trackerS++;
				}
				trackerE = trackerS;
				while (trackerE < line.length() && line[trackerE] != ' ') {
					trackerE++;
				}
				heights[k][i][j] = std::stof(line.substr(trackerS, trackerE - trackerS));
				//std::cout << heights[k][i][j] << " " << trackerS << " " << trackerE << std::endl;
				trackerS = trackerE;
			}
			trackerS = 0;
			trackerE = 0;

			i++;
			if (i == CHUNK_SIZE) {
				i = 0;
				k++;
			}
		}
		myfile.close();
	}
	else std::cout << "Unable to open file";

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
	std::vector<GLfloat> vertices(CHUNK_SIZE* CHUNK_SIZE* NUM_CHUNKS * 3);
	//GLfloat vertices[CHUNK_SIZE * CHUNK_SIZE * NUM_CHUNKS * 3];
	std::vector<GLuint> elements((CHUNK_SIZE-1)*(CHUNK_SIZE-1)*NUM_CHUNKS*3*2);
	for (k = 0; k < NUM_CHUNKS; k++) {
		for (i = 0; i < CHUNK_SIZE; i++) {
			for (j = 0; j < CHUNK_SIZE; j++) {
				int cur = k * CHUNK_SIZE * CHUNK_SIZE * 3 + i * CHUNK_SIZE * 3 + j * 3;
				//vertices[cur + 0] = i-CHUNK_SIZE/2.0f;
				//vertices[cur + 2] = -j - k * CHUNK_SIZE;
				//vertices[cur + 1] = heights[k][i][j];
				vertices[cur + 0] = i - CHUNK_SIZE / 2.0f + k * CHUNK_SIZE;
				vertices[cur + 2] = -j;
				vertices[cur + 1] = heights[k][i][j];
			}
		}
	}
	int l = 0;
	for (k = 0; k < NUM_CHUNKS; k++) {
		for (i = 0; i < CHUNK_SIZE-1; i++) {
			for (j = 0; j < CHUNK_SIZE-1; j++) {
				int index = k * (CHUNK_SIZE-1) * (CHUNK_SIZE-1) + i * (CHUNK_SIZE-1) + j;
				int vertIndex = k * CHUNK_SIZE * CHUNK_SIZE + i * CHUNK_SIZE + j;
				
				elements[l++] = vertIndex;
				elements[l++] = vertIndex + CHUNK_SIZE;
				elements[l++] = vertIndex + CHUNK_SIZE+1;
				elements[l++] = vertIndex;
				elements[l++] = vertIndex + 1;
				elements[l++] = vertIndex + CHUNK_SIZE+1;
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

	GLuint VBO2;
	glGenBuffers(1, &VBO2);
	glBindBuffer(GL_ARRAY_BUFFER, VBO2);
	glBufferData(GL_ARRAY_BUFFER, sizeof(second_verts), second_verts, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
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

	GLuint EBO2;
	glGenBuffers(1, &EBO2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(second_elements), second_elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GLuint VAO2;
	glGenVertexArrays(1, &VAO2);
	glBindVertexArray(VAO2);

	glBindBuffer(GL_ARRAY_BUFFER, VBO2);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
	glBindVertexArray(0);

	glUseProgram(shaderProgram);
	glUniform3f(glGetUniformLocation(shaderProgram, "u_color"), 1.0f, 0.5f, 0.5f);
	//glUniform3f(glGetUniformLocation(shaderProgram, "in_color"), 1.0f, 0.5f, 0.5f);
	glUseProgram(0);

	glm::mat4 Projection = glm::perspective(glm::radians(90.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);

	//glm::mat4 Projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.0f, 100.0f);
	
	//lookAt takes in camera position, target position, and up in the world space

	glm::mat4 View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
	glm::mat4 Model = glm::mat4(1.0f);
	Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
	glm::mat4 mvp = Projection * View * Model;

	GLuint MatrixID = glGetUniformLocation(shaderProgram, "MVP");
	
	

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	glfwSetCursorPosCallback(window, mouse_callback);
	void processInput(GLFWwindow* window);
	
	glm::vec3 sky = glm::vec3(140.0f, 189.0f, 214.0f) / 255.0f;
	glm::vec3 dirt = glm::vec3(155.0f, 118.0f, 83.0f) / 255.0f;
	glm::vec3 grass = glm::vec3(86.0f, 125.0f, 70.0f) / 255.0f;
	glClearColor(sky.x, sky.y, sky.z, 0.0f);
	//glPointSize(5);
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
		
		View = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		mvp = Projection * View * Model;
		//glBindVertexArray(0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// double buffering setup
		glfwSwapBuffers(window);
		// tells OS to wait for events to occur etc
		glfwPollEvents();
		
	} while (!glfwWindowShouldClose(window));

	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &VBO2);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteBuffers(1, &EBO);
	glDeleteBuffers(1, &EBO2);

	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}

void processInput(GLFWwindow* window) {
	const float cameraSpeed = 25.0f * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraUp;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraUp;
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 direction;
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(direction);
}
