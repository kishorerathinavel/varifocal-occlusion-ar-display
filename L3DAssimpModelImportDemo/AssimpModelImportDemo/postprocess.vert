#version 330

in vec3 position;
in vec3 color;
in vec2 texCoord;

out vec3 vertexPos;
out vec2 TexCoord;


void main() {
	gl_Position = vec4(position, 1.0);
	
}