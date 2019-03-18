#version 330

in vec3 position;
in vec2 texCoord;

out vec2 TexCoord;

void main() {
	gl_Position = vec4(position, 1.0) - 0.5;
	TexCoord = vec2(texCoord);
}