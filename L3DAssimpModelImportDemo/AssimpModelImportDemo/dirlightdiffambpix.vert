#version 330

layout (std140) uniform Matrices {
  mat4 projMatrix;
  mat4 viewMatrix;
  mat4 modelMatrix;
};

in vec3 position;
in vec3 normal;
in vec2 texCoord;

out vec2 TexCoord;
out vec3 Normal;
out float Depth;

void main() {
  Normal = normalize(vec3(viewMatrix * modelMatrix * vec4(normal,0.0)));	
  TexCoord = vec2(texCoord);
  gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(position,1.0);
  float z_n = gl_Position.z/gl_Position.w; // z_n in [-1, 1]
  float z_b = 0.5*z_n + 0.5; // z_b in [0, 1]
  Depth = z_b;
}
