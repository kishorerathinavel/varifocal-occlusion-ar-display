#version 330

out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D depth_map;

void main() {
  vec4 v4_centerFrag_depth = texture(depth_map, TexCoord);
  float centerFrag_depth = v4_centerFrag_depth[0];

  if(centerFrag_depth == 1) {
    FragColor = vec4(1.0);
  } else {
    FragColor = vec4(0.0);
  }
}
