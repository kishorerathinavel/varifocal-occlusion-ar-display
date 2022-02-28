#version 330

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D rgb_img;

void main() {
  vec2 corrected_TexCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);
  FragColor = vec4(texture(rgb_img, corrected_TexCoord));
}
