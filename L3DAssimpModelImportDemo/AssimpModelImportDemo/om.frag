#version 330

out vec4 FragColor;

in vec2 TexCoord;
in vec4 gl_FragCoord;

uniform sampler2D rgb_img;
uniform sampler2D blur_map;
uniform sampler2D depth_map;
uniform float zNear;
uniform float zFar;
uniform float linear_near;
uniform float linear_far;
uniform float focal_depth;

float convertDisaprityToBlur(vec4 v4_disparity) {
  float factor = 1.0;
  //return factor*v4_disparity[0]*v4_disparity[0];
  return factor*v4_disparity[0];
}

// Things to play around with:
// convertDisparityToBlur... to square or not to square?
// contributeFromAdjFrag... include distance metric?
// FragColor... (1.0/2.0) factor
void main() {
  int minX = 0, maxX = 1024, minY = 0, maxY = 768;
  float factor = 1.0;

  vec4 v4_centerFrag_rgb = texture(rgb_img, TexCoord);
  vec4 v4_centerFrag_blur = vec4(convertDisaprityToBlur(texture(blur_map, TexCoord)));
  vec4 v4_centerFrag_depth = texture(depth_map, TexCoord);

  float centerFrag_blur = v4_centerFrag_blur[0];
  float centerFrag_depth = v4_centerFrag_depth[0];

  vec4 contributionFromAdjFrag = vec4(0.0);
  for (int iterX = -10; iterX < 11; iterX = iterX + 1) {
    for (int iterY = -10; iterY < 11; iterY = iterY + 1) {
      if(iterX == 0 && iterY == 0) {
	continue;
      } else {
	vec2 adjTexCoord = vec2(TexCoord.x + 10.0*float(iterX)/maxX, TexCoord.y + 10.0*float(iterY)/maxY);
	vec4 v4_adjFrag_depth = texture(depth_map, adjTexCoord);
	float adjFrag_depth = v4_adjFrag_depth[0];
	if((adjFrag_depth == 1.0) || (adjFrag_depth - centerFrag_depth > 0.01)) {
	  continue;
	}
	else {
	  vec4 v4_adjFrag_blur = vec4(convertDisaprityToBlur(texture(blur_map, adjTexCoord)));
	  contributionFromAdjFrag = contributionFromAdjFrag + v4_adjFrag_blur;
	} 
      }
    }
  }
  FragColor = (vec4(1.0 - centerFrag_blur) + contributionFromAdjFrag/(20.0*20.0));
  // if(TexCoord.x < 0.5) {
  //   FragColor = vec4(1.0 - centerFrag_blur);
  // } else {
  //   FragColor = contributionFromAdjFrag/(40.0*40.0);
  // }
}
