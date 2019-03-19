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
  return sqrt(factor*v4_disparity[0]);
}

// Things to play around with:
// convertDisparityToBlur... to square or not to square?
// contributeFromAdjFrag... include distance metric?
// FragColor... (1.0/2.0) factor
void main() {
  int minX = 0, maxX = 1024, minY = 0, maxY = 768;
  float factor = 1.0;
  vec2 corrected_TexCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);

  vec4 v4_centerFrag_rgb = texture(rgb_img, corrected_TexCoord);
  vec4 v4_centerFrag_blur = vec4(convertDisaprityToBlur(texture(blur_map, corrected_TexCoord)));
  vec4 v4_centerFrag_depth = texture(depth_map, corrected_TexCoord);

  float centerFrag_blur = v4_centerFrag_blur[0];
  float centerFrag_depth = v4_centerFrag_depth[0];

  vec4 contributionFromAdjFrag = vec4(0.0);
  float weight = 0.0;
  for (int iterX = -10; iterX < 11; iterX = iterX + 1) {
    for (int iterY = -10; iterY < 11; iterY = iterY + 1) {
      if((iterX == 0 && iterY == 0) || (iterX*iterX + iterY*iterY > 10*10)) {
	continue;
      } else {
	float iterXf = float(iterX);
	float iterYf = float(iterY);
	vec2 adjcorrected_TexCoord = vec2(corrected_TexCoord.x + 2.0*iterXf/maxX, corrected_TexCoord.y + 2.0*iterYf/maxY);
	vec4 v4_adjFrag_depth = texture(depth_map, adjcorrected_TexCoord);
	float adjFrag_depth = v4_adjFrag_depth[0];
	if((adjFrag_depth == 1.0) || adjFrag_depth - centerFrag_depth > 0.01) {
	  continue;
	}
	else {
	  vec4 v4_adjFrag_rgb = texture(rgb_img, adjcorrected_TexCoord);
	  vec4 v4_adjFrag_blur = vec4(convertDisaprityToBlur(texture(blur_map, adjcorrected_TexCoord)));
	  float adjFrag_blur = v4_adjFrag_blur[0];
	  float dist = sqrt((iterXf*iterXf)/100.0 + (iterYf*iterYf)/100.0);
					
	  //contributionFromAdjFrag = contributionFromAdjFrag + (adjFrag_blur/441.0)*((10.0/1024.0)/distance)*v4_adjFrag_rgb;
	  contributionFromAdjFrag = contributionFromAdjFrag + dist*adjFrag_blur*v4_adjFrag_rgb;
	} 
      }
    }
  }

  FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb + contributionFromAdjFrag/(10.0*10.0);
  // if(corrected_TexCoord.x < 0.5) {
  //   FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb;
  // } else {
  //   FragColor = contributionFromAdjFrag/(10.0*10.0);
  // }
}
