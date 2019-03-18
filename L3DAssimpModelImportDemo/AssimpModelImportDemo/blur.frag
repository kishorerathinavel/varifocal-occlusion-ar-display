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
  float weight = 0.0;
  for (int iterX = -20; iterX < 22; iterX++) {
    for (int iterY = -20; iterY < 21; iterY++) {
      if(iterX == 0 && iterY == 0) {
	continue;
      } else {
	float distance = 0.0;
	if (iterX < 0) {
	  distance = distance - float(iterX)/maxX;
	} else {
	  distance = distance + float(iterX)/maxX;
	}

	if(iterY < 0) {
	  distance = distance - float(iterY)/maxY;
	} else {
	  distance = distance + float(iterY)/maxY;
	}
	

	vec2 adjTexCoord = vec2(TexCoord.x + float(iterX)/maxX, TexCoord.y + float(iterY)/maxY);
	vec4 v4_adjFrag_depth = texture(depth_map, adjTexCoord);
	float adjFrag_depth = v4_adjFrag_depth[0];
	if(adjFrag_depth - centerFrag_depth > 0.01) {
	  continue;
	}
	else {
	  vec4 v4_adjFrag_rgb = texture(rgb_img, adjTexCoord);
	  vec4 v4_adjFrag_blur = vec4(convertDisaprityToBlur(texture(blur_map, adjTexCoord)));
	  float adjFrag_blur = v4_adjFrag_blur[0];
					
	  //contributionFromAdjFrag = contributionFromAdjFrag + (adjFrag_blur/441.0)*((10.0/1024.0)/distance)*v4_adjFrag_rgb;
	  contributionFromAdjFrag = contributionFromAdjFrag + (adjFrag_blur/441.0)*v4_adjFrag_rgb;
	} 
      }
    }
  }

  FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb + (1.0/2.0)*centerFrag_blur*contributionFromAdjFrag;

  // if(gl_FragCoord.x < 256 && gl_FragCoord.y < 384) {
  //   FragColor = centerFrag_blur*contributionFromAdjFrag;
  // }
  // else {
  //   FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb + (1.0/2.0)*centerFrag_blur*contributionFromAdjFrag;
  //   //FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb + contributionFromAdjFrag;
  // }

  /*
    int minX = 0, maxX = 1024, minY = 0, maxY = 768;

  vec4 v4_centerFrag_rgb = texture(rgb_img, TexCoord);
  vec4 v4_centerFrag_blur = texture(blur_map, TexCoord);
  vec4 v4_centerFrag_depth = texture(depth_map, TexCoord);

  float centerFrag_blur = v4_centerFrag_blur[0];
  float centerFrag_depth = v4_centerFrag_depth[0];

  vec4 contributionFromAdjFrag = vec4(0.0);
  float weight = 0.0;
  for (int iterX = -10; iterX < 11; iterX++) {
    for (int iterY = -10; iterY < 11; iterY++) {
      if(iterX == 0 && iterY == 0) {
	continue;
      } else {
	vec2 adjTexCoord = vec2(TexCoord.x + float(iterX)/maxX, TexCoord.y + float(iterY)/maxY);

	if((adjTexCoord.x < minX) || (adjTexCoord.x > maxX) || (adjTexCoord.y < minY) || (adjTexCoord.y > maxY)){
	  continue;
	}
	else {
	  vec4 v4_adjFrag_depth = texture(depth_map, adjTexCoord);
	  float adjFrag_depth = v4_adjFrag_depth[0];
	  if(adjFrag_depth > centerFrag_depth) {
	    continue;
	  }
	  else {
	    vec4 v4_adjFrag_rgb = texture(rgb_img, adjTexCoord);
	    vec4 v4_adjFrag_blur = texture(blur_map, adjTexCoord);
	    float adjFrag_blur = v4_adjFrag_blur[0];
					
	    contributionFromAdjFrag = contributionFromAdjFrag + (1.0 - adjFrag_blur)*v4_adjFrag_rgb;
	    weight = weight + (1.0 - adjFrag_blur);
	  } 
	}
      }
    }
  }

  if(centerFrag_depth == 1.0) {
    FragColor = vec4(0.0);
  } else {
    if(gl_FragCoord.x < 512) {
      //FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb;
      FragColor = (1.0 - centerFrag_blur)*v4_centerFrag_rgb + centerFrag_blur*(1.0/weight)*contributionFromAdjFrag;
      //FragColor = centerFrag_blur*(1.0/weight)*contributionFromAdjFrag;
    }
    else {
      FragColor = vec4(centerFrag_blur*centerFrag_blur);
    }
  } 
  */
  
  /*
    vec4 vec4_z_b = texture(depth_map, TexCoord);
    float z_b = vec4_z_b[0];
    float z_n = 2.0 * z_b - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n *(zFar - zNear));
    float normalized_linear_depth = z_e/zFar;

    float depth_disparity = normalized_linear_depth - focal_depth;
    if(depth_disparity < 0.0) {
    depth_disparity = -1.0*depth_disparity;
    }
    FragColor = vec4(depth_disparity);
    //-----------------------------------------------
    */

  /*
    vec4 vec4_z_b = texture(depth_map, TexCoord);
    float z_b = vec4_z_b[0];
    float z_n = 2.0 * z_b - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n *(zFar - zNear));
    float normalized_linear_depth = z_e/zFar;

    if(TexCoord.x < 0.5) {
    FragColor = texture(rgb_img, TexCoord);
    }
    else {
    //FragColor = texture(depth_map, TexCoord);
    FragColor = vec4(normalized_linear_depth);
    }
    //-----------------------------------------------
    */


  /*
  // 3nd Backup code to verify things are working
  vec4 depth;
  depth = texture(depth_map, TexCoord);
  if(depth[0] > 0.99) {
  FragColor = texture(rgb_img, TexCoord);
  }
  else {
  FragColor = texture(depth_map, TexCoord);
  }
  */

  /*
  // 2nd Backup code to verify things are working
  //if(gl_FragCoord.x > 512) { // This is in screen space based on viewport size https://computergraphics.stackexchange.com/questions/5724/glsl-can-someone-explain-why-gl-fragcoord-xy-screensize-is-performed-and-for
  if(TexCoord.x < 0.5) { // This is in normalized texture coordinates space
  FragColor = texture(blur_map, TexCoord);
  //FragColor = vec4(0.0);
  }
  else {
  FragColor = texture(depth_map, TexCoord);
  }
  */


  // 1st Backup code to verify things are working
  // FragColor = texture(rgb_img, TexCoord);
  //FragColor = texture(blur_map, TexCoord);
  //FragColor = texture(depth_map, TexCoord);
}
