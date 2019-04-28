#version 330

out vec4 FragColor;

in vec2 TexCoord;
in vec4 gl_FragCoord;

uniform sampler2D rgb_img;
uniform sampler2D depth_map;
uniform float zNear;
uniform float zFar;
uniform float focal_depth;

void main() {
  /* The real depth value was mapped to the range [0,1] in a previous shader. 
     The below code does the reverse mapping from [0,1] to the real depth value.
     Code taken from: http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
     Backups of code in Google Drive. Filename: Real Depth in OpenGL_GLSL
   */
  vec4 vec4_z_b = texture(depth_map, TexCoord);
  float z_b = vec4_z_b[0];
  float z_n = 2.0 * z_b - 1.0;
  float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n *(zFar - zNear));
  float normalized_linear_depth = z_e/zFar;
  //float normalized_linear_depth = z_e;

  float depth_disparity;
  if (z_b == 1.0) {
    depth_disparity = 1.0;
  }
  else {
    depth_disparity = normalized_linear_depth - focal_depth;
  }

  if(depth_disparity < 0.0) {
    depth_disparity = -1.0*depth_disparity;
  }

  FragColor = vec4(depth_disparity);
  //-----------------------------------------------

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

  // 2nd Backup code to verify things are working
  /*
  //if(gl_FragCoord.x > 512) { // This is in screen space based on viewport size https://computergraphics.stackexchange.com/questions/5724/glsl-can-someone-explain-why-gl-fragcoord-xy-screensize-is-performed-and-for
  if(TexCoord.x < 0.5) { // This is in normalized texture coordinates space
  FragColor = texture(rgb_img, TexCoord);
  //FragColor = vec4(0.0);
  }
  else {
  FragColor = texture(depth_map, TexCoord);
  }
  */


  // 1st Backup code to verify things are working
  //FragColor = texture(rgb_img, TexCoord);
  //FragColor = texture(depth_map, TexCoord);
}
