#version 330

layout (std140) uniform Material {
  vec4 diffuse;
  vec4 ambient;
  vec4 specular;
  vec4 emissive;
  float shininess;
  int texCount;
};

out vec4 FragColor;

uniform	sampler2D texUnit;

in vec2 TexCoord;
in vec3 Normal;
in float Depth;

// // Using all terms of materials
// void main() {
//   vec4 color;
//   vec4 amb;
//   float intensity;
//   vec3 lightDir;
//   vec3 n;
	
//   lightDir = normalize(vec3(1.0,1.0,1.0) - vec3(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.z));
//   //lightDir = normalize(vec3(1.0,1.0,1.0));
//   vec4 lightColor = vec4(2.0, 0.7, 1.3, 1.0);
//   vec3 viewPos = vec3(1.0, 2.0, 1.0);
//   n = normalize(Normal);	
//   diffuse = max(dot(lightDir,n),0.0);
	
//   if (texCount == 0) {
//     color = diffuse;
//     amb = lightColor * ambient;
//   }
//   else {
//     color = texture(texUnit, TexCoord);
//     amb = lightColor * color * 0.23;
//   }
//   FragColor = (color * diffuse) + amb;
//   gl_FragDepth = Depth;
// }

// //From tutorial
// void main() {
//   vec4 color;
//   vec4 amb;
//   float intensity;
//   vec3 lightDir;
//   vec3 n;
	
//   lightDir = normalize(vec3(1.0,1.0,1.0));
//   n = normalize(Normal);	
//   intensity = max(dot(lightDir,n),0.0);
	
//   if (texCount == 0) {
//     color = intensity;
//     amb = ambient;
//   }
//   else {
//     color = texture(texUnit, TexCoord);
//     amb = color * 0.23;
//   }
//   FragColor = (color * intensity) + amb;
//   gl_FragDepth = Depth;
// }

void main()
{
  vec4 color;
  vec4 amb;
  float intensity;
  vec3 lightDir;
  vec3 n;
	
  lightDir = normalize(vec3(1.0,1.0,1.0));
  n = normalize(Normal);	
  intensity = max(dot(lightDir,n),0.0);
	
  if (texCount == 0) {
    color = diffuse;
    amb = ambient;
  }
  else {
    color = texture(texUnit, TexCoord);
    amb = color * 0.23;
  }
  FragColor = (color * intensity) + amb;
  gl_FragDepth = Depth;
}
