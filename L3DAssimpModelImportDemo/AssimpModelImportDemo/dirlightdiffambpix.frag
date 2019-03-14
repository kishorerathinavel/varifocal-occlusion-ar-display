#version 330

layout (std140) uniform Material {
  vec4 diffuse;
  vec4 ambient;
  vec4 specular;
  vec4 emissive;
  float shininess;
  int texCount;
};

uniform	sampler2D texUnit;

in vec2 TexCoord;
in vec3 Normal;
out vec4 output;
in float Depth;

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
  // color = vec4(Depth);
  // output = (color);
  output = (color * intensity) + amb;
  // gl_FragDepth = gl_FragCoord.z/gl_FragCoord.w;
  // gl_FragDepth = (Depth - 20.0)/360.0;
  // output = amb;
}
