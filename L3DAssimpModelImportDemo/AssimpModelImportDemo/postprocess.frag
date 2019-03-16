#version 120

uniform sampler2D sampler;
uniform sampler2D depth_map;

void main() {   
		gl_FragColor = texture2D(sampler, gl_TexCoord[0].xy);
}
