#version 330

out vec4 FragColor;

in vec2 TexCoord;
in vec4 gl_FragCoord;

uniform sampler2D rgb_img;
uniform sampler2D depth_map;

void main() {   
	// 1st Backup code to verify things are working
	//FragColor = texture(rgb_img, TexCoord);
	//FragColor = texture(depth_map, TexCoord);

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

	vec4 depth;
	depth = texture(depth_map, TexCoord);
	if(depth[0] < 0.75) {
		FragColor = texture(rgb_img, TexCoord);
	}
	else {
		FragColor = texture(depth_map, TexCoord);
	}
}
