//
// Lighthouse3D.com OpenGL 3.3 + GLSL 3.3 Sample
//
// Loading and displaying a Textured Model
//
// Uses:
//  Assimp lybrary for model loading
//		http://assimp.sourceforge.net/
//  Devil for image loading
//		http://openil.sourceforge.net/
//	Uniform Blocks
//  Vertex Array Objects
//
// Some parts of the code are strongly based on the Assimp 
// SimpleTextureOpenGL sample that comes with the Assimp 
// distribution, namely the code that relates to loading the images
// and the model.
//
// The code was updated and modified to be compatible with 
// OpenGL 3.3 CORE version
//
// This demo was built for learning purposes only. 
// Some code could be severely optimised, but I tried to 
// keep as simple and clear as possible.
//
// The code comes with no warranties, use it at your own risk.
// You may use it, or parts of it, wherever you want. 
//
// If you do use it I would love to hear about it. Just post a comment
// at Lighthouse3D.com

// Have Fun :-)

#ifdef _WIN32
#pragma comment(lib,"assimp.lib")
#pragma comment(lib,"devil.lib")
#pragma comment(lib,"glew32.lib")
#endif

#include <windows.h>
// include DevIL for image loading
#include <IL\il.h>

// include GLEW to access OpenGL 3.3 functions
#include <GL/glew.h>

// GLUT is the toolkit to interface with the OS
#include <GL/freeglut.h>
#include <IL\ilut.h>

// auxiliary C file to read the shader text files
#include "textfile.h"

// assimp include files. These three are usually needed.
#include "assimp/Importer.hpp"	//OO version Header!
#include "assimp/PostProcess.h"
#include "assimp/Scene.h"

#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "filepaths.h"
#include <FreeImage.h>

#include "testPythonInterface.h"
#include "lens_controls.h"

bool toggle_om = true;
bool display_on_device = true;
int display_1[] ={ 2560, 1600 };
int display_2[] ={ 1440, 2560 };
int display_3[] ={ 1920, 1080 };
int om_vp_pos[] ={display_2[0], display_2[1] - display_3[1]};
int om_vp_size[] ={1920, 1080};
int vi_vp_pos[] ={650, 1102};
int vi_vp_size[] ={ 313, 176 };
int window_position[] ={display_1[0], 0 };
int window_size[] ={ display_2[0] + display_3[0], 2560 };

// This is for a shader uniform block
struct MyMaterial {
	float diffuse[4];
	float ambient[4];
	float specular[4];
	float emissive[4];
	float shininess;
	int texCount;
};

// Model Matrix (part of the OpenGL Model View Matrix)
float modelMatrix[16];

// For push and pop matrix
std::vector<float *> matrixStack;

float zNear = 0.01, zFar = 11.4;

// For Program 1 ======================================
GLuint rbo_depth_image, fbo_rgbd, tex_rgb, tex_depth;

// Vertex Attribute Locations
GLuint vertexLoc = 0, normalLoc = 1, texCoordLoc = 2;

// Uniform Bindings Points
GLuint matricesUniLoc = 1, materialUniLoc = 2;

// The sampler uniform for textured models
// we are assuming a single texture so this will
//always be texture unit 0
GLuint texUnit;

// Uniform Buffer for Matrices
// this buffer will contain 3 matrices: projection, view and model
// each matrix is a float array with 16 components
GLuint matricesUniBuffer;
#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

// Program and Shader Identifiers
GLuint program, vertexShader, fragmentShader;

// Shader Names
char *fname_vertex_shader = "dirlightdiffambpix.vert";
char *fname_fragment_shader_rgb = "dirlightdiffambpix.frag";
// For Program 1 ======================================

// For Program 2 =====================================
GLuint fbo_blurmap_rgb, tex_blurmap_rgb;

// Vertex Attribute Locations
GLuint blurmap_vertexLoc, blurmap_textureLoc;

// Sampler Uniform
GLuint blurmap_rgb_img, blurmap_depth_map;

// Float Uniforms
GLuint blurmap_zNear_uniformLocation, blurmap_zFar_uniformLocation;
GLuint blurmap_linear_near_uniformLocation, blurmap_linear_far_uniformLocation;
GLuint blurmap_focal_depth_uniformLocation;

// Program and Shader Identifiers
GLuint blurmap_program, blurmap_vertexShader, blurmap_fragmentShader;

// Shader Names
char *fname_blurmap_vertex_shader = "blurmap.vert";
char *fname_blurmap_fragment_shader = "blurmap.frag";

// Values to pass in
float linear_near_value = 0.0, linear_far_value = 1.0;
float focal_depth_value = 0.5;
// For Program 2 =====================================

// For Program 3 =====================================
GLuint blur_zNear_uniformLocation, blur_zFar_uniformLocation;
GLuint fbo_blur_rgb, tex_blur_rgb;

// Vertex Attribute Locations
GLuint blur_vertexLoc, blur_textureLoc;

// Sampler Uniform
GLuint blur_rgb_img, blur_blur_map, blur_depth_map;

// Program and Shader Identifiers
GLuint blur_program, blur_vertexShader, blur_fragmentShader;

// Shader Names
char *fname_blur_fragment_shader = "blur.frag";
// For Program 3 =====================================

// For Program 3 =====================================
GLuint om_zNear_uniformLocation, om_zFar_uniformLocation;
GLuint fbo_om_rgb, tex_om_rgb;

// Vertex Attribute Locations
GLuint om_vertexLoc, om_textureLoc;

// Sampler Uniform
GLuint om_rgb_img, om_blur_map, om_depth_map;

// Program and Shader Identifiers
GLuint om_program, om_vertexShader, om_fragmentShader;

// Shader Names
char *fname_om_fragment_shader = "om.frag";
// For Program 3 =====================================



// Information to render each assimp node
struct MyMesh {
	GLuint vao;
	GLuint texIndex;
	GLuint uniformBlockIndex;
	int numFaces;
};

#define NUM_MODELS 1
class Model {
public:
	std::vector<struct MyMesh> myMesh;
	Assimp::Importer importer;
	const aiScene* scene;
	std::map<std::string, GLuint> textureIdMap;
	std::string basepath;
	std::string modelname;
	float scaleFactor;
	float translation[3];
	float rotation[3];

	Model() {
		scene = NULL;
		scaleFactor = 0.05;
	}

	~Model() {
		textureIdMap.clear();
		// clear myMeshes stuff
		for (unsigned int i = 0; i < myMesh.size(); ++i) {
			glDeleteVertexArrays(1, &(myMesh[i].vao));
			glDeleteTextures(1, &(myMesh[i].texIndex));
			glDeleteBuffers(1, &(myMesh[i].uniformBlockIndex));
		}
	}
}model[NUM_MODELS];

// Camera Position
float camX = 0, camY = 0, camZ = 1.2;

// Mouse Tracking Variables
int startX, startY, tracking = 0;

// Camera Spherical Coordinates
float alpha = 0.0f, beta = 0.0f;
float r = 1.2f;

bool saveFramebufferOnce = true;
bool saveFramebufferUntilStop = false;

GLuint tex_background;

float postprocess_vertices[] ={
	// positions     // texture coords
	1.0f, 1.0f, 0.0f, 1.0f, 1.0f, // top right
	1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom right
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, // bottom left
	0.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top left 
};
//float postprocess_vertices[] ={
//	// positions     // texture coords
//	1.0f, 1.0f, 0.0f, 1.0f, 1.0f, // top right
//	1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
//	-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
//	-1.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top left 
//};
unsigned int postprocess_indices[] ={
	0, 1, 3, // first triangle
	1, 2, 3  // second triangle
};
unsigned int postprocess_VBO, postprocess_VAO, postprocess_EBO;


#define M_PI       3.14159265358979323846f

static inline float
DegToRad(float degrees)
{
	return (float)(degrees * (M_PI / 180.0f));
};

// Frame counting and FPS computation
long time_fps, timebase = 0, frame = 0;
char s[32];

//-----------------------------------------------------------------
// Print for OpenGL errors
//
// Returns 1 if an OpenGL error occurred, 0 otherwise.
//

#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char *file, int line)
{

	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	if (glErr != GL_NO_ERROR)
	{
		printf("glError in file %s @ line %d: %s\n",
			file, line, gluErrorString(glErr));
		retCode = 1;
	}
	return retCode;
}


// ----------------------------------------------------
// VECTOR STUFF
//


// res = a cross b;
void crossProduct(float *a, float *b, float *res) {

	res[0] = a[1] * b[2] - b[1] * a[2];
	res[1] = a[2] * b[0] - b[2] * a[0];
	res[2] = a[0] * b[1] - b[0] * a[1];
}


// Normalize a vec3
void normalize(float *a) {

	float mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

	a[0] /= mag;
	a[1] /= mag;
	a[2] /= mag;
}


// ----------------------------------------------------
// MATRIX STUFF
//

// Push and Pop for modelMatrix

void pushMatrix() {

	float *aux = (float *)malloc(sizeof(float) * 16);
	memcpy(aux, modelMatrix, sizeof(float) * 16);
	matrixStack.push_back(aux);
}

void popMatrix() {

	float *m = matrixStack[matrixStack.size() - 1];
	memcpy(modelMatrix, m, sizeof(float) * 16);
	matrixStack.pop_back();
	free(m);
}

// sets the square matrix mat to the identity matrix,
// size refers to the number of rows (or columns)
void setIdentityMatrix(float *mat, int size) {

	// fill matrix with 0s
	for (int i = 0; i < size * size; ++i)
		mat[i] = 0.0f;

	// fill diagonal with 1s
	for (int i = 0; i < size; ++i)
		mat[i + i * size] = 1.0f;
}


//
// a = a * b;
//
void multMatrix(float *a, float *b) {

	float res[16];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			res[j * 4 + i] = 0.0f;
			for (int k = 0; k < 4; ++k) {
				res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
			}
		}
	}
	memcpy(a, res, 16 * sizeof(float));

}


// Defines a transformation matrix mat with a translation
void setTranslationMatrix(float *mat, float x, float y, float z) {

	setIdentityMatrix(mat, 4);
	mat[12] = x;
	mat[13] = y;
	mat[14] = z;
}

// Defines a transformation matrix mat with a scale
void setScaleMatrix(float *mat, float sx, float sy, float sz) {

	setIdentityMatrix(mat, 4);
	mat[0] = sx;
	mat[5] = sy;
	mat[10] = sz;
}

// Defines a transformation matrix mat with a rotation 
// angle alpha and a rotation axis (x,y,z)
void setRotationMatrix(float *mat, float angle, float x, float y, float z) {

	float radAngle = DegToRad(angle);
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;

	mat[0] = x2 + (y2 + z2) * co;
	mat[4] = x * y * (1 - co) - z * si;
	mat[8] = x * z * (1 - co) + y * si;
	mat[12] = 0.0f;

	mat[1] = x * y * (1 - co) + z * si;
	mat[5] = y2 + (x2 + z2) * co;
	mat[9] = y * z * (1 - co) - x * si;
	mat[13] = 0.0f;

	mat[2] = x * z * (1 - co) - y * si;
	mat[6] = y * z * (1 - co) + x * si;
	mat[10] = z2 + (x2 + y2) * co;
	mat[14] = 0.0f;

	mat[3] = 0.0f;
	mat[7] = 0.0f;
	mat[11] = 0.0f;
	mat[15] = 1.0f;

}

// ----------------------------------------------------
// Model Matrix 
//
// Copies the modelMatrix to the uniform buffer


void setModelMatrix() {
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ModelMatrixOffset, MatrixSize, modelMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// The equivalent to glTranslate applied to the model matrix
void translate(float x, float y, float z) {

	float aux[16];

	setTranslationMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glRotate applied to the model matrix
void rotate(float angle, float x, float y, float z) {

	float aux[16];

	setRotationMatrix(aux, angle, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glScale applied to the model matrix
void scale(float x, float y, float z) {

	float aux[16];

	setScaleMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// ----------------------------------------------------
// Projection Matrix 
//
// Computes the projection Matrix and stores it in the uniform buffer

void buildProjectionMatrix(float fov, float ratio, float nearp, float farp) {

	float projMatrix[16];

	float f = 1.0f / tan(fov * (M_PI / 360.0f));

	setIdentityMatrix(projMatrix, 4);

	projMatrix[0] = f / ratio;
	projMatrix[1 * 4 + 1] = f;
	projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
	projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
	projMatrix[2 * 4 + 3] = -1.0f;
	projMatrix[3 * 4 + 3] = 0.0f;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}


// ----------------------------------------------------
// View Matrix
//
// Computes the viewMatrix and stores it in the uniform buffer
//
// note: it assumes the camera is not tilted, 
// i.e. a vertical up vector along the Y axis (remember gluLookAt?)
//

void setCamera(float posX, float posY, float posZ,
	float lookAtX, float lookAtY, float lookAtZ) {

	float dir[3], right[3], up[3];

	up[0] = 0.0f;	up[1] = 1.0f;	up[2] = 0.0f;

	dir[0] = (lookAtX - posX);
	dir[1] = (lookAtY - posY);
	dir[2] = (lookAtZ - posZ);
	normalize(dir);

	crossProduct(dir, up, right);
	normalize(right);

	crossProduct(right, dir, up);
	normalize(up);

	float viewMatrix[16], aux[16];

	viewMatrix[0] = right[0];
	viewMatrix[4] = right[1];
	viewMatrix[8] = right[2];
	viewMatrix[12] = 0.0f;

	viewMatrix[1] = up[0];
	viewMatrix[5] = up[1];
	viewMatrix[9] = up[2];
	viewMatrix[13] = 0.0f;

	viewMatrix[2] = -dir[0];
	viewMatrix[6] = -dir[1];
	viewMatrix[10] = -dir[2];
	viewMatrix[14] = 0.0f;

	viewMatrix[3] = 0.0f;
	viewMatrix[7] = 0.0f;
	viewMatrix[11] = 0.0f;
	viewMatrix[15] = 1.0f;

	setTranslationMatrix(aux, -posX, -posY, -posZ);

	multMatrix(viewMatrix, aux);

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, viewMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}




// ----------------------------------------------------------------------------

#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

void get_bounding_box_for_node(const aiNode* nd,
	aiVector3D* min,
	aiVector3D* max, const aiScene* scene)

{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];

			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		get_bounding_box_for_node(nd->mChildren[n], min, max, scene);
	}
}


void get_bounding_box(aiVector3D* min, aiVector3D* max, const aiScene* scene) {
	min->x = min->y = min->z = 1e10f;
	max->x = max->y = max->z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, min, max, scene);
}

bool Import3DFromFile(Model &model) {

	std::string pFile = model.basepath + model.modelname;
	//check if file exists
	std::ifstream fin(pFile.c_str());
	if (!fin.fail()) {
		fin.close();
	}
	else {
		printf("Couldn't open file: %s\n", pFile.c_str());
		printf("%s\n", model.importer.GetErrorString());
		return false;
	}

	printf("Reading file... \n");
	model.scene = model.importer.ReadFile(pFile, aiProcessPreset_TargetRealtime_Quality);
	// If the import failed, report it
	if (!model.scene)
	{
		printf("%s\n", model.importer.GetErrorString());
		return false;
	}

	printf("Done reading file... \n");


	// Now we can access the file's contents.
	printf("Import of scene %s succeeded. \n", pFile.c_str());

	//float tempScaleFactor;
	//aiVector3D scene_min, scene_max, scene_center;
	//get_bounding_box(&scene_min, &scene_max, scene);
	//float tmp;
	//tmp = scene_max.x-scene_min.x;
	//tmp = scene_max.y - scene_min.y > tmp?scene_max.y - scene_min.y:tmp;
	//tmp = scene_max.z - scene_min.z > tmp?scene_max.z - scene_min.z:tmp;
	//tempScaleFactor = 1.0 / tmp;

	// We're done. Everything will be cleaned up by the importer destructor
	return true;
}


int LoadGLTextures(Model& model) {
	ILboolean success;


	/* scan scene's materials for textures */
	for (unsigned int m = 0; m < model.scene->mNumMaterials; ++m)
	{
		int texIndex = 0;
		aiString path;	// filename

		aiReturn texFound = model.scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		while (texFound == AI_SUCCESS) {
			//fill map with textures, OpenGL image ids set to 0
			model.textureIdMap[path.data] = 0;
			// more textures?
			texIndex++;
			texFound = model.scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		}
	}

	int numTextures = model.textureIdMap.size();

	/* create and fill array with DevIL texture ids */
	ILuint* imageIds = new ILuint[numTextures];
	ilGenImages(numTextures, imageIds);

	/* create and fill array with GL texture ids */
	GLuint* textureIds = new GLuint[numTextures];
	glGenTextures(numTextures, textureIds); /* Texture name generation */

	/* get iterator */
	std::map<std::string, GLuint>::iterator itr = model.textureIdMap.begin();
	int i = 0;
	for (; itr != model.textureIdMap.end(); ++i, ++itr)
	{
		//save IL image ID
		std::string filename = (*itr).first;  // get filename
		(*itr).second = textureIds[i];	  // save texture id for filename in map

		ilBindImage(imageIds[i]); /* Binding of DevIL image name */
		ilEnable(IL_ORIGIN_SET);
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
		std::string fileloc = model.basepath + filename;	/* Loading of image */
		success = ilLoadImage(fileloc.c_str());
		if (success) {
			/* Convert image to RGBA */
			ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);

			/* Create and load textures to OpenGL */
			glBindTexture(GL_TEXTURE_2D, textureIds[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ilGetInteger(IL_IMAGE_WIDTH),
				ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_RGBA, GL_UNSIGNED_BYTE,
				ilGetData());
		}
		else
			printf("Couldn't load Image: %s\n", filename.c_str());
	}
	/* Because we have already copied image data into texture data
	we can release memory used by image. */
	ilDeleteImages(numTextures, imageIds);

	//Cleanup
	delete[] imageIds;
	delete[] textureIds;

	//return success;
	return true;
}



//// Can't send color down as a pointer to aiColor4D because AI colors are ABGR.
//void Color4f(const aiColor4D *color)
//{
//	glColor4f(color->r, color->g, color->b, color->a);
//}

void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

void color4_to_float4(const aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

void genVAOs_blurmap() {
	glGenVertexArrays(1, &postprocess_VAO);
	glGenBuffers(1, &postprocess_VBO);
	glGenBuffers(1, &postprocess_EBO);

	glBindVertexArray(postprocess_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, postprocess_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(postprocess_vertices), postprocess_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, postprocess_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(postprocess_indices), postprocess_indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(blurmap_vertexLoc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(blurmap_vertexLoc);
	// texture coord attribute
	glVertexAttribPointer(blurmap_textureLoc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(blurmap_textureLoc);
}

void genVAOs_om() {
	glGenVertexArrays(1, &postprocess_VAO);
	glGenBuffers(1, &postprocess_VBO);
	glGenBuffers(1, &postprocess_EBO);

	glBindVertexArray(postprocess_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, postprocess_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(postprocess_vertices), postprocess_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, postprocess_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(postprocess_indices), postprocess_indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(om_vertexLoc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(om_vertexLoc);
	// texture coord attribute
	glVertexAttribPointer(om_textureLoc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(om_textureLoc);
}

void genVAOs_blur() {
	glGenVertexArrays(1, &postprocess_VAO);
	glGenBuffers(1, &postprocess_VBO);
	glGenBuffers(1, &postprocess_EBO);

	glBindVertexArray(postprocess_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, postprocess_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(postprocess_vertices), postprocess_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, postprocess_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(postprocess_indices), postprocess_indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(blur_vertexLoc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(blur_vertexLoc);
	// texture coord attribute
	glVertexAttribPointer(blur_textureLoc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(blur_textureLoc);
}


void genVAOsAndUniformBuffer(Model& model) {

	struct MyMesh aMesh;
	struct MyMaterial aMat;
	GLuint buffer;

	// For each mesh
	for (unsigned int n = 0; n < model.scene->mNumMeshes; ++n) {
		const aiMesh* mesh = model.scene->mMeshes[n];

		// create array with faces
		// have to convert from Assimp format to array
		unsigned int *faceArray;
		faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
		unsigned int faceIndex = 0;

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];

			memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
			faceIndex += 3;
		}
		aMesh.numFaces = model.scene->mMeshes[n]->mNumFaces;

		// generate Vertex Array for mesh
		glGenVertexArrays(1, &(aMesh.vao));
		glBindVertexArray(aMesh.vao);

		// buffer for faces
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, GL_STATIC_DRAW);

		// buffer for vertex positions
		if (mesh->HasPositions()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(vertexLoc);
			glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex normals
		if (mesh->HasNormals()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mNormals, GL_STATIC_DRAW);
			glEnableVertexAttribArray(normalLoc);
			glVertexAttribPointer(normalLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex texture coordinates
		if (mesh->HasTextureCoords(0)) {
			float *texCoords = (float *)malloc(sizeof(float) * 2 * mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

				texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
				texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;

			}
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * mesh->mNumVertices, texCoords, GL_STATIC_DRAW);
			glEnableVertexAttribArray(texCoordLoc);
			glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);
		}

		// unbind buffers
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// create material uniform buffer
		aiMaterial *mtl = model.scene->mMaterials[mesh->mMaterialIndex];

		aiString texPath;	//contains filename of texture
		if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)) {
			//bind texture
			unsigned int texId = model.textureIdMap[texPath.data];
			aMesh.texIndex = texId;
			aMat.texCount = 1;
		}
		else
			aMat.texCount = 0;

		float c[4];
		set_float4(c, 0.5f, 0.5f, 0.5f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);
		memcpy(aMat.diffuse, c, sizeof(c));

		set_float4(c, 0.1f, 0.1f, 0.1f, 1.0f);
		aiColor4D ambient;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
			color4_to_float4(&ambient, c);
		memcpy(aMat.ambient, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D specular;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
			color4_to_float4(&specular, c);
		memcpy(aMat.specular, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D emission;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
			color4_to_float4(&emission, c);
		memcpy(aMat.emissive, c, sizeof(c));

		float shininess = 0.0;
		unsigned int max;
		aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
		aMat.shininess = shininess;

		glGenBuffers(1, &(aMesh.uniformBlockIndex));
		glBindBuffer(GL_UNIFORM_BUFFER, aMesh.uniformBlockIndex);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(aMat), (void *)(&aMat), GL_STATIC_DRAW);

		model.myMesh.push_back(aMesh);
	}
}


// ------------------------------------------------------------
//
// Reshape Callback Function
//
float ratio;
void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	ratio = (1.0f * w) / h;
	buildProjectionMatrix(35.0f, ratio, zNear, zFar);
}


// ------------------------------------------------------------
//
// Render stuff
//

// Render Assimp Model

void recursive_render(Model& model, const aiNode* nd)
{

	// Get node transformation matrix
	aiMatrix4x4 m = nd->mTransformation;
	// OpenGL matrices are column major
	m.Transpose();

	// save model matrix and apply node transformation
	pushMatrix();

	float aux[16];
	memcpy(aux, &m, sizeof(float) * 16);
	multMatrix(modelMatrix, aux);
	setModelMatrix();


	// draw all meshes assigned to this node
	for (unsigned int n = 0; n < nd->mNumMeshes; ++n) {
		// bind material uniform
		glBindBufferRange(GL_UNIFORM_BUFFER, materialUniLoc, model.myMesh[nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MyMaterial));
		// bind texture
		glBindTexture(GL_TEXTURE_2D, model.myMesh[nd->mMeshes[n]].texIndex);
		// bind VAO
		glBindVertexArray(model.myMesh[nd->mMeshes[n]].vao);
		// draw
		glDrawElements(GL_TRIANGLES, model.myMesh[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);

	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n) {
		recursive_render(model, nd->mChildren[n]);
	}
	popMatrix();
}

ILuint imageID;
void saveScreenShot(char* fname) {
	imageID = ilGenImage();
	ilBindImage(imageID);
	ilutGLScreen();
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(fname);
	//ilDeleteImage(imageID);
}

void saveImage(GLuint fbo, const char* outFilename1, const char* outFilename2) {
	//allocate FreeImage memory
	int width = 1920, height = 1080;
	int oldFramebuffer;

	FIBITMAP *depth_img = FreeImage_Allocate(width, height, 32);
	if (depth_img == NULL) {
		printf("couldn't allocate depth_img for saving!\n");
		return;
	}

	FIBITMAP *color_img = FreeImage_Allocate(width, height, 24);
	if (color_img == NULL) {
		printf("couldn't allocate color_img for saving!\n");
		return;
	}

	//save existing bound FBO
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFramebuffer);

	//bind desired FBO
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_INT, FreeImage_GetBits(depth_img));
	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, FreeImage_GetBits(color_img));

	//restore existing FBO
	glBindFramebuffer(GL_FRAMEBUFFER, oldFramebuffer);

	//write depth_img
	FreeImage_Save(FreeImage_GetFIFFromFilename(outFilename1), depth_img, outFilename1);
	//write color_img
	FreeImage_Save(FreeImage_GetFIFFromFilename(outFilename2), color_img, outFilename2);

	//deallocate
	FreeImage_Unload(depth_img);
	//deallocate
	FreeImage_Unload(color_img);
}

void drawModels() {
	// set the model matrix to the identity Matrix
	for (int modelIter = 0; modelIter < NUM_MODELS; modelIter++) {
		setIdentityMatrix(modelMatrix, 4);
		translate(model[modelIter].translation[0], model[modelIter].translation[1], model[modelIter].translation[2]);
		rotate(model[modelIter].rotation[0], 1.0f, 0.0f, 0.0f);		// use our shader
		rotate(model[modelIter].rotation[1], 0.0f, 1.0f, 0.0f);		// use our shader
		rotate(model[modelIter].rotation[2], 0.0f, 0.0f, 1.0f);		// use our shader
		scale(model[modelIter].scaleFactor, model[modelIter].scaleFactor, model[modelIter].scaleFactor);
		recursive_render(model[modelIter], model[modelIter].scene->mRootNode);
	}
}

void savePosition() {
	// save position information for each model
	std::ofstream Position;
	Position.open("Position.txt",std::ios::trunc);
	Position << "N " << NUM_MODELS << std::endl;
	
	for (int modelIter = 0; modelIter < NUM_MODELS; modelIter++) {
	
		Position << "M " << modelIter << std::endl;
		Position << "T " << model[modelIter].translation[0]<<" "<< model[modelIter].translation[1]<<" "<< model[modelIter].translation[2]<<std::endl;
		Position << "R " << model[modelIter].rotation[0] << " " << model[modelIter].rotation[1] << " " << model[modelIter].rotation[2] << std::endl;
		Position << "S " << model[modelIter].scaleFactor << std::endl;
	}

	Position <<"C " << r <<" "<<alpha <<" "<<beta<< std::endl;
	Position.close();
}


Model *currModel = &model[0];
void usePosition() {
	FILE* fp;
	float x, y, z;
	int c1, c2;
	int mn;

	fp = fopen("Position.txt", "rb");

	if (fp == NULL) {
		printf("Error loading Position \n");
		exit(-1);
	}

	while (!feof(fp)) {
		c1 = fgetc(fp);
		

		while (!(c1 == 'M' || c1 == 'T' || c1 == 'R' || c1 == 'S' || c1 == 'C')) {
			c1 = fgetc(fp);
			if (feof(fp))
				break;
		}

		c2 = fgetc(fp);

		if ((c1 == 'M') && (c2 == ' ')) {
			fscanf(fp, "%d", &mn);
			currModel = &model[mn];
		}

		if ((c1 == 'T') && (c2 == ' ')) {
			fscanf(fp,"%f %f %f", &x,&y,&z);
			currModel->translation[0] = x;
			currModel->translation[1] = y;
			currModel->translation[2] = z;
		}

		if ((c1 == 'R') && (c2 == ' ')) {
			fscanf(fp, "%f %f %f", &x, &y, &z);
			currModel->rotation[0] = x;
			currModel->rotation[1] = y;
			currModel->rotation[2] = z;
		}

		if ((c1 == 'S') && (c2 == ' ')) {
			fscanf(fp, "%f", &x);
			currModel->scaleFactor = x;
		}

		if ((c1 == 'C') && (c2 == ' ')) {
			fscanf(fp, "%f %f %f", &x, &y, &z);
			r = x;
			alpha = y;
			beta = z;
         }
	}
	fclose(fp);
	currModel = &model[0];
}

void drawTextureToFramebuffer(int textureID) {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1, 1, 1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
	glTexCoord2f(1, 0); glVertex3f(1, 0, 0);
	glTexCoord2f(1, 1); glVertex3f(1, 1, 0);
	glTexCoord2f(0, 1); glVertex3f(0, 1, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

bool rgb = true;
int imgCounter = 0;
char fname[1024], fname1[1024], fname2[1024];
// Rendering Callback Function
void renderScene() {

	glBindFramebuffer(GL_FRAMEBUFFER, fbo_rgbd);
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, 1920, 1080);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setCamera(camX, camY, camZ, 0, 0, 0);
	buildProjectionMatrix(35.0f, ratio, zNear, zFar);

	glUseProgram(program);
	// we are only going to use texture unit 0
	// unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately
	//glUniform1i(texUnit, tex_background);

	drawModels();

	if (saveFramebufferOnce | saveFramebufferUntilStop) {
		sprintf(fname1, "./outputs/trial_%02d_depth.png", imgCounter);
		sprintf(fname2, "./outputs/trial_%02d_rgb.png", imgCounter);
		saveImage(fbo_rgbd, fname1, fname2);
		imgCounter++;
		saveFramebufferOnce = false;
	}

	// FPS computation and display
	frame++;
	time_fps = glutGet(GLUT_ELAPSED_TIME);
	if (time_fps - timebase > 1000) {
		sprintf(s, "FPS: %4.2f %d", frame*1000.0 / (time_fps - timebase), frame);
		timebase = time_fps;
		frame = 0;
		glutSetWindowTitle(s);
	}
	glPopAttrib();

	glBindFramebuffer(GL_FRAMEBUFFER, fbo_blurmap_rgb);
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, 1920, 1080);

	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bool old1new0 = 0;
	if (old1new0 == 1) { // Uses fixed pipeline
		glUseProgram(0);
		//drawTextureToFramebuffer(tex_background);
		drawTextureToFramebuffer(tex_depth);
		//drawTextureToFramebuffer(tex_rgb);
	}
	else { // Uses shaders
		glUseProgram(blurmap_program);

		// Important that these two lines come after the glUseProgram() command
		glUniform1i(blurmap_rgb_img, 0);
		glUniform1i(blurmap_depth_map, 1);
		glUniform1f(blurmap_zFar_uniformLocation, zFar);
		glUniform1f(blurmap_zNear_uniformLocation, zNear);
		glUniform1f(blurmap_linear_near_uniformLocation, linear_near_value);
		glUniform1f(blurmap_linear_far_uniformLocation, linear_far_value);
		glUniform1f(blurmap_focal_depth_uniformLocation, focal_depth_value);
		glEnable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, tex_rgb);
		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, tex_depth);

		glBindVertexArray(postprocess_VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glDisable(GL_TEXTURE_2D);
		// Important to set default active texture back to GL_TEXTURE0
		glActiveTexture(GL_TEXTURE0);
	}
	glPopAttrib();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (display_on_device) {
		glPushAttrib(GL_VIEWPORT_BIT);

		// Displaying the virtual image
		glViewport(vi_vp_pos[0], vi_vp_pos[1], vi_vp_size[0], vi_vp_size[1]);
		old1new0 = 0;
		if (old1new0 == 1) { // Uses fixed pipeline
			glUseProgram(0);
			drawTextureToFramebuffer(tex_blurmap_rgb);
		}
		else { // Uses shaders
			glUseProgram(blur_program);

			// Important that these two lines come after the glUseProgram() command
			glUniform1i(blur_rgb_img, 0);
			glUniform1i(blur_blur_map, 1);
			glUniform1i(blur_depth_map, 2);
			glUniform1f(blur_zFar_uniformLocation, zFar);
			glEnable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0 + 0);
			glBindTexture(GL_TEXTURE_2D, tex_rgb);
			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_2D, tex_blurmap_rgb);
			glActiveTexture(GL_TEXTURE0 + 2);
			glBindTexture(GL_TEXTURE_2D, tex_depth);

			glBindVertexArray(postprocess_VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glDisable(GL_TEXTURE_2D);
			// Important to set default active texture back to GL_TEXTURE0
			glActiveTexture(GL_TEXTURE0);
		}

		// Displaying the occlusion image
		glViewport(om_vp_pos[0], om_vp_pos[1], om_vp_size[0], om_vp_size[1]);
		if (toggle_om) {
			old1new0 = 0;
			if (old1new0 == 1) { // Uses fixed pipeline
				glUseProgram(0);
				drawTextureToFramebuffer(tex_depth);
			}
			else { // Uses shaders
				glUseProgram(om_program);

				// Important that these two lines come after the glUseProgram() command
				glUniform1i(om_rgb_img, 0);
				glUniform1i(om_blur_map, 1);
				glUniform1i(om_depth_map, 2);
				glUniform1f(om_zFar_uniformLocation, zFar);
				glEnable(GL_TEXTURE_2D);
				glActiveTexture(GL_TEXTURE0 + 0);
				glBindTexture(GL_TEXTURE_2D, tex_rgb);
				glActiveTexture(GL_TEXTURE0 + 1);
				glBindTexture(GL_TEXTURE_2D, tex_blurmap_rgb);
				glActiveTexture(GL_TEXTURE0 + 2);
				glBindTexture(GL_TEXTURE_2D, tex_depth);

				glBindVertexArray(postprocess_VAO);
				glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
				glDisable(GL_TEXTURE_2D);
				// Important to set default active texture back to GL_TEXTURE0
				glActiveTexture(GL_TEXTURE0);
			}
		}
		else {
			glUseProgram(0);
			drawTextureToFramebuffer(tex_background);
		}
		glPopAttrib();
		glUseProgram(0);
	}
	else {
		glPushAttrib(GL_VIEWPORT_BIT);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, 512, 384);

		old1new0 = 0;
		if (old1new0 == 1) { // Uses fixed pipeline
			glUseProgram(0);
			drawTextureToFramebuffer(tex_blurmap_rgb);
		}
		else { // Uses shaders
			glUseProgram(blur_program);

			// Important that these two lines come after the glUseProgram() command
			glUniform1i(blur_rgb_img, 0);
			glUniform1i(blur_blur_map, 1);
			glUniform1i(blur_depth_map, 2);
			glUniform1f(blur_zFar_uniformLocation, zFar);
			glEnable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0 + 0);
			glBindTexture(GL_TEXTURE_2D, tex_rgb);
			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_2D, tex_blurmap_rgb);
			glActiveTexture(GL_TEXTURE0 + 2);
			glBindTexture(GL_TEXTURE_2D, tex_depth);

			glBindVertexArray(postprocess_VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glDisable(GL_TEXTURE_2D);
			// Important to set default active texture back to GL_TEXTURE0
			glActiveTexture(GL_TEXTURE0);
		}

		glViewport(0, 384, 512, 384);
		old1new0 = 0;
		if (old1new0 == 1) { // Uses fixed pipeline
			glUseProgram(0);
			drawTextureToFramebuffer(tex_depth);
		}
		else { // Uses shaders
			glUseProgram(om_program);

			// Important that these two lines come after the glUseProgram() command
			glUniform1i(om_rgb_img, 0);
			glUniform1i(om_blur_map, 1);
			glUniform1i(om_depth_map, 2);
			glUniform1f(om_zFar_uniformLocation, zFar);
			glEnable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0 + 0);
			glBindTexture(GL_TEXTURE_2D, tex_rgb);
			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_2D, tex_blurmap_rgb);
			glActiveTexture(GL_TEXTURE0 + 2);
			glBindTexture(GL_TEXTURE_2D, tex_depth);

			glBindVertexArray(postprocess_VAO);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
			glDisable(GL_TEXTURE_2D);
			// Important to set default active texture back to GL_TEXTURE0
			glActiveTexture(GL_TEXTURE0);
		}

		glUseProgram(0);
		glViewport(512, 0, 512, 384);
		drawTextureToFramebuffer(tex_blurmap_rgb);

		glViewport(512, 384, 512, 384);
		drawTextureToFramebuffer(tex_rgb);

		glPopAttrib();

	}

	// swap buffers
	glutSwapBuffers();
}

int modify_valuei(int value, int delta, int min_value, int max_value) {
	value = value + delta;
	if (value < min_value) {
		value = min_value;
	}

	if (value > max_value) {
		value = max_value;
	}

	return value;
}


void print_valuei(int value, char name[]) {
	printf("%s: %d \n", name, value);
}

float modify_valuef(float value, float delta, float min_value, float max_value) {
	value = value + delta;
	if (value < min_value) {
		value = min_value;
	}

	if (value > max_value) {
		value = max_value;
	}

	return value;
}

void updateCamVariables() {
	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);

	// set camera matrix
	setCamera(camX, camY, camZ, 0, 0, 0);

}


// ------------------------------------------------------------
//
// Events from the Keyboard
//
float stepSize = 0.1;
int keymapmode = 1;
void processKeys(unsigned char key, int xx, int yy) {
	if (key == 27) {
		glutLeaveMainLoop();
	}

	if (key == '`') {
		keymapmode = (keymapmode++) % 4;
		printf("keymapmode: %d \n", keymapmode);
	}
	else {
		if (keymapmode == 1) {
			/*
			Remaining letters:
			  e     op
			      jkl
			    b  
			*/

			switch (key) {
			case 'p': toggle_om = !toggle_om; break;
			case 'q': decrement_index(true, true); break;
			case 'w': increment_index(true, true); break;
			case 'a': decrement_index(true, false); break;
			case 's': increment_index(true, false); break;
			case 'z': decrement_index(false, true); break;
			case 'x': increment_index(false, true); break;
			case 'n': set_fl_absolute_middle(); break;
			case 'm': set_fl_middle(); break;
			case 'r': reset_orig_fl(); break;
			case 'd': modify_current_fl(1, -0.1); break;
			case 'f': modify_current_fl(1, 0.1); break;
			case 'c': modify_current_fl(4, -0.1); break;
			case 'v': modify_current_fl(4, 0.1); break;
			case 't': linear_near_value = modify_valuef(linear_near_value, -0.01, 0.0, linear_far_value); printf("linear_near_value: %f \n", linear_near_value);  break;
			case 'y': linear_near_value = modify_valuef(linear_near_value,  0.01, 0.0, linear_far_value); printf("linear_near_value: %f \n", linear_near_value); break;
			case 'g': linear_far_value = modify_valuef(linear_far_value, -0.01, linear_near_value, 1.0); printf("linear_far_value: %f \n", linear_far_value); break;
			case 'h': linear_far_value = modify_valuef(linear_far_value,  0.01, linear_near_value, 1.0); printf("linear_far_value: %f \n", linear_far_value); break;
			case 'j': focal_depth_value = modify_valuef(focal_depth_value, -0.01, 0.0, 1.0); printf("focal_depth_value: %f \n", focal_depth_value); break;
			case 'k': focal_depth_value = modify_valuef(focal_depth_value, 0.01, 0.0, 1.0); printf("focal_depth_value: %f \n", focal_depth_value); break;
			case 'u': zFar = modify_valuef(zFar, -0.1, zNear, 200.0); printf("zFar: %f \n", zFar); break;
			case 'i': zFar = modify_valuef(zFar, 0.1, zNear, 200.0); printf("zFar: %f \n", zFar); break;
			default: printf("Entered key does nothing \n");
			}
		}
		else if (keymapmode == 2) {
			/*
			Remaining letters:
			qwertyuiop
			asdfghjkl
			zxcvbnm
			*/

			switch (key) {
			case 27: {
						 glutLeaveMainLoop();
						 break;
			}
			case 'z': r -= 0.1f; break;
			case 'x': r += 0.1f; break;
			case 'm': glEnable(GL_MULTISAMPLE); break;
			case 'M': glDisable(GL_MULTISAMPLE); break;
			case '1': currModel = &model[0]; printf("Current Model is 1 \n"); break;
			case '2': currModel = &model[1]; printf("Current Model is 2 \n"); break;
			case '3': currModel = &model[2]; printf("Current Model is 2 \n"); break;
			case '9': stepSize = stepSize / 3.0; break;
			case '0': stepSize = stepSize * 3.0; break;
			case 's': rgb = true; saveFramebufferOnce = true; printf("Saving framebuffer \n"); break;
			case 'S': {
						  rgb = true;
						  saveFramebufferUntilStop = !saveFramebufferUntilStop;
						  if (saveFramebufferUntilStop) {
							  printf("Saving framebuffer until stop. Press S again to stop \n");
						  }
						  else {
							  printf("Stoped saving \n");
						  }
						  break;
			}
			case 'q': {
						  currModel->scaleFactor -= 0.005f*stepSize;
						  if (currModel->scaleFactor < 0.005)
							  currModel->scaleFactor = 0.005;
						  printf("currModel->scaleFactor: %f\n", currModel->scaleFactor);
						  break;

			}
			case 'w': {
						  currModel->scaleFactor += 0.005f*stepSize;
						  printf("currModel->scaleFactor: %f\n", currModel->scaleFactor);
						  break;
			}
			case 'e': currModel->rotation[0] -= stepSize; break;
			case 'r': currModel->rotation[0] += stepSize; break;
			case 'd': currModel->rotation[1] -= stepSize; break;
			case 'f': currModel->rotation[1] += stepSize; break;
			case 'c': currModel->rotation[2] -= stepSize; break;
			case 'v': currModel->rotation[2] += stepSize; break;
			case 't': currModel->translation[0] -= stepSize; break;
			case 'y': currModel->translation[0] += stepSize; break;
			case 'g': currModel->translation[1] -= stepSize; break;
			case 'h': currModel->translation[1] += stepSize; break;
			case 'b': currModel->translation[2] -= stepSize; break;
			case 'n': currModel->translation[2] += stepSize; break;
			case 'p': savePosition(); printf("Saving Position Information \n"); break;
			case 'u': usePosition(); break;
			default: printf("Entered key does nothing \n");
			}
			updateCamVariables();
		}
		else if (keymapmode == 3) {
			//int display_1[] ={ 2560, 1600 };
			//int display_2[] ={ 1440, 2560 };
			//int display_3[] ={ 1920, 1080 };
			//int om_vp_pos[] ={display_1[0] + display_2[0], 0};
			//int om_vp_size[] ={1920, 1080};
			//int vi_vp_pos[] ={display_1[0],0};
			//int vi_vp_size[] ={1920,1080};
			/*
			Remaining letters:
			qwertyuiop
			asdfghjkl
			zxcvbnm
			*/

			switch (key) {
			case 'q': vi_vp_pos[0] = modify_valuei(vi_vp_pos[0], -1, 0, display_2[0]); print_valuei(vi_vp_pos[0], "vi_vp_pos[0]"); break;
			case 'w': vi_vp_pos[0] = modify_valuei(vi_vp_pos[0], 1, 0, display_2[0]); print_valuei(vi_vp_pos[0], "vi_vp_pos[0]"); break;
			case 'e': vi_vp_pos[1] = modify_valuei(vi_vp_pos[1], -1, 0, display_2[0]); print_valuei(vi_vp_pos[1], "vi_vp_pos[1]"); break;
			case 'r': vi_vp_pos[1] = modify_valuei(vi_vp_pos[1], 1, 0, display_2[0]); print_valuei(vi_vp_pos[1], "vi_vp_pos[1]"); break;
			case 'a': vi_vp_size[0] = modify_valuei(vi_vp_size[0], -1, 0, display_2[0]); vi_vp_size[1] = (float(om_vp_size[1])/om_vp_size[0])*vi_vp_size[0]; print_valuei(vi_vp_size[0], "vi_vp_size[0]"); print_valuei(vi_vp_size[1], "vi_vp_size[1]"); break;
			case 's': vi_vp_size[0] = modify_valuei(vi_vp_size[0], 1, 0, display_2[0]); vi_vp_size[1] = (float(om_vp_size[1])/om_vp_size[0])*vi_vp_size[0]; print_valuei(vi_vp_size[0], "vi_vp_size[0]"); print_valuei(vi_vp_size[1], "vi_vp_size[1]"); break;
			default: printf("Entered key does nothing \n");
			}

		}
	}

	//  uncomment this if not using an idle func
	//	glutPostRedisplay();
}


// ------------------------------------------------------------
//
// Mouse Events
//

void processMouseButtons(int button, int state, int xx, int yy)
{
	// start tracking the mouse
	if (state == GLUT_DOWN) {
		startX = xx;
		startY = yy;
		if (button == GLUT_LEFT_BUTTON)
			tracking = 1;
		else if (button == GLUT_RIGHT_BUTTON)
			tracking = 2;
	}

	//stop tracking the mouse
	else if (state == GLUT_UP) {
		if (tracking == 1) {
			alpha += (startX - xx);
			beta += (yy - startY);
		}
		else if (tracking == 2) {
			r += (yy - startY) * 0.01f;
		}
		tracking = 0;
	}
}

// Track mouse motion while buttons are pressed
void processMouseMotion(int xx, int yy)
{

	int deltaX, deltaY;
	float alphaAux, betaAux;
	float rAux;

	deltaX = startX - xx;
	deltaY = yy - startY;

	// left mouse button: move camera
	if (tracking == 1) {
		alphaAux = alpha + deltaX;
		betaAux = beta + deltaY;

		if (betaAux > 85.0f)
			betaAux = 85.0f;
		else if (betaAux < -85.0f)
			betaAux = -85.0f;
		rAux = r;

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}
	// right mouse button: zoom
	else if (tracking == 2) {

		alphaAux = alpha;
		betaAux = beta;
		rAux = r + (deltaY * 0.01f);

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}
	//  uncomment this if not using an idle func
	//	glutPostRedisplay();
}

void mouseWheel(int wheel, int direction, int x, int y) {
	r += direction * 0.1f;
	updateCamVariables();
}

// --------------------------------------------------------
//
// Shader Stuff
//

void printShaderInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}


void printProgramInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}

GLuint setupOmShader() {
	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(fname_blurmap_vertex_shader);
	fs = textFileRead(fname_om_fragment_shader);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(f);
	printShaderInfoLog(v);
	
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindAttribLocation(p, om_vertexLoc, "position");
	glBindAttribLocation(p, om_textureLoc, "texCoord");
	glBindFragDataLocation(p, 0, "FragColor");

	glLinkProgram(p);
	glValidateProgram(p);
	printProgramInfoLog(p);

	om_program = p;
	om_vertexShader = v;
	om_fragmentShader = f;

	om_rgb_img = glGetUniformLocation(om_program, "rgb_img");
	om_blur_map = glGetUniformLocation(om_program, "blur_map");
	om_depth_map = glGetUniformLocation(om_program, "depth_map");
	om_zFar_uniformLocation = glGetUniformLocation(om_program, "zFar");

	return(p);
}

GLuint setupBlurShader() {
	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(fname_blurmap_vertex_shader);
	fs = textFileRead(fname_blur_fragment_shader);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(f);
	printShaderInfoLog(v);
	
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindAttribLocation(p, blur_vertexLoc, "position");
	glBindAttribLocation(p, blur_textureLoc, "texCoord");
	glBindFragDataLocation(p, 0, "FragColor");

	glLinkProgram(p);
	glValidateProgram(p);
	printProgramInfoLog(p);

	blur_program = p;
	blur_vertexShader = v;
	blur_fragmentShader = f;

	blur_rgb_img = glGetUniformLocation(blur_program, "rgb_img");
	blur_blur_map = glGetUniformLocation(blur_program, "blur_map");
	blur_depth_map = glGetUniformLocation(blur_program, "depth_map");
	blur_zFar_uniformLocation = glGetUniformLocation(blur_program, "zFar");

	return(p);
}

GLuint setupBlurmapShader() {
	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(fname_blurmap_vertex_shader);
	fs = textFileRead(fname_blurmap_fragment_shader);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(f);
	printShaderInfoLog(v);
	
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindAttribLocation(p, blurmap_vertexLoc, "position");
	glBindAttribLocation(p, blurmap_textureLoc, "texCoord");
	glBindFragDataLocation(p, 0, "FragColor");

	glLinkProgram(p);
	glValidateProgram(p);
	printProgramInfoLog(p);

	blurmap_program = p;
	blurmap_vertexShader = v;
	blurmap_fragmentShader = f;

	blurmap_rgb_img = glGetUniformLocation(blurmap_program, "rgb_img");
	blurmap_depth_map = glGetUniformLocation(blurmap_program, "depth_map");
	blurmap_zNear_uniformLocation = glGetUniformLocation(blurmap_program, "zNear");
	blurmap_zFar_uniformLocation = glGetUniformLocation(blurmap_program, "zFar");
	blurmap_linear_near_uniformLocation = glGetUniformLocation(blurmap_program, "linear_near");
	blurmap_linear_far_uniformLocation = glGetUniformLocation(blurmap_program, "linear_far");
	blurmap_focal_depth_uniformLocation = glGetUniformLocation(blurmap_program, "focal_depth");

	return(p);
}

GLuint setupShader() {

	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(fname_vertex_shader);
	fs = textFileRead(fname_fragment_shader_rgb);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(v);
	printShaderInfoLog(f);

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindFragDataLocation(p, 0, "FragColor");

	glBindAttribLocation(p, vertexLoc, "position");
	glBindAttribLocation(p, normalLoc, "normal");
	glBindAttribLocation(p, texCoordLoc, "texCoord");

	glLinkProgram(p);
	glValidateProgram(p);
	printProgramInfoLog(p);

	program = p;
	vertexShader = v;
	fragmentShader = f;

	GLuint k = glGetUniformBlockIndex(p, "Matrices");
	glUniformBlockBinding(p, k, matricesUniLoc);
	glUniformBlockBinding(p, glGetUniformBlockIndex(p, "Material"), materialUniLoc);

	texUnit = glGetUniformLocation(p, "texUnit");

	return(p);
}

// ------------------------------------------------------------
//
// Model loading and OpenGL setup
//

const GLfloat light_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[] = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

void loadTexture(const char* lpszPathName, GLuint tex) {
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	fif = FreeImage_GetFileType(lpszPathName, 0);
	if (fif == FIF_UNKNOWN) {
		fif = FreeImage_GetFIFFromFilename(lpszPathName);
	}

	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		FIBITMAP *image = FreeImage_Load(fif, lpszPathName, 0);
		if (image != NULL) {
			//convert to 32-bpp so things will be properly aligned 
			FIBITMAP* temp = image;
			image = FreeImage_ConvertTo32Bits(image);
			FreeImage_Unload(temp);


			glBindTexture(GL_TEXTURE_2D, tex);
			glPixelStorei(GL_UNPACK_ROW_LENGTH, FreeImage_GetPitch(image) / 4);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FreeImage_GetWidth(image), FreeImage_GetHeight(image), 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(image));
			FreeImage_Unload(image);
		}
		else {
			printf("error reading image '%s', exiting...\n", lpszPathName);
			exit(1);
		}
	}
	else {
		printf("missing/unknown/unsupported image '%s', exiting...\n", lpszPathName);
		exit(1);
	}

}

int init()
{
	/* initialization of DevIL */
	ilInit();

	for (int modelIter = 0; modelIter < NUM_MODELS; modelIter++) {
		model[modelIter].basepath = file_path_and_name[modelIter][0];
		model[modelIter].modelname = file_path_and_name[modelIter][1];
		if (!Import3DFromFile(model[modelIter]))
			return(0);
		LoadGLTextures(model[modelIter]);
	}

	glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)glutGetProcAddress("glGetUniformBlockIndex");
	glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)glutGetProcAddress("glUniformBlockBinding");
	glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glutGetProcAddress("glGenVertexArrays");
	glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glutGetProcAddress("glBindVertexArray");
	glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)glutGetProcAddress("glBindBufferRange");
	glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glutGetProcAddress("glDeleteVertexArrays");

	program = setupShader();

	for (int modelIter = 0; modelIter < NUM_MODELS; modelIter++) {
		genVAOsAndUniformBuffer(model[modelIter]);
	}

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//char fileName[1024] = "background.png";
	char fileName[1024] = "white1.png";
	glGenTextures(1, &tex_background);
	glBindTexture(GL_TEXTURE_2D, tex_background);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	loadTexture(fileName, tex_background);

	//
	// Uniform Block
	//
	glGenBuffers(1, &matricesUniBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferData(GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, matricesUniLoc, matricesUniBuffer, 0, MatricesUniBufferSize);	//setUniforms();
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenTextures(1, &tex_rgb);
	glBindTexture(GL_TEXTURE_2D, tex_rgb);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &tex_depth);
	glBindTexture(GL_TEXTURE_2D, tex_depth);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	printf("Works so far \n");
	//create fbos/renderbuffers
	glGenRenderbuffers(1, &rbo_depth_image);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_image);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1920, 1080);

	glGenFramebuffers(1, &fbo_rgbd);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo_rgbd);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth_image);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_depth, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_rgb, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Error in creating framebuffer \n");
	}

	blurmap_program = setupBlurmapShader();
	genVAOs_blurmap();

	glGenTextures(1, &tex_blurmap_rgb);
	glBindTexture(GL_TEXTURE_2D, tex_blurmap_rgb);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenFramebuffers(1, &fbo_blurmap_rgb);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo_blurmap_rgb);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_blurmap_rgb, 0);

	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Error in creating framebuffer \n");
	}

	blur_program = setupBlurShader();
	genVAOs_blur();

	om_program = setupOmShader();
	genVAOs_om();

	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//glEnable(GL_LIGHT0);
	//glEnable(GL_NORMALIZE);
	//glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_LIGHTING);

	//glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	//glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	//glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	//glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	//glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	//glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	//glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	//glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glEnable(GL_MULTISAMPLE);

	return true;
}


// ------------------------------------------------------------
//
// Main function
//
int main(int argc, char **argv) {

	//int success = initLenses();
	//if (!success) {
	//	printf("Encounted error in initLenses() \n");
	//	return(0);
	//}
	//  GLUT initialization
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

	//glutInitContextVersion(3, 3);
	//glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);

	if (display_on_device) {
		//printf("%d %d \n", window_position[0], window_position[1]);
		//printf("%d %d \n", window_size[0], window_size[1]);
		//glutInitWindowPosition(window_position[0], window_position[1]);
		//glutInitWindowSize(window_size[0], window_size[1]);

		//window_position[0] = 0; // position in x
		//window_position[1] = 0; // position in y
		//window_size[0] = 2*1920;
		//window_size[1] = 1080;
		printf("%d %d \n", window_position[0], window_position[1]);
		printf("%d %d \n", window_size[0], window_size[1]);
		glutInitWindowPosition(window_position[0], window_position[1]);
		glutInitWindowSize(window_size[0], window_size[1]);
	}
	else {
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(1920, 1080);
	}
	glutCreateWindow("Varifocal Occlusion");


	//  Callback Registration
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	//	Mouse and Keyboard Callbacks
	glutKeyboardFunc(processKeys);
	glutMouseFunc(processMouseButtons);
	glutMotionFunc(processMouseMotion);

	glutMouseWheelFunc(mouseWheel);

	//	Init GLEW
	//glewExperimental = GL_TRUE;
	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 not supported\n");
		return(1);
	}

	//  Init the app (load model and textures) and OpenGL
	if (!init())
		printf("Could not Load the Model\n");

	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));


	// return from main loop
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	usePosition();
	updateCamVariables();
	//  GLUT main loop
	glutMainLoop();

	// delete buffers
	glDeleteBuffers(1, &matricesUniBuffer);
	glDeleteRenderbuffers(1, &rbo_depth_image);
	glDeleteFramebuffers(1, &fbo_rgbd);
	return(0);
}

