#include "lens_controls.h"
#include <string>
#include <stdio.h>

float f1_orig[] = { 16.2, 13.1 };
float f4_orig[] = { 5.2, 7.2 };

float f1_copy[] = { 16.2, 13.1 };
float f4_copy[] = { 5.2, 7.2 };

int focal_length_index = 0;

CPyInstance pInstance;
CPyObject pDict;
CPyObject python_class;
CPyObject lens1, lens4;
CPyObject sysPath;
CPyObject modulePath;
CPyObject pModule;
CPyObject value;

void update_lens(bool f1 = true, bool f4 = true) {
	char str[1024] = "";
	if (f1) {
		value = PyObject_CallMethod(lens1, "set_diopter", "(f)", f1_copy[focal_length_index]);
		sprintf(str, "%s L1[%d]:%0.2f", str, focal_length_index, f1_copy[focal_length_index]);
	}
	if (f4) {
		value = PyObject_CallMethod(lens4, "set_diopter", "(f)", f4_copy[focal_length_index]);
		sprintf(str, "%s L4[%d]:%0.2f", str, focal_length_index, f1_copy[focal_length_index]);
	}
	printf("%s\n", str);
}

void set_fl_absolute_middle() {

	float f1_middle = (7.0 + 27.0) / 2.0;
	float f4_middle = (4.0 + 11.0) / 2.0;
	int num_focal_lengths = sizeof(f1_orig) / sizeof(f1_orig[0]);
	for (int iter = 0; iter < num_focal_lengths; iter++) {
		f1_copy[iter] = f1_middle;
		f4_copy[iter] = f4_middle;
	}
	update_lens(true, true);
}

void set_fl_middle() {
	int num_focal_lengths = sizeof(f1_orig) / sizeof(f1_orig[0]);
	float f1_middle = (f1_copy[0] + f1_copy[num_focal_lengths - 1]) / 2.0;
	float f4_middle = (f4_copy[0] + f4_copy[num_focal_lengths - 1]) / 2.0;
	for (int iter = 0; iter < num_focal_lengths; iter++) {
		f1_copy[iter] = f1_middle;
		f4_copy[iter] = f4_middle;
	}
	update_lens(true, true);
}

void reset_orig_fl() {
	int num_focal_lengths = sizeof(f1_orig) / sizeof(f1_orig[0]);
	for (int iter = 0; iter < num_focal_lengths; iter++) {
		f1_copy[iter] = f1_orig[iter];
		f4_copy[iter] = f4_orig[iter];
	}
	update_lens(true, true);
}

void modify_current_fl(int lens_num, float delta) {
	bool f1 = false;
	bool f4 = false;
	if (lens_num == 1) {
		f1_copy[focal_length_index] = f1_copy[focal_length_index] + delta;
		if (f1_copy[focal_length_index] > 27.0) {
			f1_copy[focal_length_index] = 27.0;
		}
		if (f1_copy[focal_length_index] < 7.0) {
			f1_copy[focal_length_index] = 7.0;
		}
		f1 = true;
	}
	if (lens_num == 4) {
		f4_copy[focal_length_index] = f4_copy[focal_length_index] + delta;
		if (f4_copy[focal_length_index] > 11.0) {
			f4_copy[focal_length_index] = 11.0;
		}
		if (f4_copy[focal_length_index] < 4.0) {
			f4_copy[focal_length_index] = 4.0;
		}
		f4 = true;
	}
	update_lens(f1, f4);
}

void decrement_index(bool f1 = true, bool f4 = true) {
	focal_length_index--;
	if (focal_length_index < 0) {
		focal_length_index = 0;
	}
	update_lens(f1,f4);
}

void increment_index(bool f1 = true, bool f4 = true) {
	focal_length_index++;
	if (focal_length_index > (sizeof(f1_copy)/sizeof(f1_copy[0])) - 1) {
		focal_length_index = (sizeof(f1_copy)/sizeof(f1_copy[0])) - 1;
	}
	update_lens(f1,f4);

}


int initLenses() {
	sysPath = PySys_GetObject((char*)"path");
	modulePath = PyUnicode_FromString("E:/kishoreWorkspace/Displays/Occlusion/voss/L3DAssimpModelImportDemo/AssimpModelImportDemo/");
	PyList_Append(sysPath, modulePath);
	CPyObject pName = PyUnicode_FromString("lens");
	pModule = PyImport_Import(pName);

	if(!pModule) {
		printf_s("Error: Module not imported \n");
		return(0);
	}

	pDict = PyModule_GetDict(pModule);

	python_class = PyDict_GetItemString(pDict, "Lens");
	if(!(python_class && PyCallable_Check(python_class))) {
		printf_s("Error: Failed to get class \n");
		return(0);		
	}

	CPyObject arglist;
	CPyObject return_value;
	CPyObject repr;
	CPyObject str;
	CPyObject value;
	const char *bytes;

	// lens1
	arglist = Py_BuildValue("(s)", "COM9");
	lens1 = PyObject_CallObject(python_class, arglist);
	if(!lens1) {
		printf_s("Error: Failed to create object \n");
		return(0);			
	}
	return_value = PyObject_GetAttrString(lens1, "firmware_version");
	repr = PyObject_Repr(return_value);
	str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
	bytes = PyBytes_AsString(str);
	printf("%s\n", bytes);
	value = PyObject_CallMethod(lens1, "to_focal_power_mode", nullptr);

	// lens4
	arglist = Py_BuildValue("(s)", "COM8");
	lens4 = PyObject_CallObject(python_class, arglist);
	if(!lens4) {
		printf_s("Error: Failed to create object \n");
		return(0);			
	}
	return_value = PyObject_GetAttrString(lens4, "firmware_version");
	repr = PyObject_Repr(return_value);
	str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
	bytes = PyBytes_AsString(str);
	printf("%s\n", bytes);
	value = PyObject_CallMethod(lens4, "to_focal_power_mode", nullptr);
}
