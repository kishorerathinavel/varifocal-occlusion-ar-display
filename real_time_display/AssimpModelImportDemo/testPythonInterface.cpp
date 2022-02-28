#include "testPythonInterface.h"

void testPythonInterface1() {
	PyObject* pInt;

	Py_Initialize();

	PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
	
	Py_Finalize();

	printf("\nPress any key to exit...\n");
	if(!_getch()) _getch();
}

void testPythonInterface2() {
	char filename[] = "testPythonInterface2.py";
	FILE* fp;

	Py_Initialize();

	fp = _Py_fopen(filename, "r");
	PyRun_SimpleFile(fp, filename);

	Py_Finalize();
}

void testPythonInterface3() {
    CPyInstance pyInstance;

	PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
	

	printf("\nPress any key to exit...\n");
	if(!_getch()) _getch();
}

void testPythonInterface4() {
	CPyInstance pInstance;

	CPyObject p;
	p = PyLong_FromLong(50);
	printf_s("Value = %1d\n", PyLong_AsLong(p));
}

void testPythonInterface5() {
	CPyInstance pInstance;

	CPyObject pName = PyUnicode_FromString("testPythonInterface5");
	CPyObject pModule = PyImport_Import(pName);

	if(pModule) {
		CPyObject pFunc = PyObject_GetAttrString(pModule, "getInteger");
		if(pFunc && PyCallable_Check(pFunc)) {
			CPyObject pValue = PyObject_CallObject(pFunc, NULL);
			printf_s("C: getInteger() = %1d\n", PyLong_AsLong(pValue));
		}
		else {
			printf("Error: function getInteger() \n");
		}
	}
	else {
		printf_s("Error: Module not imported \n");
	}
}

int testPythonInterface5_2() {
	CPyInstance pInstance;

	CPyObject pName = PyUnicode_FromString("testPythonInterface5");
	CPyObject pModule = PyImport_Import(pName);

	if(!pModule) {
		printf_s("Error: Module not imported \n");
		return(0);
	}

	CPyObject pFunc = PyObject_GetAttrString(pModule, "getInteger");
	if(!(pFunc && PyCallable_Check(pFunc))) {
		printf("Error: function getInteger() \n");
		return(0);
	}

	CPyObject pValue = PyObject_CallObject(pFunc, NULL);
	printf_s("C: getInteger() = %1d\n", PyLong_AsLong(pValue));
}

int testPythonInterface6() {
	CPyInstance pInstance;

	CPyObject pName = PyUnicode_FromString("testPythonInterface6");
	CPyObject pModule = PyImport_Import(pName);

	if(!pModule) {
		printf_s("Error: Module not imported \n");
		return(0);
	}

	CPyObject pDict = PyModule_GetDict(pModule);
	if(!pDict) {
		printf_s("Error: Failed to get dictionary \n");
		return(0);	
	}

	CPyObject python_class = PyDict_GetItemString(pDict, "Adder");
	if(!(python_class && PyCallable_Check(python_class))) {
		printf_s("Error: Failed to get class \n");
		return(0);		
	}

	CPyObject arglist = Py_BuildValue("(i)", 20);
	if(!arglist) {
		printf_s("Error: Failed to build arglist \n");
		return(0);				
	}

	CPyObject python_object = PyObject_CallObject(python_class, arglist);
	if(!python_object) {
		printf_s("Error: Failed to create object \n");
		return(0);			
	}

	CPyObject value = PyObject_CallMethod(python_object, "test1", "(i)", 5);
	if(!value) {
		printf_s("Error: Failed to call object's function \n");
		return(0);				
	}
	printf_s("C: value = %d\n", PyLong_AsLong(value));
}

int testPythonInterfaceLens() {
	CPyInstance pInstance;

	CPyObject sysPath = PySys_GetObject((char*)"path");
	CPyObject modulePath = PyUnicode_FromString("E:/kishoreWorkspace/Displays/Occlusion/voss/L3DAssimpModelImportDemo/AssimpModelImportDemo/");
	PyList_Append(sysPath, modulePath);
	CPyObject pName = PyUnicode_FromString("lens");
	CPyObject pModule = PyImport_Import(pName);

	if(!pModule) {
		printf_s("Error: Module not imported \n");
		return(0);
	}

	CPyObject pDict = PyModule_GetDict(pModule);
	if(!pDict) {
		printf_s("Error: Failed to get dictionary \n");
		return(0);	
	}

	CPyObject python_class = PyDict_GetItemString(pDict, "Lens");
	if(!(python_class && PyCallable_Check(python_class))) {
		printf_s("Error: Failed to get class \n");
		return(0);		
	}

	CPyObject arglist = Py_BuildValue("(s)", "COM8");
	if(!arglist) {
		printf_s("Error: Failed to build arglist \n");
		return(0);				
	}

	CPyObject python_object = PyObject_CallObject(python_class, arglist);
	if(!python_object) {
		printf_s("Error: Failed to create object \n");
		return(0);			
	}

	CPyObject python_object_variable = PyObject_GetAttrString(python_object, "firmware_version");
	if(!python_object_variable) {
		printf_s("Error: Failed to return value from object member \n");
		return(0);				
	}
	CPyObject repr = PyObject_Repr(python_object_variable);
	CPyObject str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
	const char *bytes = PyBytes_AsString(str);
	printf("%s\n", bytes);

	CPyObject value = PyObject_CallMethod(python_object, "to_focal_power_mode", nullptr);
	if(!value) {
		printf_s("Error: Failed to return value from object function \n");
		return(0);					
	}

	value = PyObject_CallMethod(python_object, "set_diopter", "(f)", 5.2);
}

