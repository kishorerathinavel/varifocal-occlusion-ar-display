from sympy import *

def convert_dpt2cm(value_dpt):
    return convert_m2cm(1/value_dpt)

def convert_m2cm(value_m):
    return 100*value_m

def calculate_image_distance(d_O, f):
    return (d_O*f)/(d_O - f)

def calculate_focal_length(d_O, d_I):
    return (d_O * d_I)/(d_I + d_O)

def convert_cm2m(value_cm):
    return value_cm/100.0

def convert_cm2dpt(value_cm):
    return 1/(convert_cm2m(value_cm))

def makeLensMatrix(f):
    mat = Matrix([[1,0],[-f,1]])
    return(mat)

def makeFreeSpacePropagationMatrix(d):
    mat = Matrix([[1,d], [0,1]])
    return mat

def print_matrix(mat):
    print(latex(mat[0,0], order = 'ilex'))
    print(latex(mat[0,1], order = 'ilex'))
    print(latex(mat[1,0], order = 'ilex'))
    print(latex(mat[1,1], order = 'ilex'))
 
