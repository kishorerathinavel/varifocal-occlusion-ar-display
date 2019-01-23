from sympy import *
import numpy as np

def calc_perceptually_useful_distances(max_dist, diop_diff, num_dist):
    # Assuming that min_dist is specified in cm
    max_diop_dist = cf.convert_cm2dpt(max_dist)
    prev_diop_dist = max_diop_dist
    dists_l = []
    dists_l.append(max_dist)
    for iter in range(1,num_dist):
        next_diop_dist = prev_diop_dist + diop_diff
        next_dist = cf.convert_dpt2cm(next_diop_dist)
        dists_l.append(next_dist)
        prev_diop_dist = next_diop_dist
    return dists_l

def conv_lol_flat_l(my_input, output_list):
    if isinstance(my_input, list):
        for element in my_input:
            conv_lol_flat_l(element, output_list)
    elif isinstance(my_input, Tuple):
        for element in my_input:
            conv_lol_flat_l(element, output_list)
    else:
        return output_list.append(my_input)

def convert_sympy_mutableDenseMatrix_custom_prntableList(denseMatrix):
    denseMatrix_np = np.array(denseMatrix.tolist()).astype(np.float64)
    r_denseMatrix_np = np.round(denseMatrix_np, 2)
    denseMatrix_l = r_denseMatrix_np.tolist()
    return denseMatrix_l

def convert_dpt2cm(value_dpt):
    return convert_m2cm(1/value_dpt)

def convert_m2cm(value_m):
    return 100*value_m

def calculate_image_distance(d_O, f):
    d_I = 0
    if(d_O == f):
        d_I = np.sign(d_O)*np.sign(f)*100000 # large number approximating infinity
    else:
        d_I = (d_O*f)/(d_O - f)
    return d_I

def calculate_focal_length(d_O, d_I):
    return (d_O * d_I)/(d_I + d_O)

def convert_cm2m(value_cm):
    return value_cm/100.0

def convert_cm2dpt(value_cm):
    return 1/(convert_cm2m(value_cm))

def makeLensMatrix(f):
    mat = Matrix([[1,0],[-1/f,1]])
    return(mat)

def makeFreeSpacePropagationMatrix(d):
    mat = Matrix([[1,d], [0,1]])
    return mat

def print_matrix(mat, tex='True'):
    print(tex)
    if(tex == 'True'):
        print(latex(mat[0,0], order = 'ilex'))
        print(latex(mat[0,1], order = 'ilex'))
        print(latex(mat[1,0], order = 'ilex'))
        print(latex(mat[1,1], order = 'ilex'))
    else:
        print(mat[0,0])
        print(mat[0,1])
        print(mat[1,0])
        print(mat[1,1])
 
