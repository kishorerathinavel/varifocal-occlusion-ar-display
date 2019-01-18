import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf

print_first_iter = True

def custom_print(parameters_l):
    if(print_first_iter == True):
        for parameter in parameters_l:
            print(parameter)

def energy_function(f, d_f1_f2):
    '''
    Propagate using Lensmaker's equations
    '''

    global print_first_iter
    
    dists_l = [1000.0, 250.0, 64.0, 16.0]
    err_mag_l = []
    err_dist_l = []
    for dist in dists_l:
        O1 = dist
        I1 = cf.calculate_image_distance(O1, f[0])
        m1 = -I1/O1
        O2 = d_f1_f2 - I1
        I2 = cf.calculate_image_distance(O2, f[1])
        m2 = -I2/O2
        mT = m1*m2
        curr_err_dist = I2 - dist
        curr_err_mag = mT - 1.0
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
        custom_print([['O1', O1], ['I1', I1], ['O2', O2], ['I2', I2], ['m1', m1], ['m2', m2], ['mT', mT], ['curr_err_dist', curr_err_dist], ['curr_err_mag', curr_err_mag], '\n'])

    print_first_iter = False
    
    err_dist_arr = np.array(err_dist_l)
    err_mag_arr = np.array(err_mag_l)
    
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    combined_energy = energy_dist + energy_mag
    # print(combined_energy)
    return combined_energy

def attempt_1():
    '''
    We need to minimize:
    1. Error in magnification of real world objects
    2. Error in depth difference for real world objects

    Assume:
    fixed d_f1_f2 = 0

    Arguements to optimize:
    f1, f2
    
    Inputs to energy function:
    f1, f2, d_f1_f2
    
    Outpus of energy function:
    norm of the vector of errors mentioned above
    '''
    f0 = [5.0, -15.0]
    d_f1_f2 = 0.0
    res = minimize(energy_function, f0, args=(d_f1_f2), method='nelder-mead', options={'xtol':1e-5, 'disp':True})
    print('-----------')
    print(res)
    print('-----------')

if __name__ == '__main__':
    attempt_1()

