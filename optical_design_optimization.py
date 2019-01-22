import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf

'''
Need to minimize:
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

def propagate(f, d_f1_f2, dist):
    O1 = dist
    I1 = cf.calculate_image_distance(O1, f[0])
    m1 = -I1/O1
    O2 = d_f1_f2 - I1
    I2 = cf.calculate_image_distance(O2, f[1])
    m2 = -I2/O2
    mT = m1*m2
    return (I1, I2, mT)

''' Propagate using Lensmaker's equations '''
def energy_function(f, d_f1_f2, d_f1_LCoS, vip_dist): 
    rw_dists_l = [1000.0, 250.0, 64.0, 16.0]
    err_mag_l = []
    err_dist_l = []
    for rw_dist in rw_dists_l:
        [I1, I2, mT] = propagate(f, d_f1_f2, rw_dist)
        curr_err_dist = rw_dist + I2
        curr_err_mag = mT - 1.0
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    [I1, I2, mT] = propagate(f, d_f1_f2, vip_dist)
    curr_err_om = I1 - d_f1_LCoS
    energy_om = curr_err_om**2

    combined_energy = energy_dist + energy_mag + energy_om
    return combined_energy # change to more meaningful outputs

def using_differential_evolution():
    for trial in range(0, 10):
        d_f1_f2 = 0.0
        d_f1_LCoS = 10.0
        vip_dists_l = [1000.0, 250.0, 64.0, 16.0]
        for vip_dist in vip_dists_l:
            res = differential_evolution(energy_function, bounds=[(-10.0, 10.0), (-10.0, 10.0)], args=[d_f1_f2, d_f1_LCoS, vip_dist])
            for rw_dist in vip_dists_l:
                [I1, I2, mT] = propagate(res.x, d_f1_f2, rw_dist)
                curr_err_dist = rw_dist + I2
                curr_err_mag = mT - 1.0
                [I1, I2, mT] = propagate(res.x, d_f1_f2, vip_dist)
                curr_err_om = I1 - d_f1_LCoS
                print('f: %6.2f %6.2f | err: %6.2f %6.2f %6.2f'% (res.x[0], res.x[1], curr_err_dist, curr_err_mag, curr_err_om))


if __name__ == '__main__':
    using_differential_evolution()

