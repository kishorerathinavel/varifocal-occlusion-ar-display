import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf
import OD

'''
Need to minimize:
1. Error in magnification of real world objects
2. Error in depth difference for real world objects
3. Error in distance of occlusion mask

Arguements to optimize:
f1, f2, f3, f4

Outpus of energy function:
norm of the vector of errors mentioned above
'''

''' Propagate using Lensmaker's equations '''
def energy_function(f, IOD): 
    rw_dists_l = [1000.0, 250.0, 64.0, 16.0]
    IOD.populate_focal_lengths(f)

    # Error associated with distance and magnitude of a bunch of real world distances
    err_mag_l = []
    err_dist_l = []
    for rw_dist in rw_dists_l:
        IOD.propagate_rw_all(rw_dist)
        curr_err_dist = rw_dist + IOD.final_I
        curr_err_mag = IOD.rw_magnification - 1.0
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    # Error associated with distanec of real world that is brought into focus at LCoS
    IOD.propagate_rw_all(IOD.d_W_f1)
    curr_err_infocus_rw = IOD.I1 - IOD.d_f1_LCoS
    energy_infocus_rw = curr_err_infocus_rw**2
    
    # Error associated with distance at which the occlusion mask is seen
    IOD.propagate_om()
    curr_err_om = IOD.final_I + IOD.d_vip_eye
    energy_om = curr_err_om**2

    # combined_energy = energy_dist + energy_mag + 100*energy_om
    combined_energy = energy_dist + energy_mag + 100*energy_om + 100*energy_infocus_rw
    # combined_energy = 100*energy_infocus_rw
    return combined_energy # change to more meaningful outputs

def using_differential_evolution():
    IOD = OD.optical_design()
    IOD.num_lenses = 2
    IOD.include_all = True
    dists_l = [1000.0, 250.0, 64.0, 16.0]
    # dists_l = [16.0]

    myBounds = []
    if(IOD.num_lenses == 2):
        myBounds = [(2.0, 4.0), (0.0, 800.0)]
    elif(IOD.num_lenses == 3):
        myBounds = [(0.0, 4.0), (0.0, 5.0), (-50.0, 800.0)]
    elif(IOD.num_lenses == 4):
        # myBounds = [(0.0, 4.0), (-10.0, 0.0), (0.0, 10.0), (0.0, 10.0)]
        myBounds = [(0.0, 4.0), (0.0, 10.0), (0.0, 10.0), (-50.0, 50.0)]
    energy_function_args = [IOD]

    print('     OM      RW        f1      f2      f3      f4       ~RW    ~MAG     ~OM    ~RW0')
    # print('1000.00 1000.00 |    3.98    4.00    4.00    4.02 |   -0.00    0.00   -0.00   -0.00')
    for vip_dist in dists_l:
        IOD.d_vip_eye = vip_dist

        # IOD.populate_d_eye(curr_dist)
        IOD.d_W_f1 = IOD.d_vip_eye

        curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
        res = curr_res
        min_energy = curr_res.fun

        for trials in range(0, 20):
            curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
            if(curr_res.fun < min_energy):
                min_energy = curr_res.fun
                res = curr_res

        IOD.populate_focal_lengths(res.x)
        for rw_dist in dists_l:
            IOD.propagate_rw_all(rw_dist)
            curr_err_dist = rw_dist + IOD.final_I
            curr_err_mag = IOD.rw_magnification - 1.0
            curr_err_infocus_rw = IOD.I1 - IOD.d_f1_LCoS
            IOD.propagate_om()
            curr_err_om = IOD.final_I + IOD.d_vip_eye

            print('%7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f'% (vip_dist, rw_dist, IOD.f1, IOD.f2, IOD.f3, IOD.f4, curr_err_dist, curr_err_mag, curr_err_om, curr_err_infocus_rw))

if __name__ == '__main__':
    using_differential_evolution()
