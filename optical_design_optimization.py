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

common_f2_f3 = 0.0

def call_populate_focal_lengths(f, IOD):
    if(IOD.num_lenses == 2):
        IOD.populate_focal_lengths((f[0], common_f2_f3, common_f2_f3, f[1]))
    elif(IOD.num_lenses == 3):
        IOD.populate_focal_lengths((f[0], f[1], f[1], f[2]))

''' Propagate using Lensmaker's equations '''
def energy_function(f, IOD): 
    rw_dists_l = [1000.0, 250.0, 64.0, 16.0]

    call_populate_focal_lengths(f, IOD)
    
    # Error associated with distance and magnitude of a bunch of real world distances
    err_mag_l = []
    err_dist_l = []
    for rw_dist in rw_dists_l:
        IOD.propagate_rw_all(rw_dist)
        curr_err_dist = cf.convert_cm2dpt(rw_dist) + cf.convert_cm2dpt(IOD.final_I)
        curr_err_mag = IOD.rw_magnification - 1.0
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    # Error associated with distance at which the occlusion mask is seen
    IOD.propagate_om()
    curr_err_om = cf.convert_cm2dpt(IOD.final_I) + cf.convert_cm2dpt(IOD.d_vip_eye)
    energy_om = curr_err_om**2

    combined_energy = energy_dist + energy_mag + 100*energy_om
    return combined_energy

def using_differential_evolution():
    global common_f2_f3

    show_fl_in_diopters = False

    IOD = OD.optical_design()
    IOD.num_lenses = 2
    IOD.include_all = True
    dists_l = [1000.0, 250.0, 64.0, 16.0]

    common_f2_f3 = 3.5
    IOD.d_f2_f3 = (2*IOD.d_f1_f2*common_f2_f3)/(IOD.d_f1_f2 - common_f2_f3)
    print('d_f2_f2: %7.2f' % (IOD.d_f2_f3))

    myBounds = []
    if(IOD.num_lenses == 2):
        myBounds = [(2.0, 4.0), (0.0, 800.0)]
    elif(IOD.num_lenses == 3):
        myBounds = [(0.0, 4.0), (0.0, 5.0), (-50.0, 800.0)]
    elif(IOD.num_lenses == 4):
        myBounds = [(0.0, 4.0), (0.0, 10.0), (0.0, 10.0), (-50.0, 50.0)]
    energy_function_args = [IOD]

    print('     OM      RW        f1      f2      f3      f4       ~RW    ~MAG     ~OM    ~RW0')
    # print('1000.00 1000.00 |    3.98    4.00    4.00    4.02 |   -0.00    0.00   -0.00   -0.00')
    for vip_dist in dists_l:
        IOD.d_vip_eye = vip_dist

        # IOD.populate_d_eye(curr_dist)
        # IOD.d_W_f1 = IOD.d_vip_eye

        curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
        res = curr_res
        min_energy = curr_res.fun

        for trials in range(0, 30):
            curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
            if(curr_res.fun < min_energy):
                min_energy = curr_res.fun
                res = curr_res

        call_populate_focal_lengths(res.x, IOD)
        for rw_dist in dists_l:
            IOD.propagate_rw_all(rw_dist)
            curr_err_dist = cf.convert_cm2dpt(rw_dist) + cf.convert_cm2dpt(IOD.final_I)
            curr_err_mag = IOD.rw_magnification - 1.0
            IOD.propagate_rw_all(vip_dist)
            curr_err_infocus_rw = cf.convert_cm2dpt(IOD.I1) - cf.convert_cm2dpt(IOD.d_f1_LCoS)
            IOD.propagate_om()
            curr_err_om = cf.convert_cm2dpt(IOD.final_I) + cf.convert_cm2dpt(IOD.d_vip_eye)


            if(show_fl_in_diopters == True):
                print('%7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f'% (vip_dist, rw_dist, cf.convert_cm2dpt(IOD.f1), cf.convert_cm2dpt(IOD.f2), cf.convert_cm2dpt(IOD.f3), cf.convert_cm2dpt(IOD.f4), curr_err_dist, curr_err_mag, curr_err_om, curr_err_infocus_rw))
            else:
                print('%7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f'% (vip_dist, rw_dist, IOD.f1, IOD.f2, IOD.f3, IOD.f4, curr_err_dist, curr_err_mag, curr_err_om, curr_err_infocus_rw))

def optotune_focal_length_ranges():
    TC_bounds = [5, 12]
    C_bounds = [10, 20]
    TC_bounds_diopters = [cf.convert_cm2dpt(elem) for elem in TC_bounds]
    C_bounds_diopters = [cf.convert_cm2dpt(elem) for elem in C_bounds]
    print(TC_bounds)
    print(TC_bounds_diopters)
    print(C_bounds)
    print(C_bounds_diopters)

    offset_lens = -15
    nC_bounds = [cf.convert_dpt2cm(cf.convert_cm2dpt(offset_lens) + cf.convert_cm2dpt(elem)) for elem in C_bounds]
    print(nC_bounds)

    res_bounds = [-66.6, 28.6]
    nC_bounds = [cf.convert_dpt2cm(cf.convert_cm2dpt(elem) - cf.convert_cm2dpt(offset_lens)) for elem in C_bounds]
    print(nC_bounds)

if __name__ == '__main__':
    # using_differential_evolution()
    optotune_focal_length_ranges()
