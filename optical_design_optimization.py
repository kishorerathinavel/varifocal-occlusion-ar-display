import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf
import OD
from pprint import pprint


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
        curr_err_dist = cf.convert_cm2dpt(rw_dist) + cf.convert_cm2dpt(IOD.lens_l[-1].d_image)
        curr_err_mag = IOD.magnification - 1.0
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    # Error associated with distance at which the occlusion mask is seen
    IOD.propagate_om()
    curr_err_om = cf.convert_cm2dpt(IOD.lens_l[-1].d_image) + cf.convert_cm2dpt(IOD.d_vip_eye)
    energy_om = curr_err_om**2

    combined_energy = energy_dist + energy_mag + 100*energy_om
    return combined_energy

def initialize_IOD(IOD):
    common_f2_f3 = 3.5

    # Lens 1
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)
    
    # Lens 2
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    curr_lens.d_prev_lens = 2*IOD.min_d_f_LCoS
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    # Lens 3
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    d_f2_f3 = (2*IOD.lens_l[1].d_prev_lens*common_f2_f3)/(IOD.lens_l[1].d_prev_lens - common_f2_f3)
    print('d_f2_f2: %7.2f' % (d_f2_f3))
    curr_lens.d_prev_lens = d_f2_f3
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    # Lens 4
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = 2*IOD.min_d_f_LCoS
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)

    IOD.num_lenses = len(IOD.lens_l)
    IOD.num_lenses_om = IOD.num_lenses - 1

def using_differential_evolution():
    show_fl_in_diopters = True
    IOD = OD.optical_design()
    initialize_IOD(IOD)

    dists_l = [1000.0, 250.0, 64.0, 16.0]

    '''  Focal power range for optotune lenses
         |             | min(cm) | max(cm) | min(D) | max(D) | comments  |
         | EL-10-30-TC |       5 |      12 |   8.33 |     20 | thinner   |
         | EL-10-30-C  |      10 |      20 |      5 |     10 | wo offset |
    '''
    myBounds = []
    if(IOD.num_lenses == 2):
        myBounds = [(2.0, 4.0), (0.0, 800.0)]
        # myBounds = [(10.0, 20.0), (10.0, 20.0)] # Assuming EL-10-30-C
        # myBounds = [(5.0, 12.0), (5.0, 12.0)] # Assuming EL-10-30-TC
    elif(IOD.num_lenses == 3):
        myBounds = [(0.0, 4.0), (0.0, 5.0), (-50.0, 800.0)]
    elif(IOD.num_lenses == 4):
        myBounds = [(0.0, 4.0), (0.0, 10.0), (0.0, 10.0), (-50.0, 50.0)]
    energy_function_args = [IOD]

    for lens in IOD.lens_l:
        pprint(vars(lens))
    pprint(vars(IOD))

    print('     OM      RW        f1      f2      f3      f4       ~RW    ~MAG     ~OM    ~RW0')
    if(show_fl_in_diopters == True):
        print('   (cm)    (cm)     (dpt)   (dpt)   (dpt)   (dpt)')
    else:
        print('   (cm)    (cm)      (cm)    (cm)    (cm)    (cm)')
    # print('1000.00 1000.00 |    3.98    4.00    4.00    4.02 |   -0.00    0.00   -0.00   -0.00')

    for vip_dist in dists_l:
        IOD.d_vip_eye = vip_dist

        curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
        res = curr_res
        min_energy = curr_res.fun

        for trials in range(0, 30):
            curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
            if(curr_res.fun < min_energy):
                min_energy = curr_res.fun
                res = curr_res

        IOD.populate_focal_lengths(res.x)
        for rw_dist in dists_l:
            IOD.propagate_rw_all(rw_dist)
            curr_err_dist = cf.convert_cm2dpt(rw_dist) + cf.convert_cm2dpt(IOD.lens_l[-1].d_image)
            curr_err_mag = IOD.magnification - 1.0
            IOD.propagate_rw_all(vip_dist)
            curr_err_infocus_rw = cf.convert_cm2dpt(IOD.lens_l[0].d_image) - cf.convert_cm2dpt(IOD.min_d_f_LCoS)
            IOD.propagate_om()
            curr_err_om = cf.convert_cm2dpt(IOD.lens_l[-1].d_image) + cf.convert_cm2dpt(IOD.d_vip_eye)

            f1 = IOD.lens_l[0].focal_length
            f2 = IOD.lens_l[1].focal_length
            f3 = IOD.lens_l[2].focal_length
            f4 = IOD.lens_l[3].focal_length
            if(show_fl_in_diopters == True):
                print('%7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f'% (vip_dist, rw_dist, cf.convert_cm2dpt(f1), cf.convert_cm2dpt(f2), cf.convert_cm2dpt(f3), cf.convert_cm2dpt(f4), curr_err_dist, curr_err_mag, curr_err_om, curr_err_infocus_rw))
            else:
                print('%7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f'% (vip_dist, rw_dist, f1, f2, f3, f4, curr_err_dist, curr_err_mag, curr_err_om, curr_err_infocus_rw))

if __name__ == '__main__':
    using_differential_evolution()
