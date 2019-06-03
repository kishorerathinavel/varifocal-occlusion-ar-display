import pprint as pp
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf
import OD
from sympy import *

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

min_rw_dist = 30.0
max_rw_dist = 300.0

dists_l = []

show_fl_in_diopters = True

def calc_err_dist(dist, IOD):
    curr_err_dist = 0.0
    if(show_fl_in_diopters == True):
        curr_err_dist = cf.convert_cm2dpt(dist + IOD.length) + cf.convert_cm2dpt(IOD.lens_l[-1].d_image)
    else:
        curr_err_dist = dist + IOD.length + IOD.lens_l[-1].d_image

    return curr_err_dist

def calc_err_om(IOD):
    curr_err_om = 0.0
    if(show_fl_in_diopters == True):
        curr_err_om = cf.convert_cm2dpt(IOD.lens_l[-1].d_image) + cf.convert_cm2dpt(IOD.d_vip_eye + IOD.length)
    else:
        curr_err_om = IOD.lens_l[-1].d_image + IOD.d_vip_eye + IOD.length
    return curr_err_om
 
def energy_function_abcd(f, IOD):
    IOD.populate_focal_lengths(f)
    IOD.calc_ABCD_matrices()
    IOD.calc_TA()
    II = Matrix([[1,0], [0,1]])
    IOD.TT = II
    IOD.calc_TA_diff_TT()
    IOD.calc_OO_norm()

    energy_see_through = IOD.norm
    
    IOD.propagate_om()
    curr_err_om = calc_err_om(IOD)
    energy_om = curr_err_om**2

    combined_energy = energy_see_through + 100*energy_om
    return combined_energy
    
def energy_function_custom(f, IOD): 
    IOD.populate_focal_lengths(f)
    
    # Error associated with distance and magnitude of a bunch of real world distances
    err_mag_l = []
    err_dist_l = []
    for rw_dist in dists_l:
        IOD.propagate_rw_all(rw_dist)
        curr_err_dist = calc_err_dist(rw_dist, IOD)
        curr_err_mag = IOD.magnification - IOD.target_magnification
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)
    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))
    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)

    # Error associated with distance at which the occlusion mask is seen
    IOD.propagate_om()
    curr_err_om = calc_err_om(IOD)
    energy_om = curr_err_om**2

    combined_energy = energy_mag + energy_om
    return combined_energy

# This was the version of the function during ISMAR submission.
def bk_initialize_IOD(IOD):
    IOD.target_magnification = 1.0
    common_f2_f3 = 3.5
    half_width_35mm_lens = 2.00
    # half_width_35mm_lens = 1.00
    # d_f1_f2 = 2*IOD.min_d_f_LCoS + half_width_35mm_lens
    d_f1_f2 = 13
    # d_f3_f4 = d_f1_F2 # Has to be symmetrical
    d_f3_f4 = d_f1_f2
    d_f2_f3 = (2*d_f1_f2*common_f2_f3)/(d_f1_f2 - common_f2_f3)
    # min_d_f2_f3 = 10.7 - 2*half_width_35mm_lens
    # if(d_f2_f3 < min_d_f2_f3):
    #     print('d_f2_f3 can not be smaller than %f' %(min_d_f2_f3))
    #     d_f2_f3 = min_d_f2_f3
    # d_f2_f3 = 7.0


    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)
    
    # curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(20.0)
    # curr_lens.d_prev_lens = 1.0
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    curr_lens.d_prev_lens = d_f1_f2
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    curr_lens.d_prev_lens = d_f2_f3
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = d_f3_f4
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(-3.0)
    # curr_lens.d_prev_lens = 2.0
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    IOD.length = 0
    IOD.num_lenses = 0
    IOD.num_lenses_om = 3
    IOD.num_tunable_lenses = 0
    for curr_lens in IOD.lens_l:
        IOD.num_lenses = IOD.num_lenses + 1
        # IOD.length = IOD.length + curr_lens.d_prev_lens
        if(curr_lens.tunable == True):
            IOD.num_tunable_lenses = IOD.num_tunable_lenses + 1


    # ACTUAL VALUES
    # common_f2_f3 = 3.5

    # curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(-5.0)
    # curr_lens.d_prev_lens = 0.0
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = -1
    # curr_lens.d_prev_lens = 1.6
    # curr_lens.tunable = True
    # IOD.lens_l.append(curr_lens)
    
    # curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(20.0)
    # curr_lens.d_prev_lens = 2.0
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = common_f2_f3
    # curr_lens.d_prev_lens = 4+3+5
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = common_f2_f3
    # curr_lens.d_prev_lens = 14.5
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(-2)
    # curr_lens.d_prev_lens = 5.2
    # curr_lens.tunable = False
    # IOD.lens_l.append(curr_lens)

    # curr_lens = OD.lens()
    # curr_lens.focal_length = -1
    # curr_lens.d_prev_lens = 3
    # curr_lens.tunable = True
    # IOD.lens_l.append(curr_lens)

def initialize_IOD(IOD):
    IOD.target_magnification = 1.0
    common_f2_f3 = 3.5
    # L1:
    curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(-5.0)
    curr_lens.focal_length = cf.convert_dpt2cm(-5.0)
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    # L2:
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    # curr_lens.d_prev_lens = 1.6
    curr_lens.d_prev_lens = 1.5
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)
    # L3:
    curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(20.0)
    # curr_lens.d_prev_lens = 2.0
    curr_lens.focal_length = cf.convert_dpt2cm(20.0)
    curr_lens.d_prev_lens = 2.0 - 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    # L4:
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    # curr_lens.d_prev_lens = 3+3+IOD.cube_f2
    curr_lens.d_prev_lens = 3+3+IOD.cube_f2 # DO NOT CHANGE
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    # L5:
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f2_f3
    # curr_lens.d_prev_lens = 14.5
    curr_lens.d_prev_lens = 14.5 - 2.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    # L6:
    curr_lens = OD.lens()
    # curr_lens.focal_length = cf.convert_dpt2cm(-2)
    # curr_lens.d_prev_lens = 5.2
    curr_lens.focal_length = cf.convert_dpt2cm(+4)
    curr_lens.d_prev_lens = 6.5
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    # L7:
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    # curr_lens.d_prev_lens = 3
    curr_lens.d_prev_lens = 3.0
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)

    IOD.length = 0
    IOD.num_lenses = 0
    IOD.num_lenses_om = 4
    IOD.num_tunable_lenses = 0
    for curr_lens in IOD.lens_l:
        IOD.num_lenses = IOD.num_lenses + 1
        # IOD.length = IOD.length + curr_lens.d_prev_lens
        if(curr_lens.tunable == True):
            IOD.num_tunable_lenses = IOD.num_tunable_lenses + 1
            
def initialize_stage1(IOD):
    IOD.target_magnification = -1.0

    # Lens 1
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)
    
    # Lens 2
    curr_lens = OD.lens()
    curr_lens.focal_length = -1
    curr_lens.d_prev_lens = 8.0
    curr_lens.tunable = True
    IOD.lens_l.append(curr_lens)

    IOD.length = 0
    IOD.num_lenses = 2
    IOD.num_lenses_om = 1
    IOD.num_tunable_lenses = 0
    for curr_lens in IOD.lens_l:
        # IOD.length = IOD.length + curr_lens.d_prev_lens
        if(curr_lens.tunable == True):
            IOD.num_tunable_lenses = IOD.num_tunable_lenses + 1

def initialize_rw_distances():
    global dists_l

    dists_l_D = []
    human_depth_resolution_D = 0.3
    max_rw_dist_D = cf.convert_cm2dpt(min_rw_dist)
    min_rw_dist_D = cf.convert_cm2dpt(max_rw_dist)
    curr_dist_D = max_rw_dist_D
    dists_l_D.append(curr_dist_D)
    dists_l.append(cf.convert_dpt2cm(curr_dist_D))
    while 1:
        curr_dist_D = curr_dist_D - human_depth_resolution_D
        if(curr_dist_D < min_rw_dist_D):
            break

        dists_l_D.append(curr_dist_D)
        dists_l.append(cf.convert_dpt2cm(curr_dist_D))

    # curr_dist_D = 0.3
    # dists_l_D.append(curr_dist_D)
    # dists_l.append(cf.convert_dpt2cm(curr_dist_D))

    # print("dists_l", end = " ")
    # print(dists_l)

    # print("dists_l_D", end = " ")
    # print(dists_l_D)

    # print(len(dists_l))

def using_differential_evolution():
    IOD = OD.optical_design()
    energy_function = energy_function_custom
    initialize_IOD(IOD)
    # initialize_stage1(IOD)
    initialize_rw_distances()
    
    '''  Focal power range for optotune lenses
         |             | min(cm) | max(cm) | min(D) | max(D) | comments  |
         | EL-10-30-TC |       5 |      12 |   8.33 |     20 | thinner   |
         | EL-10-30-C  |      10 |      20 |      5 |     10 | wo offset |
    '''
    # myBounds = []
    if(IOD.num_tunable_lenses == 2):
        # myBounds = [(cf.convert_dpt2cm(10.0), cf.convert_dpt2cm(5.0)), (cf.convert_dpt2cm(10.0), cf.convert_dpt2cm(5.0))]
        myBounds = [(-90.0, 90.0), (-90.0, 90.0)]
        # myBounds = [(10.0, 20.0), (10.0, 20.0)] # Assuming EL-10-30-C
        # myBounds = [(5.0, 12.0), (5.0, 12.0)] # Assuming EL-10-30-TC
    elif(IOD.num_tunable_lenses == 3):
        myBounds = [(0.0, 4.0), (0.0, 5.0), (-50.0, 800.0)]
    elif(IOD.num_tunable_lenses == 4):
        myBounds = [(0.0, 4.0), (0.0, 10.0), (0.0, 10.0), (-50.0, 50.0)]
    energy_function_args = [IOD]

    # ppo = pp.PrettyPrinter(indent=4)
    # for lens in IOD.lens_l:
    #     print("Lens:", end = " ")
    #     ppo.pprint(vars(lens))
    # print("IOD:", end = " ")
    # ppo.pprint(vars(IOD))

    if(show_fl_in_diopters == True):
        str1 = '     OM      RW  '
        str2 = '  (dpt)    (dpt) '
        for idx, curr_lens in enumerate(IOD.lens_l):
            str1 = '%s      f%d' %(str1, idx)
            str2 = '%s   (dpt)' %(str2)
        str1 = '%s       ~RW     MAG     ~OM  ' % (str1)
        str2 = '%s     (dpt)   (dpt)   (dpt)' %(str2)
        print(str1)
        print(str2)
        # print('     OM      RW        f1      f2      f3      f4       ~RW    ~MAG     ~OM  ')
        # print('   (cm)    (cm)     (dpt)   (dpt)   (dpt)   (dpt)     (dpt)           (dpt)')
    else:
        str1 = '     OM      RW  '
        str2 = '   (cm)     (cm) '
        for idx, curr_lens in enumerate(IOD.lens_l):
            str1 = '%s      f%d' %(str1, idx)
            str2 = '%s    (cm)' %(str2)
        str1 = '%s       ~RW     MAG     ~OM  ' % (str1)
        str2 = '%s      (cm)    (cm)    (cm)' %(str2)
        print(str1)
        print(str2)
        # print('     OM      RW        f1      f2      f3      f4       ~RW    ~MAG     ~OM  ')
        # print('   (cm)    (cm)      (cm)    (cm)    (cm)    (cm)      (cm)            (cm)')
    # print('1000.00 1000.00 |    3.98    4.00    4.00    4.02 |   -0.00    0.00   -0.00   -0.00')

    f1_l = []
    f4_l = []
    for vip_dist in dists_l:
        # print('###########################\n\n')
        IOD.d_vip_eye = vip_dist

        curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
        res = curr_res
        min_energy = curr_res.fun

        for trials in range(0, 10):
            curr_res = differential_evolution(energy_function, bounds=myBounds, args=energy_function_args)
            if(curr_res.fun < min_energy):
                min_energy = curr_res.fun
                res = curr_res

        IOD.populate_focal_lengths(res.x)
        IOD.propagate_rw_all(vip_dist)

        # print('============')
        # print('------------')
        # for idx, lens in enumerate(IOD.lens_l):
        #     print('Lens %d:' % (idx))
        #     ppo.pprint(vars(lens))
        # print('------------')
        # ppo.pprint(vars(IOD))
        # print('------------')
        # print('============')

        IOD.propagate_om()
        # print('============')
        # print('------------')
        # for idx, lens in enumerate(IOD.lens_l):
        #     print('Lens %d:' % (idx))
        #     ppo.pprint(vars(lens))
        # print('------------')
        # ppo.pprint(vars(IOD))
        # print('------------')
        # print('============')

        for rw_dist in dists_l:
            IOD.propagate_rw_all(rw_dist)
            curr_err_dist = calc_err_dist(rw_dist, IOD)

            # print('%7.7f %7.7f %7.7f %7.7f' %(IOD.lens_l[0].magnification,
            #                                   IOD.lens_l[1].magnification,
            #                                   IOD.lens_l[2].magnification,
            #                                   IOD.lens_l[3].magnification))
            # print('%7.7f %7.7f %7.7f %7.7f' %(IOD.lens_l[0].magnification,
            #                                   IOD.lens_l[0].magnification*IOD.lens_l[1].magnification,
            #                                   IOD.lens_l[0].magnification*IOD.lens_l[1].magnification*IOD.lens_l[2].magnification,
            #                                   IOD.lens_l[0].magnification*IOD.lens_l[1].magnification*IOD.lens_l[2].magnification*IOD.lens_l[3].magnification))
            curr_err_mag = IOD.magnification
            
            IOD.propagate_om()
            curr_err_om = calc_err_om(IOD)

            if(show_fl_in_diopters == True):
                str = '%7.2f %7.2f |'% (cf.convert_cm2dpt(vip_dist), cf.convert_cm2dpt(rw_dist))
                for curr_lens in IOD.lens_l:
                    str = '%s %7.2f' %(str, cf.convert_cm2dpt(curr_lens.focal_length))
                str = '%s | %7.2f %7.2f %7.2f' %(str, curr_err_dist, curr_err_mag, curr_err_om)
                print(str)
            else:
                str = '%7.2f %7.2f |'% (vip_dist, rw_dist)
                for curr_lens in IOD.lens_l:
                    str = '%s %7.2f' %(str, curr_lens.focal_length)
                str = '%s | %7.2f %7.2f %7.2f' %(str, curr_err_dist, curr_err_mag, curr_err_om)
                print(str)

            # if(show_fl_in_diopters == True):
            #     f1_l.append(cf.convert_cm2dpt(IOD.lens_l[1].focal_length))
            #     f4_l.append(cf.convert_cm2dpt(IOD.lens_l[-2].focal_length))
            # else:
            #     f1_l.append(IOD.lens_l[1].focal_length)
            #     f4_l.append(IOD.lens_l[-2].focal_length)

        IOD.calc_ABCD_matrices()
        IOD.calc_TA()
        II = Matrix([[1,0], [0,1]])
        IOD.TT = II
        IOD.calc_TA_diff_TT()
        IOD.calc_OO_norm()
        print("------------------------------------------------------------------")
        # print(IOD.TT)
        # print("IOD.TA:", end = " ")
        # print(IOD.TA)
        # print("IOD.OO:", end = " ")
        # print(IOD.OO)
        # print('Norm: %7.2f' %(IOD.norm))

    # for idx, lens in enumerate(IOD.lens_l):
    #     print("lens.d_prev_lens:", end = " ")
    #     print(lens.d_prev_lens)
    
    # print('Mean of f1: %f' % (np.mean(f1_l)))
    # print('Offset lens for f1: %f Diopters' % (np.mean(f1_l) - 7.5))

    # print('Mean of f4:' % (np.mean(f4_l)))
    # print('Offset lens for f4: %f Diopters' % (np.mean(f4_l) - 7.5))


if __name__ == '__main__':
    using_differential_evolution()
