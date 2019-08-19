'''
Refer to abcd.svg for a diagram of the optical designs modelled here
Description of each main function:
+ tunable_all_symmetric: Refer to abcd.svg/png
  1. f1 = f4
  2. f2 = f3
  3. d_f1_f2 = d_f3_f4
+ tunable_f1_f2_f3: Refer to abcd.svg/png
  1. f4 is fixed-focal-length
  2. d_f1_f2 = d_f3_f4
+ tunable_f1_f2: Refer to abcd.svg/png
  + f3 and f4 are fixed focal length
+ howlett: Howlett-Smithwick-SID2017
+ howlett_1D: Single optical axis version of Howlett-Smithwick-SID2017
'''

import common_functions as cf
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import OD

prnt_flag = 'False'
# prnt_flag = 'True'
outputs_dir = 'outputs'

class outputs():
    pass

def custom_prnt(str):
    if(prnt_flag == 'True'):
        print(str)

def graph_outputs(op, d_f2_f3_l, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l):
    num_d_f2_f3 = len(d_f2_f3_l)
    num_soln = len(soln_l)
    num_dists_l = len(dists_l)
    iter_ylabel = 0

    # for key, value in op.__dict__.items():
    #     dimensions = len(value.shape)
    #     if(dimensions == 1):
    #         print("D1: " + key)
    #     elif(dimensions == 2):
    #         print("D2: " + key)
    #     elif(dimensions == 3):
    #         print("D3: " + key)
    # return

    for key, value in op.__dict__.items():
        dimensions = len(value.shape)
        if(dimensions == 1):
            str = "./%s/D%d_%s.png" % (outputs_dir, dimensions, key)
            plt.clf()
            plt.plot(dists_l, value)
            plt.ylabel(ylabels_l[iter_ylabel])
            plt.xlabel("distance")
            curr_ylim = ylim_l[iter_ylabel]
            if(curr_ylim[0] != curr_ylim[1]):
                plt.ylim(curr_ylim[0], curr_ylim[1])
            plt.savefig(str)
        elif(dimensions == 2):
            for iter_d_f2_f3 in range(0, num_d_f2_f3):
                str = "./%s/D%d_%d_%s.png" % (outputs_dir, dimensions, iter_d_f2_f3, key)
                plt.clf()
                plt.plot(dists_l, value[iter_d_f2_f3,:])
                plt.ylabel(ylabels_l[iter_ylabel])
                plt.xlabel("distance")
                curr_ylim = ylim_l[iter_ylabel]
                if(curr_ylim[0] != curr_ylim[1]):
                    plt.ylim(curr_ylim[0], curr_ylim[1])
                plt.savefig(str)
        elif(dimensions == 3):
            for iter_d_f2_f3 in range(0, num_d_f2_f3):
                for iter_soln in range(0, num_soln):
                    str = "./%s/D%d_%d_%d_%s.png" % (outputs_dir, dimensions, iter_d_f2_f3, iter_soln, key)
                    plt.clf()
                    plt.plot(dists_l, value[iter_d_f2_f3, :, iter_soln])
                    plt.ylabel(ylabels_l[iter_ylabel])
                    plt.xlabel("distance")
                    curr_ylim = ylim_l[iter_ylabel]
                    if(curr_ylim[0] != curr_ylim[1]):
                        plt.ylim(curr_ylim[0], curr_ylim[1])
                    plt.savefig(str)
        elif(dimensions == 4):
            for iter_d_f2_f3 in range(0, num_d_f2_f3):
                for iter_dist in range(0, num_dists_l):
                    for iter_soln in range(0, num_soln):
                        str = "./%s/D%d_%d_%d_%d_%s.png" % (outputs_dir, dimensions, iter_d_f2_f3, iter_dist, iter_soln, key)
                        plt.clf()
                        plt.plot(dists_l, value[iter_d_f2_f3, iter_dist, iter_soln, :])
                        plt.ylabel(ylabels_l[iter_ylabel])
                        plt.xlabel("distance")
                        curr_ylim = ylim_l[iter_ylabel]
                        if(curr_ylim[0] != curr_ylim[1]):
                            plt.ylim(curr_ylim[0], curr_ylim[1])
                        plt.savefig(str)
        iter_ylabel = iter_ylabel + 1

'''
f4 = 5
d12 = d34
'''
def tunable_f1_f2_f3():
    IOD = OD.optical_design()
    op = outputs()
    # diop_diff = 0.6
    # max_dist = cf.convert_m2cm(10)
    # num_dist = 3
    # dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)
    dists_l = [1000, 250, 64, 16]
    num_dist = len(dists_l)

    d_f2_f3_l = [3, 8, 13, 15, 16, 17, 23]
    # d_f2_f3_l = list(range(3, 28, 5))
    # d_f2_f3_l.append(16)
    # d_f2_f3_l.sort()
    # d_f2_f3_l = list(range(3, 18, 5))
    num_d_f2_f3 = len(d_f2_f3_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = num_soln

    std_output_arr = np.zeros((num_d_f2_f3, num_dist))
    op.f1_arr = np.copy(std_output_arr)
    op.d_W_f1_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln))
    op.f2_arr = np.copy(std_output_arr)
    op.f3_arr = np.copy(std_output_arr)
    op.norm_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln, num_dist))
    op.mag_arr = np.copy(std_output_arr)
    op.rw_dist = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)
    op.diff_dist = np.copy(std_output_arr)

    ylabels_l = ["f1", "d_W_f1", "f2", "f3", "norm", "d_OM_f4", "mag_arr", "rw_dist", "img_dist", "diff_dist"]
    ylim_l = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [-1,-1], [-1, -1], [-1, -1]]
    output_arrays_resized = 'False'

    for IOD.d_f2_f3 in d_f2_f3_l:
        d_f2_f3_index = d_f2_f3_l.index(IOD.d_f2_f3)
        # str = "d_f2_f3 = %0.2f ========================" % (IOD.d_f2_f3)
        # print(str)
        for curr_dist in dists_l:
            # str = "d_vip_eye = %0.2f ========================" % (curr_dist)
            # print(str)
            dist_index = dists_l.index(curr_dist)
            IOD.d_vip_eye = curr_dist # Should be > 23
            str = "d_vip_eye = %f" % curr_dist
            custom_prnt(str)

            # IOD.populate_d_eye(curr_dist)
            IOD.d_W_f1 = IOD.d_vip_eye
            IOD.f1 = cf.calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
            op.f1_arr[d_f2_f3_index,dist_index] = IOD.f1
            sym_f2 = Symbol('f_2')
            IOD.f2 = sym_f2
            sym_f3 = Symbol('f_3')
            IOD.f3 = sym_f3
            IOD.f4 = 5
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()

            II = Matrix([[1,0], [0,1]])
            IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
            S14 = cf.makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
            TT = II
            # TT = S14
            IOD.TT = II

            IOD.calc_TA_diff_TT()
            OO = IOD.OO

            OO_l = OO.tolist()
            flat_OO_l = []
            cf.conv_lol_flat_l(OO_l, flat_OO_l)

            soln_l = list(nonlinsolve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f2, sym_f3]))
            custom_prnt("Solutions")
            custom_prnt(soln_l)
            custom_prnt("\n")
            # END Getting solutions for all equations together

            # Get the norm of OO = TT - TA for each solution 
            norm_l = []
            for curr_soln in soln_l:
                IOD.f2 = curr_soln[0]
                IOD.f3 = curr_soln[1]
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()
                norm_l.append(IOD.norm)
            rounded_norm_l = [round(elem,2) for elem in norm_l]
            # custom_prnt(type(rounded_norm_l))
            custom_prnt("Norm of solutions")
            custom_prnt(rounded_norm_l)
            # END Get the norm of OO = TT - TA for each solution

            # Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices
            if(output_arrays_resized == 'False'):
                num_soln = len(soln_l)
                if(prev_num_soln < num_soln):
                    num_new_colns = num_soln - prev_num_soln
                    new_cols = np.zeros((num_d_f2_f3, num_dist, num_new_colns))
                    op.f2_arr = np.dstack((op.f2_arr, new_cols))
                    op.f3_arr = np.dstack((op.f2_arr, new_cols))
                    op.norm_arr = np.dstack((op.norm_arr, new_cols))
                    op.d_OM_f4_arr = np.dstack((op.d_OM_f4_arr, new_cols))

                    new_cols = np.zeros((num_d_f2_f3, num_dist, num_new_colns, num_dist))
                    op.mag_arr = np.dstack((op.mag_arr, new_cols))
                    op.rw_dist = np.dstack((op.rw_dist, new_cols))
                    op.img_dist = np.dstack((op.img_dist, new_cols))
                    op.diff_dist = np.dstack((op.diff_dist, new_cols))
                output_arrays_resized = 'True'
            # END Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices

            for curr_soln in soln_l:
                # str = "curr_soln_index = %d ========================" % (soln_l.index(curr_soln))
                # print(str)
                soln_index = soln_l.index(curr_soln)
                IOD.f2 = curr_soln[0]
                IOD.f3 = curr_soln[1]
                op.f2_arr[d_f2_f3_index,dist_index, soln_index] = IOD.f2
                op.f3_arr[d_f2_f3_index,dist_index, soln_index] = IOD.f3

                custom_prnt("\n")
                str = "f1 = %f cm" % (IOD.f1)
                custom_prnt(str)
                str = "f1 = %f D" % (cf.convert_cm2dpt(IOD.f1))
                custom_prnt(str)
                str = "f2 = %f cm" % (IOD.f2)
                custom_prnt(str)
                str = "f2 = %f D" % (cf.convert_cm2dpt(IOD.f2))
                custom_prnt(str)
                str = "f3 = %f cm" % (IOD.f3)
                custom_prnt(str)
                str = "f3 = %f D" % (cf.convert_cm2dpt(IOD.f3))
                custom_prnt(str)

                # Verify that TA ~= TT
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()

                TA_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
                str = "Actual Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TA_l)

                TT_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
                str = "Target Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TT_l)

                OO_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
                str = "residual matrix"
                custom_prnt(str)
                custom_prnt(OO_l)
                rounded_norm = round(IOD.norm, 2)
                op.norm_arr[d_f2_f3_index,dist_index, soln_index] = rounded_norm
                str = "norm = %f" % (rounded_norm)
                custom_prnt(str)
                # END Verify that TT ~= TA

                # Calculate where image of real world is formed
                # Calculate d_WI_f4
                # IOD.populate_d_eye(IOD.d_vip_eye)
                IOD.d_W_f1 = IOD.d_vip_eye
                IOD.propagate_rw_all(IOD.d_W_f1)
                # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
                str = "I1 = %f" % (IOD.I1)
                custom_prnt(str)
                str = "d_f1_LCoS = %f" % (IOD.d_f1_LCoS)
                custom_prnt(str)
                # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
                str = "d_WI_f4 = %f" % (IOD.d_WI_f4)
                custom_prnt(str)
                # END Calculate where image of real world is formed

                # Calculate where image of occlusion mask
                if (abs(IOD.I1 - IOD.d_f1_LCoS) < 0.01):
                    custom_prnt("No need to propagate OM separately because rw at vip formed at LCoS")
                    IOD.d_OM_f4 = IOD.d_WI_f4
                else:
                    custom_prnt("RW at vip did not form at LCoS")
                    # str = "d_vip_eye = %f; d_W_f1 = %f" % (IOD.d_vip_eye, IOD.d_W_f1)
                    # custom_prnt(str)
                    # str = "saveI1 = %f; saveO1 = %f; savef1 = %f" % (saveI1, saveO1, savef1)
                    # custom_prnt(str)
                    # str = "I1 = %f; O1 = %f; f1 = %f" % (IOD.I1, IOD.O1, IOD.f1)
                    # custom_prnt(str)
                    IOD.propagate_om()
                # END Calculate where image of occlusion mask
                str = "d_OM_f4 = %f" %(IOD.d_OM_f4)
                custom_prnt(str)
                str = "d_W_f1 = %f" % (-IOD.d_W_f1)
                custom_prnt(str)

                op.d_OM_f4_arr[d_f2_f3_index,dist_index, soln_index] = IOD.d_OM_f4

                str = "Magnification at all distances:"
                custom_prnt(str)
                for ncurr_dist in dists_l:
                    ncurr_dist_index = dists_l.index(ncurr_dist)

                    # IOD.populate_d_eye(ncurr_dist)
                    IOD.d_W_f1 = ncurr_dist

                    IOD.propagate_rw_all(IOD.d_W_f1) # Assume that ncurr_dist = d_W_f1

                    diff_dist = IOD.d_WI_f4 + IOD.d_W_f1
                    op.rw_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = IOD.d_W_f1
                    op.img_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = -IOD.d_WI_f4
                    op.diff_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = diff_dist

                    op.mag_arr[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification

    # Print relevant info from op
    for d_f2_f3 in d_f2_f3_l:
        iter1 = d_f2_f3_l.index(d_f2_f3)
        str = "d_f2_f3 = %0.2f" % (d_f2_f3)
        print(str)
        for dist in dists_l:
            iter2 = dists_l.index(dist)
            # str = "    dist = %0.2f" % (dist)
            # print(str)
            mag_mean_l = []
            mag_std_l = []
            diff_mean_l = []
            diff_std_l = []
            for iter3 in range(0, num_soln):
                # str = "soln_number = %0.2f ========================" % (iter3)
                # print(str)
                mag_l = op.mag_arr[iter1, iter2, iter3, :]
                diff_l = op.diff_dist[iter1, iter2, iter3, :]

                mag_mean_l.append(np.mean(mag_l))
                mag_std_l.append(np.std(mag_l))
                diff_mean_l.append(np.mean(diff_l))
                diff_std_l.append(np.std(diff_l))
                # str = 'Avg(mag): %0.2f  Std(mag): %0.2f  Avg(dif): %0.2f  Std(dif): %0.2f' % (np.mean(mag_l), np.std(mag_l), np.mean(diff_l), np.std(diff_l))
                # print(str)

            mag_one = np.array([1]*len(mag_mean_l))
            mag_diff_from_1 = np.array(mag_mean_l) - mag_one
            # print(mag_diff_from_1)
            mag_diff_from_1 = np.multiply(mag_diff_from_1, mag_diff_from_1)
            # print(mag_diff_from_1)
            min_index = np.argmin(mag_diff_from_1)

            print('    vip_dist = %7.2f | avg(mag): %0.2f | std(mag): %0.2f | avg(dif): %6.2f | std(dif): %0.2f' % (dist, mag_mean_l[min_index], mag_std_l[min_index], diff_mean_l[min_index], diff_std_l[min_index]))
        print('\n')

    # Collect the average difference of the better solution
    # graph_outputs(op, d_f2_f3_l, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l)
    # end of tunable_f1_f2_f3

'''
f1 = f4
f2 = f3
d12 = d34
'''
def tunable_all_symmetric():
    IOD = OD.optical_design()
    op = outputs()
    # diop_diff = 0.6
    # max_dist = cf.convert_m2cm(10)
    # num_dist = 3
    # dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)
    dists_l = [1000, 250, 64, 16]
    num_dist = len(dists_l)

    d_f2_f3_l = [3, 8, 13, 15, 16, 17, 23]
    # d_f2_f3_l = list(range(3, 28, 5))
    # d_f2_f3_l.append(15)
    # d_f2_f3_l.append(16)
    # d_f2_f3_l.append(17)
    # d_f2_f3_l.sort()
    # d_f2_f3_l = list(range(3, 18, 5))
    num_d_f2_f3 = len(d_f2_f3_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = num_soln

    std_output_arr = np.zeros((num_d_f2_f3, num_dist))
    op.f1_arr = np.copy(std_output_arr)
    op.d_W_f1_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln))
    op.f2_arr = np.copy(std_output_arr)
    op.norm_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln, num_dist))
    op.mag_arr = np.copy(std_output_arr)
    op.rw_dist = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)
    op.diff_dist = np.copy(std_output_arr)

    ylabels_l = ["f1", "d_W_f1", "f2", "norm", "d_OM_f4", "mag_arr", "rw_dist", "img_dist", "diff_dist"]
    ylim_l = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [-1,-1], [-1, -1], [-1, -1]]
    output_arrays_resized = 'False'

    for IOD.d_f2_f3 in d_f2_f3_l:
        d_f2_f3_index = d_f2_f3_l.index(IOD.d_f2_f3)
        # str = "d_f2_f3 = %0.2f ========================" % (IOD.d_f2_f3)
        # print(str)
        for curr_dist in dists_l:
            # str = "d_vip_eye = %0.2f ========================" % (curr_dist)
            # print(str)
            dist_index = dists_l.index(curr_dist)
            IOD.d_vip_eye = curr_dist # Should be > 23
            str = "d_vip_eye = %f" % curr_dist
            custom_prnt(str)

            # IOD.populate_d_eye(curr_dist)
            IOD.d_W_f1 = IOD.d_vip_eye
            IOD.f1 = cf.calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
            op.f1_arr[d_f2_f3_index,dist_index] = IOD.f1
            sym_f2 = Symbol('f_2')
            IOD.f2 = sym_f2
            IOD.prototype_v4_populate_dependent_focalLengths()
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()

            II = Matrix([[1,0], [0,1]])
            IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
            S14 = cf.makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
            TT = II
            # TT = S14
            IOD.TT = II

            IOD.calc_TA_diff_TT()
            OO = IOD.OO

            # custom_prnt_matrix(OO)
            OO_l = OO.tolist()
            flat_OO_l = []
            cf.conv_lol_flat_l(OO_l, flat_OO_l)

            # Getting solutions for all equations together
            soln_l = list(nonlinsolve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f2]))
            # END Getting solutions for all equations together

            # Converting from sympy set,tuple to python list
            soln_l2 = []
            cf.conv_lol_flat_l(soln_l, soln_l2)
            # END Converting from sympy set to python list

            # Extracting unique solutions
            rounded_soln_l = [round(elem,2) for elem in soln_l2]
            unique_soln_l = list(set(rounded_soln_l))
            custom_prnt("Unique solutions")
            custom_prnt(unique_soln_l)
            # END Extracting unique solutions

            # Get the norm of OO = TT - TA for each solution 
            norm_l = []
            for curr_soln in unique_soln_l:
                IOD.f2 = curr_soln
                IOD.prototype_v4_populate_dependent_focalLengths()
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()
                norm_l.append(IOD.norm)
            rounded_norm_l = [round(elem,2) for elem in norm_l]
            # custom_prnt(type(rounded_norm_l))
            custom_prnt("Norm of solutions")
            custom_prnt(rounded_norm_l)
            # END Get the norm of OO = TT - TA for each solution

            # Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices
            if(output_arrays_resized == 'False'):
                num_soln = len(unique_soln_l)
                if(prev_num_soln < num_soln):
                    num_new_colns = num_soln - prev_num_soln
                    new_cols = np.zeros((num_d_f2_f3, num_dist, num_new_colns))
                    op.f2_arr = np.dstack((op.f2_arr, new_cols))
                    op.f3_arr = np.dstack((op.f2_arr, new_cols))
                    op.norm_arr = np.dstack((op.norm_arr, new_cols))
                    op.d_OM_f4_arr = np.dstack((op.d_OM_f4_arr, new_cols))

                    new_cols = np.zeros((num_d_f2_f3, num_dist, num_new_colns, num_dist))
                    op.mag_arr = np.dstack((op.mag_arr, new_cols))
                    op.rw_dist = np.dstack((op.rw_dist, new_cols))
                    op.img_dist = np.dstack((op.img_dist, new_cols))
                    op.diff_dist = np.dstack((op.diff_dist, new_cols))
                output_arrays_resized = 'True'
            # END Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices

            for curr_soln in unique_soln_l:

                # if(curr_soln < 4):
                #     continue

                soln_index = unique_soln_l.index(curr_soln)
                IOD.f2 = curr_soln
                # op.f2_arr[dist_index, soln_index] = IOD.f2

                custom_prnt("\n")
                str = "f1 = %f cm" % (IOD.f1)
                custom_prnt(str)
                str = "f1 = %f D" % (cf.convert_cm2dpt(IOD.f1))
                custom_prnt(str)
                str = "f2 = %f cm" % (IOD.f2)
                custom_prnt(str)
                str = "f2 = %f D" % (cf.convert_cm2dpt(IOD.f2))
                custom_prnt(str)

                # Verify that TA ~= TT
                IOD.prototype_v4_populate_dependent_focalLengths()
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()

                TA_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
                str = "Actual Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TA_l)

                TT_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
                str = "Target Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TT_l)

                OO_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
                str = "residual matrix"
                custom_prnt(str)
                custom_prnt(OO_l)
                rounded_norm = round(IOD.norm, 2)
                op.norm_arr[d_f2_f3_index,dist_index, soln_index] = rounded_norm
                str = "norm = %f" % (rounded_norm)
                custom_prnt(str)
                # END Verify that TT ~= TA

                # Calculate where image of real world is formed
                # Calculate d_WI_f4
                # IOD.populate_d_eye(IOD.d_vip_eye)
                IOD.d_W_f1 = IOD.d_vip_eye
                IOD.propagate_rw_all(IOD.d_W_f1)
                # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
                str = "I1 = %f" % (IOD.I1)
                custom_prnt(str)
                str = "d_f1_LCoS = %f" % (IOD.d_f1_LCoS)
                custom_prnt(str)
                # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
                str = "d_WI_f4 = %f" % (IOD.d_WI_f4)
                custom_prnt(str)
                # END Calculate where image of real world is formed

                # Calculate where image of occlusion mask
                if (abs(IOD.I1 - IOD.d_f1_LCoS) < 0.01):
                    custom_prnt("No need to propagate OM separately because rw at vip formed at LCoS")
                    IOD.d_OM_f4 = IOD.d_WI_f4
                else:
                    custom_prnt("RW at vip did not form at LCoS")
                    # str = "d_vip_eye = %f; d_W_f1 = %f" % (IOD.d_vip_eye, IOD.d_W_f1)
                    # custom_prnt(str)
                    # str = "saveI1 = %f; saveO1 = %f; savef1 = %f" % (saveI1, saveO1, savef1)
                    # custom_prnt(str)
                    # str = "I1 = %f; O1 = %f; f1 = %f" % (IOD.I1, IOD.O1, IOD.f1)
                    # custom_prnt(str)
                    IOD.propagate_om()
                # END Calculate where image of occlusion mask
                str = "d_OM_f4 = %f" %(IOD.d_OM_f4)
                custom_prnt(str)
                str = "d_W_f1 = %f" % (-IOD.d_W_f1)
                custom_prnt(str)

                op.d_OM_f4_arr[d_f2_f3_index,dist_index, soln_index] = IOD.d_OM_f4

                str = "Magnification at all distances:"
                custom_prnt(str)
                for ncurr_dist in dists_l:
                    ncurr_dist_index = dists_l.index(ncurr_dist)

                    # IOD.populate_d_eye(ncurr_dist)
                    IOD.d_W_f1 = ncurr_dist

                    IOD.propagate_rw_all(IOD.d_W_f1) # Assume that ncurr_dist = d_W_f1

                    diff_dist = IOD.d_WI_f4 + IOD.d_W_f1
                    op.rw_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = IOD.d_W_f1
                    op.img_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = IOD.d_WI_f4
                    op.diff_dist[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = diff_dist

                    op.mag_arr[d_f2_f3_index, dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification

    # Print relevant info from op
    for d_f2_f3 in d_f2_f3_l:
        iter1 = d_f2_f3_l.index(d_f2_f3)
        str = "d_f2_f3 = %0.2f" % (d_f2_f3)
        print(str)
        for dist in dists_l:
            iter2 = dists_l.index(dist)
            # str = "    dist = %0.2f" % (dist)
            # print(str)
            mag_mean_l = []
            mag_std_l = []
            diff_mean_l = []
            diff_std_l = []
            for iter3 in range(0, num_soln):
                # str = "soln_number = %0.2f ========================" % (iter3)
                # print(str)
                mag_l = op.mag_arr[iter1, iter2, iter3, :]
                diff_l = op.diff_dist[iter1, iter2, iter3, :]

                mag_mean_l.append(np.mean(mag_l))
                mag_std_l.append(np.std(mag_l))
                diff_mean_l.append(np.mean(diff_l))
                diff_std_l.append(np.std(diff_l))
                # str = 'Avg(mag): %0.2f  Std(mag): %0.2f  Avg(dif): %0.2f  Std(dif): %0.2f' % (np.mean(mag_l), np.std(mag_l), np.mean(diff_l), np.std(diff_l))
                # print(str)

            mag_one = np.array([1]*len(mag_mean_l))
            mag_diff_from_1 = np.array(mag_mean_l) - mag_one
            # print(mag_diff_from_1)
            mag_diff_from_1 = np.multiply(mag_diff_from_1, mag_diff_from_1)
            # print(mag_diff_from_1)
            min_index = np.argmin(mag_diff_from_1)

            print('    vip_dist = %7.2f | avg(mag): %0.2f | std(mag): %0.2f | avg(dif): %7.2f | std(dif): %6.2f' % (dist, mag_mean_l[min_index], mag_std_l[min_index], diff_mean_l[min_index], diff_std_l[min_index]))
        print('\n')
    # end of tunable_all_symmetric
    # graph_outputs(op, d_f2_f3_l, dists_l, unique_soln_l, outputs_dir, ylabels_l, ylim_l)


'''
f3 = f4 = 5
d12 = d34
'''
def tunable_f1_f2():
    IOD = OD()
    op = outputs()
    diop_diff = 0.5
    min_dist = 25
    num_dist = 8
    dists_l = cf.calc_perceptually_useful_distances(min_dist, diop_diff, num_dist)

    std_output_arr = np.zeros(num_dist)
    op.f1_arr = np.copy(std_output_arr)
    op.d_W_f1_arr = np.copy(std_output_arr)
    op.d_f1_LCoS_arr =  np.copy(std_output_arr)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = 2
    std_output_arr = np.zeros((num_dist, num_soln))
    op.f2_arr = np.copy(std_output_arr)
    op.norm_arr = np.copy(std_output_arr)
    op.I1_arr = np.copy(std_output_arr)
    op.d_WI_f4_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_dist, num_soln, num_dist))
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels_l = ["f1", "d_W_f1", "d_f1_LCoS", "f2", "norm", "I1", "d_WI_f4", "d_OM_f4", "mag_arr", "img_dist"]
    ylim_l = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [0,-200]]
    output_arrays_resized = 'False'

    for curr_dist in dists_l:
        dist_index = dists_l.index(curr_dist)
        IOD.d_vip_eye = curr_dist # Should be > 23
        str = "d_vip_eye = %f" % curr_dist
        custom_prnt(str)

        IOD.populate_d_eye(curr_dist)
        IOD.f1 = cf.calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
        op.f1_arr[dist_index] = IOD.f1
        sym_f2 = Symbol('f_2')
        IOD.f2 = sym_f2
        IOD.f3 = 5
        IOD.f4 = 5
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()

        II = Matrix([[1,0], [0,1]])
        IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
        S14 = cf.makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
        TT = II
        IOD.TT = II

        IOD.calc_TA_diff_TT()
        OO = IOD.OO

        OO_l = OO.tolist()
        flat_OO_l = []
        cf.conv_lol_flat_l(OO_l, flat_OO_l)

        soln_l = []
        for curr_oo in flat_OO_l:
            curr_soln = list(solve(curr_oo, sym_f2))
            soln_l.append(curr_soln)

        custom_prnt("Solutions")
        custom_prnt(soln_l)
        custom_prnt("\n")
        # END Getting solutions for all equations together

        # Get the norm of OO = TT - TA for each solution 
        norm_l = []
        for curr_soln in soln_l:
            IOD.f2 = curr_soln[0]
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()
            IOD.calc_TA_diff_TT()
            IOD.calc_OO_norm()
            norm_l.append(IOD.norm)
        rounded_norm_l = [round(elem,2) for elem in norm_l]
        # custom_prnt(type(rounded_norm_l))
        custom_prnt("Norm of solutions")
        custom_prnt(rounded_norm_l)
        # END Get the norm of OO = TT - TA for each solution

        # Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices
        if(output_arrays_resized == 'False'):
            num_soln = len(soln_l)
            if(prev_num_soln < num_soln):
                num_new_colns = num_soln - prev_num_soln
                new_cols = np.zeros((num_dist, num_new_colns))
                op.f2_arr = np.hstack((op.f2_arr, new_cols))
                op.norm_arr = np.hstack((op.norm_arr, new_cols))
                op.I1_arr = np.hstack((op.I1_arr, new_cols))
                op.d_WI_f4_arr = np.hstack((op.d_WI_f4_arr, new_cols))
                op.d_OM_f4_arr = np.hstack((op.d_OM_f4_arr, new_cols))

                new_cols = np.zeros((num_dist, num_new_colns, num_dist))
                op.mag_arr = np.hstack((op.mag_arr, new_cols))
                op.img_dist = np.hstack((op.img_dist, new_cols))
            output_arrays_resized = 'True'
        # END Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices

        for curr_soln in soln_l:
            soln_index = soln_l.index(curr_soln)
            IOD.f2 = curr_soln[0]
            op.f2_arr[dist_index, soln_index] = IOD.f2

            custom_prnt("\n")
            str = "f1 = %f cm" % (IOD.f1)
            custom_prnt(str)
            str = "f1 = %f D" % (cf.convert_cm2dpt(IOD.f1))
            custom_prnt(str)
            str = "f2 = %f cm" % (IOD.f2)
            custom_prnt(str)
            str = "f2 = %f D" % (cf.convert_cm2dpt(IOD.f2))
            custom_prnt(str)

            # Verify that TA ~= TT
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()
            IOD.calc_TA_diff_TT()
            IOD.calc_OO_norm()

            TA_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
            str = "Actual Transfer matrix:"
            custom_prnt(str)
            custom_prnt(TA_l)

            TT_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
            str = "Target Transfer matrix:"
            custom_prnt(str)
            custom_prnt(TT_l)

            OO_l = cf.convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
            str = "residual matrix"
            custom_prnt(str)
            custom_prnt(OO_l)
            rounded_norm = round(IOD.norm, 2)
            op.norm_arr[dist_index, soln_index] = rounded_norm
            str = "norm = %f" % (rounded_norm)
            custom_prnt(str)
            # END Verify that TT ~= TA

            # Calculate where image of real world is formed
            # Calculate d_WI_f4
            IOD.populate_d_eye(IOD.d_vip_eye)
            IOD.propagate_rw_all(IOD.d_W_f1)
            # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            op.I1_arr[dist_index, soln_index] = IOD.I1
            op.d_f1_LCoS_arr[dist_index] = IOD.d_f1_LCoS
            str = "I1 = %f" % (IOD.I1)
            custom_prnt(str)
            str = "d_f1_LCoS = %f" % (IOD.d_f1_LCoS)
            custom_prnt(str)
            # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            str = "d_WI_f4 = %f" % (IOD.d_WI_f4)
            custom_prnt(str)
            op.d_WI_f4_arr[dist_index, soln_index] = IOD.d_WI_f4
            # END Calculate where image of real world is formed

            # Calculate where image of occlusion mask
            if (abs(IOD.I1 - IOD.d_f1_LCoS) < 0.01):
                custom_prnt("No need to propagate OM separately because rw at vip formed at LCoS")
                IOD.d_OM_f4 = IOD.d_WI_f4
            else:
                custom_prnt("RW at vip did not form at LCoS")
                str = "d_vip_eye = %f; d_W_f1 = %f" % (IOD.d_vip_eye, IOD.d_W_f1)
                custom_prnt(str)
                str = "saveI1 = %f; saveO1 = %f; savef1 = %f" % (saveI1, saveO1, savef1)
                custom_prnt(str)
                str = "I1 = %f; O1 = %f; f1 = %f" % (IOD.I1, IOD.O1, IOD.f1)
                custom_prnt(str)
                IOD.propagate_om()
            # END Calculate where image of occlusion mask
            str = "d_OM_f4 = %f" %(IOD.d_OM_f4)
            custom_prnt(str)
            str = "d_W_f1 = %f" % (-IOD.d_W_f1)
            custom_prnt(str)

            op.d_OM_f4_arr[dist_index, soln_index] = IOD.d_OM_f4
            op.d_W_f1_arr[dist_index] = -IOD.d_W_f1

            str = "Magnification at all distances:"
            custom_prnt(str)
            for ncurr_dist in dists_l:
                ncurr_dist_index = dists_l.index(ncurr_dist)
                IOD.populate_d_eye(ncurr_dist)
                IOD.propagate_rw_all(IOD.d_W_f1)
                custom_prnt(IOD.rw_magnification)
                op.mag_arr[dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification
                op.img_dist[dist_index, soln_index, ncurr_dist_index] = IOD.d_WI_f4
        
    graph_outputs(op, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l)
    # end of tunable_f1_f2

'''
Modelling a modified optical system from
Howlett-Smithwick-SID2017-Perspective correct occlusion-capable augmented reality displays using cloaking optics constraints
'''
def howlett_1D():
    print('####################################')
    IOD = OD.optical_design()
    op = outputs()
    # diop_diff = 0.6
    # max_dist = cf.convert_m2cm(10)
    # num_dist = 3
    # dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)
    dists_l = [1000, 250, 64, 16]
    num_dist = len(dists_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    std_output_arr = np.zeros(num_dist)
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels_l = ["mag_arr", "img_dist"]
    ylim_l = [[-1.5, 1.5], [-1, -1]]

    common_f = 5
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f 
    curr_lens.d_prev_lens = 2*common_f
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = 2*common_f
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    howlett_t = 20

    II = Matrix([[1,0], [0,1]])

    d_f2_f3_l = [howlett_t - 3, howlett_t - 1, howlett_t, howlett_t + 1, howlett_t + 3]
    # d_f2_f3_l = [25, 27, howlett_t, 29, 31]
    num_d_f2_f3 = len(d_f2_f3_l)
    # d_f2_f3_l = list(range(3, 28, 5))
    # d_f2_f3_l.append(howlett_t)
    for curr_t in d_f2_f3_l:
        print('')
        print('')
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        str = "d_f2_f3 = %0.2f" % (curr_t)
        print(str)

        IOD.length = 0

        IOD.lens_l[2].d_prev_lens = curr_t
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()

        IOD.TT = II
        IOD.calc_TA_diff_TT()
        IOD.calc_OO_norm()
        # print('TT = ', end = " ")
        # print(IOD.TT)
        # print('TA = ', end = " ")
        # print(IOD.TA)
        # print('OO = ', end = " ")
        # print(IOD.OO)
        # print('Norm: %7.2f' %(IOD.norm))

        # OO = IOD.OO
        # OO_l = OO.tolist()
        # flat_OO_l = []
        # cf.conv_lol_flat_l(OO_l, flat_OO_l)

        str = "Magnification at all distances:"
        custom_prnt(str)
        diff_l = []
        mag_l = []
        for dist in dists_l:
            # str = "    dist = %0.2f" % (dist)
            # print(str)

            # dist_index = dists_l.index(dist)

            # # IOD.populate_d_eye(dist)
            # IOD.d_vip_eye = dist
            # IOD.d_W_f1 = IOD.d_vip_eye - IOD.d_f4_eye - howlett_d
            # IOD.d_W_f4 = IOD.d_vip_eye - IOD.d_f4_eye

            IOD.propagate_rw_all(dist) # Assume that dist = d_W_f1

            diff_dist = IOD.lens_l[-1].d_image + IOD.length + dist
            diff_l.append(diff_dist)

            # mag_l.append(IOD.magnification)
            mag_l.append(IOD.lens_l[-1].d_image/(dist + IOD.length))


        mag_arr = np.array(mag_l, dtype=np.float64)
        diff_arr = np.array(diff_l, dtype=np.float64)

        print('    vip_dist = inf | avg(mag): %0.2f | std(mag): %0.2f | avg(dif): %6.2f | std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr)))
        # str = '        avg(mag): %0.2f  std(mag): %0.2f  avg(dif): %0.2f  std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
        # print(str)
        print('\n')

        # for curr_lens in IOD.lens_l:
        #     print(curr_lens.M)

        # for curr_lens in IOD.lens_l:
        #     print(curr_lens.S)

        # print(IOD.TA)
        # print(IOD.OO)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        # soln_l = []
        # graph_outputs(op, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l)
    
    
'''
Using the new OD.py
Modelling the optical system from
Howlett-Smithwick-SID2017-Perspective correct occlusion-capable augmented reality displays using cloaking optics constraints
'''
def howlett_upgraded_OD():
    print('####################################')
    IOD = OD.optical_design()
    op = outputs()
    # diop_diff = 0.6
    # max_dist = cf.convert_m2cm(10)
    # num_dist = 3
    # dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)
    dists_l = [1000, 250, 64, 16]
    num_dist = len(dists_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    std_output_arr = np.zeros(num_dist)
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels_l = ["mag_arr", "img_dist"]
    ylim_l = [[-1.5, 1.5], [-1, -1]]

    common_f = 4
    howlett_d = 8 
    howlett_t = howlett_d + 4*common_f
    # howlett_t = 4*IOD.f1

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)
    
    curr_lens = OD.lens()
    curr_lens.focal_length = common_f 
    curr_lens.d_prev_lens = howlett_d
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = 0.0
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)

    curr_lens = OD.lens()
    curr_lens.focal_length = common_f
    curr_lens.d_prev_lens = howlett_d
    curr_lens.tunable = False
    IOD.lens_l.append(curr_lens)


    d_f2_f3_l = [howlett_t - 3, howlett_t - 1, howlett_t, howlett_t + 1, howlett_t + 3]
    # d_f2_f3_l = [25, 27, howlett_t, 29, 31]
    num_d_f2_f3 = len(d_f2_f3_l)
    # d_f2_f3_l = list(range(3, 28, 5))
    # d_f2_f3_l.append(howlett_t)
    for curr_t in d_f2_f3_l:
        print('')
        print('')
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        str = "d_f2_f3 = %0.2f" % (curr_t)
        print(str)

        IOD.lens_l[2].d_prev_lens = curr_t
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()

        IOD.length = 8
        # for curr_lens in IOD.lens_l:
        #     IOD.length = IOD.length + curr_lens.d_prev_lens

        S14 = cf.makeFreeSpacePropagationMatrix(howlett_d)
        IOD.TT = S14

        IOD.calc_TA_diff_TT()
        IOD.calc_OO_norm()
        # print('TT = ', end = " ")
        # print(IOD.TT)
        # print('TA = ', end = " ")
        # print(IOD.TA)
        # print('OO = ', end = " ")
        # print(IOD.OO)
        # print('Norm: %7.2f' %(IOD.norm))

        # OO = IOD.OO
        # OO_l = OO.tolist()
        # flat_OO_l = []
        # cf.conv_lol_flat_l(OO_l, flat_OO_l)

        str = "Magnification at all distances:"
        custom_prnt(str)
        diff_l = []
        mag_l = []
        for dist in dists_l:
            # str = "    dist = %0.2f" % (dist)
            # print(str)

            # dist_index = dists_l.index(dist)

            # # IOD.populate_d_eye(dist)
            # IOD.d_vip_eye = dist
            # IOD.d_W_f1 = IOD.d_vip_eye - IOD.d_f4_eye - howlett_d
            # IOD.d_W_f4 = IOD.d_vip_eye - IOD.d_f4_eye

            IOD.propagate_rw_all(dist) # Assume that dist = d_W_f1

            diff_dist = IOD.lens_l[-1].d_image + IOD.length + dist
            diff_l.append(diff_dist)

            # mag_l.append(IOD.magnification)
            mag_l.append(IOD.lens_l[-1].d_image/(dist + IOD.length))


        mag_arr = np.array(mag_l, dtype=np.float64)
        diff_arr = np.array(diff_l, dtype=np.float64)

        print('    vip_dist = inf | avg(mag): %0.2f | std(mag): %0.2f | avg(dif): %6.2f | std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr)))
        # str = '        avg(mag): %0.2f  std(mag): %0.2f  avg(dif): %0.2f  std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
        # print(str)
        print('\n')

        # for curr_lens in IOD.lens_l:
        #     print(curr_lens.M)

        # for curr_lens in IOD.lens_l:
        #     print(curr_lens.S)

        # print(IOD.TA)
        # print(IOD.OO)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        # soln_l = []
        # graph_outputs(op, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l)

'''
Modelling the optical system from
Howlett-Smithwick-SID2017-Perspective correct occlusion-capable augmented reality displays using cloaking optics constraints
'''
def howlett():
    IOD = OD()
    op = outputs()
    # diop_diff = 0.6
    # max_dist = cf.convert_m2cm(10)
    # num_dist = 3
    # dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)
    dists_l = [1000, 250, 64, 16]
    num_dist = len(dists_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    std_output_arr = np.zeros(num_dist)
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels_l = ["mag_arr", "img_dist"]
    ylim_l = [[-1.5, 1.5], [-1, -1]]

    IOD.f1 = 5
    IOD.f2 = 5
    IOD.f3 = 5
    IOD.f4 = 5

    howlett_d = IOD.d_f1_f2
    howlett_t = howlett_d + 4*IOD.f1
    # howlett_t = 4*IOD.f1

    d_f2_f3_l = [25, 27, howlett_t, 29, 31]
    num_d_f2_f3 = len(d_f2_f3_l)
    # d_f2_f3_l = list(range(3, 28, 5))
    # d_f2_f3_l.append(howlett_t)
    for IOD.d_f2_f3 in d_f2_f3_l:
        str = "d_f2_f3 = %0.2f" % (IOD.d_f2_f3)
        print(str)

        IOD.d_f1_f2 = 2*IOD.f1
        IOD.d_f3_f4 = 2*IOD.f1
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()
        II = Matrix([[1,0], [0,1]])
        IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
        S14 = cf.makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
        TT = II
        # TT = S14
        IOD.TT = II

        IOD.calc_TA_diff_TT()
        OO = IOD.OO
        OO_l = OO.tolist()
        flat_OO_l = []
        cf.conv_lol_flat_l(OO_l, flat_OO_l)

        str = "Magnification at all distances:"
        custom_prnt(str)
        diff_l = []
        mag_l = []
        for dist in dists_l:
            # str = "    dist = %0.2f" % (dist)
            # print(str)

            dist_index = dists_l.index(dist)

            # IOD.populate_d_eye(dist)
            IOD.d_vip_eye = dist
            IOD.d_W_f1 = IOD.d_vip_eye - IOD.d_f4_eye - howlett_d
            IOD.d_W_f4 = IOD.d_vip_eye - IOD.d_f4_eye

            IOD.propagate_rw_all(IOD.d_W_f1) # Assume that dist = d_W_f1

            diff_dist = IOD.d_WI_f4 + IOD.d_W_f4
            diff_l.append(diff_dist)

            mag_l.append(IOD.rw_magnification)


        mag_arr = np.array(mag_l, dtype=np.float64)
        diff_arr = np.array(diff_l, dtype=np.float64)

        print('    vip_dist = inf | avg(mag): %0.2f | std(mag): %0.2f | avg(dif): %6.2f | std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr)))
        # str = '        avg(mag): %0.2f  std(mag): %0.2f  avg(dif): %0.2f  std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
        # print(str)
        print('\n')

        # soln_l = []
        # graph_outputs(op, dists_l, soln_l, outputs_dir, ylabels_l, ylim_l)


def initialize_IOD(IOD):
    init_d12 =
    init_d23 =
    init_d34 =
    init_f2 =
    init_f3 =
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

def paper_revision():
    IOD = OD.optical_design()
    initialize_IOD(IOD)
    initialize_rw_distances()

 

# An attempt to automate tunable_all_symmetric, tunable_f1_f2_f3, etc. by specifying just the unknowns
def main_all():
    print('DO NOT USE THIS MAIN. IT''S WORK IN PROGRESS')
    IOD = OD()
    op = outputs()
    diop_diff = 0.6
    max_dist = cf.convert_m2cm(10)
    num_dist = 5
    dists_l = cf.calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)

    focal_lengths = ['f2', 'f3', 5]
    # print(type(focal_lengths[0]))
    # print(type(focal_lengths[1]))
    # print(type(focal_lengths[2]))
    # for focal_length in focal_lengths:
    #     if(isinstance(focal_length, int)):
    #         print('int confirmed')
    #     elif(isinstance(focal_length, str)):
    #         print('str confirmed')


    d_f2_f3_l = list(range(3, 18, 2))
    num_d_f2_f3 = len(d_f2_f3_l)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = 2
    
    std_output_arr = np.zeros((num_d_f2_f3, num_dist))
    op.f1_arr = np.copy(std_output_arr)
    op.d_W_f1_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln, num_dist))
    op.mag_arr = np.copy(std_output_arr)
    op.rw_dist = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)
    op.diff_dist = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_d_f2_f3, num_dist, num_soln))
    op.norm_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)

    ylabels_l = ["f1", "d_W_f1", "mag_arr", "rw_dist", "img_dist", "diff_dist", "norm", "d_OM_f4"]
    ylim_l = [[-1,-1], [-1,-1], [-1.5, 1.5], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1]]

    for focal_length in focal_lengths:
        if(isinstance(focal_length, str)):
            setattr(op, focal_length, np.copy(std_output_arr))
            ylabels_l.append(focal_length)
            ylim_l.append([-1,-1])

    output_arrays_resized = 'False'

    for IOD.d_f2_f3 in d_f2_f3_l:
        for curr_dist in dists_l:
            dist_index = dists_l.index(curr_dist)
            IOD.d_vip_eye = curr_dist # Should be > 23
            IOD.d_vip_eye = curr_dist
    
if __name__ == '__main__':
    # tunable_f1_f2_f3()
    # tunable_f1_f2()
    # howlett_upgraded_OD()
    # howlett_1D()
    paper_revision()
    # tunable_all_symmetric()

