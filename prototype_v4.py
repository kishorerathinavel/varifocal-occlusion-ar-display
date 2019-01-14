from common_functions import *
from sympy import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from sympy.core.symbol import symbols
#from sympy.solvers.solveset import nonlinsolve

prnt_flag = 'False'
# prnt_flag = 'True'
outputs_dir = 'outputs'

class OD(): # Short for Implemented Optical Design
    def __init__(self):
        self.d_f1_LCoS = 4.0 # Minimum possible
        self.d_LCoS_f2 = 4.0 # Minimum possible
        self.d_f1_f2 = self.d_f1_LCoS + self.d_LCoS_f2
        self.d_f2_f3 = 16 # Guess
        self.d_f3_f4 = self.d_f1_f2
        self.d_f4_eye = 2.0

        # Uninitialized
        self.f1 = 0.0
        self.f2 = 0.0
        self.f3 = 0.0
        self.f4 = 0.0
        self.d_W_f1 = 0.0  # World (not necessarily vip) to f1
        self.d_W_eye = 0.0 # World (not necessarily vip) to eye
        self.d_WI_eye = 0.0 # Image of world (not necessarily image of world at vip) to eye
        self.d_OI_eye = 0.0 # Occlusion mask to eye
        self.d_vip_eye = 0.0
        self.magnification = 0.0

    def populate_d_eye(self, ncurr_dist):
        self.d_W_eye = ncurr_dist
        self.d_W_f1 = self.d_W_eye - self.d_f4_eye - self.d_f3_f4 - self.d_f2_f3 - self.d_f1_f2

    def propagate_rw_all(self, ncurr_dist):
        self.O1 = ncurr_dist
        self.I1 = calculate_image_distance(self.O1, self.f1)
        self.m1 = self.I1/self.O1
        self.O2 = self.d_f1_f2 - self.I1
        self.I2 = calculate_image_distance(self.O2, self.f2)
        self.m2 = self.I2/self.O2
        self.O3 = self.d_f2_f3 - self.I2
        self.I3 = calculate_image_distance(self.O3, self.f3)
        self.m3 = self.I3/self.O3
        self.O4 = self.d_f3_f4 - self.I3
        self.I4 = calculate_image_distance(self.O4, self.f4)
        self.m4 = self.I4/self.O4
        self.rw_magnification = self.m1*self.m2*self.m3*self.m4
        self.d_WI_f4 = self.I4
        self.d_WI_f1 = self.d_WI_f4 + self.d_f1_f2 + self.d_f2_f3 + self.d_f3_f4

    def propagate_om(self):
        self.O2 = self.d_LCoS_f2
        self.I2 = calculate_image_distance(self.O2, self.f2)
        self.O3 = self.d_f2_f3 - self.I2
        self.I3 = calculate_image_distance(self.O3, self.f3)
        self.O4 = self.d_f3_f4 - self.I3
        self.I4 = calculate_image_distance(self.O4, self.f4)
        self.d_OM_f4 = self.I4

    def calc_ABCD_matrices(self):
        self.M1 = makeLensMatrix(1/self.f1)
        self.M2 = makeLensMatrix(1/self.f2)
        self.M3 = makeLensMatrix(1/self.f3)
        self.M4 = makeLensMatrix(1/self.f4)
        self.S12 = makeFreeSpacePropagationMatrix(self.d_f1_f2)
        self.S23 = makeFreeSpacePropagationMatrix(self.d_f2_f3)
        self.S34 = makeFreeSpacePropagationMatrix(self.d_f3_f4)

    def calc_TA(self):
            self.TA = self.M4*self.S34*self.M3*self.S23*self.M2*self.S12*self.M1

    def prototype_v4_populate_dependent_focalLengths(self):
        self.f3 = self.f2
        self.f4 = self.f1

    def calc_TA_diff_TT(self):
        self.OO = self.TT - self.TA

    def calc_OO_norm(self):
        OO_np = np.array(self.OO.tolist()).astype(np.float64)
        self.norm = LA.norm(OO_np)
    
class outputs():
    pass

def main_nonlinsolve_eg():
    x, y, z = symbols('x, y, z', real=True)
    custom_prnt(nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y]))

def convert_sympy_mutableDenseMatrix_custom_prntableList(denseMatrix):
    denseMatrix_np = np.array(denseMatrix.tolist()).astype(np.float64)
    r_denseMatrix_np = np.round(denseMatrix_np, 2)
    denseMatrix_l = r_denseMatrix_np.tolist()
    return denseMatrix_l

def calc_perceptually_useful_distances(max_dist, diop_diff, num_dist):
    # Assuming that min_dist is specified in cm
    max_diop_dist = convert_cm2dpt(max_dist)
    prev_diop_dist = max_diop_dist
    dists = []
    dists.append(max_dist)
    for iter in range(1,num_dist):
        next_diop_dist = prev_diop_dist + diop_diff
        next_dist = convert_dpt2cm(next_diop_dist)
        dists.append(next_dist)
        prev_diop_dist = next_diop_dist
    return dists

def conv_lol_flat_l(my_input, output_list):
    if isinstance(my_input, list):
        for element in my_input:
            conv_lol_flat_l(element, output_list)
    elif isinstance(my_input, Tuple):
        for element in my_input:
            conv_lol_flat_l(element, output_list)
    else:
        return output_list.append(my_input)

def custom_prnt(str):
    if(prnt_flag == 'True'):
        print(str)

def graph_outputs(op, dists, soln_l, outputs_dir, ylabels, ylim_arr):
    num_soln = len(soln_l)
    num_dists = len(dists)
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
            plt.plot(dists, value)
            plt.ylabel(ylabels[iter_ylabel])
            plt.xlabel("distance")
            curr_ylim = ylim_arr[iter_ylabel]
            if(curr_ylim[0] != curr_ylim[1]):
                plt.ylim(curr_ylim[0], curr_ylim[1])
            plt.savefig(str)
        elif(dimensions == 2):
            for iter_soln in range(0, num_soln):
                str = "./%s/D%d_%d_%s.png" % (outputs_dir, dimensions, iter_soln, key)
                plt.clf()
                plt.plot(dists, value[:,iter_soln])
                plt.ylabel(ylabels[iter_ylabel])
                plt.xlabel("distance")
                curr_ylim = ylim_arr[iter_ylabel]
                if(curr_ylim[0] != curr_ylim[1]):
                    plt.ylim(curr_ylim[0], curr_ylim[1])
                plt.savefig(str)
        elif(dimensions == 3):
            for iter_dist in range(0, num_dists):
                for iter_soln in range(0, num_soln):
                    str = "./%s/D%d_%d_%d_%s.png" % (outputs_dir, dimensions, iter_dist, iter_soln, key)
                    plt.clf()
                    plt.plot(dists, value[iter_dist,iter_soln, :])
                    plt.ylabel(ylabels[iter_ylabel])
                    plt.xlabel("distance")
                    curr_ylim = ylim_arr[iter_ylabel]
                    if(curr_ylim[0] != curr_ylim[1]):
                        plt.ylim(curr_ylim[0], curr_ylim[1])
                    plt.savefig(str)
        iter_ylabel = iter_ylabel + 1
        

'''
f4 = 5
d12 = d34
solve iteratively
'''
def main6():
    IOD = OD()
    op = outputs()
    diop_diff = 0.5
    min_dist = 25
    num_dist = 8
    dists = calc_perceptually_useful_distances(min_dist, diop_diff, num_dist)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = 2
    std_output_arr = np.zeros((num_dist, num_soln))
    op.f1_arr = np.copy(std_output_arr)
    op.f2_arr = np.copy(std_output_arr)
    op.f3_arr = np.copy(std_output_arr)
    op.norm_arr = np.copy(std_output_arr)
    op.I1_arr = np.copy(std_output_arr)
    op.d_f1_LCoS_arr =  np.copy(std_output_arr)
    op.d_WI_f4_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)
    op.d_W_f1_arr = np.copy(std_output_arr)
    op.mag_arr = np.copy(std_output_arr)

    for curr_dist in dists:
        dist_index = dists.index(curr_dist)
        IOD.d_vip_eye = curr_dist # Should be > 23
        str = "d_vip_eye = %f" % curr_dist
        custom_prnt(str)

        IOD.populate_d_eye(curr_dist)
        IOD.O1 = IOD.d_W_f1
        IOD.I1 = IOD.d_f1_LCoS
        IOD.f1 = calculate_focal_length(IOD.O1, IOD.I1)
        m1 = -IOD.I1/IOD.O1
        IOD.O2 = IOD.d_LCoS_f2
        IOD.I2 = -(IOD.d_LCoS_f2 + IOD.d_f1_LCoS + IOD.d_W_f1)
        IOD.f2 = calculate_focal_length(IOD.O2, IOD.I2)
        m2 = -IOD.I2/IOD.O2
        mT = m1*m2

        str = "m1 = %f, m2 = %f, mT = %f" % (m1, m2, mT)
        custom_prnt(str)
        return

        II = Matrix([[1,0], [0,1]])
        IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
        S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
        TT = II
        IOD.TT = S14

        sym_f3 = Symbol('f_3')
        IOD.f3 = sym_f3
        IOD.f4 = 5
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()
        IOD.calc_TA_diff_TT()
        OO = IOD.OO
        custom_prnt_matrix(OO)
        soln_l = list(solve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f3]))
        custom_prnt(soln_l)

        # sym_f2 = Symbol('f_2')
        # IOD.f2 = sym_f2

'''
f4 = 5
d12 = d34
solve using python solver
'''
def main5():
    IOD = OD()
    op = outputs()
    diop_diff = 0.6
    max_dist = convert_m2cm(10)
    num_dist = 5
    dists = calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)

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
    op.f3_arr = np.copy(std_output_arr)
    op.norm_arr = np.copy(std_output_arr)
    op.I1_arr = np.copy(std_output_arr)
    op.d_WI_f4_arr = np.copy(std_output_arr)
    op.d_OM_f4_arr = np.copy(std_output_arr)

    std_output_arr = np.zeros((num_dist, num_soln, num_dist))
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels = ["f1", "d_W_f1", "d_f1_LCoS", "f2", "f3", "norm", "I1", "d_WI_f4", "d_OM_f4", "mag_arr", "img_dist"]
    ylim_arr = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [-1,-1]]
    output_arrays_resized = 'False'

    for IOD.d_f2_f3 in range(3, 18, 2):
        str = "d_f2_f3 = %0.2f ========================" % (IOD.d_f2_f3)
        print(str)
        for curr_dist in dists:
            str = "d_vip_eye = %0.2f ========================" % (curr_dist)
            print(str)
            dist_index = dists.index(curr_dist)
            IOD.d_vip_eye = curr_dist # Should be > 23
            str = "d_vip_eye = %f" % curr_dist
            custom_prnt(str)

            IOD.populate_d_eye(curr_dist)
            IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
            op.f1_arr[dist_index] = IOD.f1
            sym_f2 = Symbol('f_2')
            IOD.f2 = sym_f2
            sym_f3 = Symbol('f_3')
            IOD.f3 = sym_f3
            IOD.f4 = 5
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()

            II = Matrix([[1,0], [0,1]])
            IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
            S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
            TT = II
            IOD.TT = II

            IOD.calc_TA_diff_TT()
            OO = IOD.OO

            OO_l = OO.tolist()
            flat_OO_l = []
            conv_lol_flat_l(OO_l, flat_OO_l)

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
                    new_cols = np.zeros((num_dist, num_new_colns))
                    op.f2_arr = np.hstack((op.f2_arr, new_cols))
                    op.f3_arr = np.hstack((op.f2_arr, new_cols))
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
                str = "curr_soln_index = %d ========================" % (soln_l.index(curr_soln))
                print(str)
                soln_index = soln_l.index(curr_soln)
                IOD.f2 = curr_soln[0]
                IOD.f3 = curr_soln[1]
                op.f2_arr[dist_index, soln_index] = IOD.f2
                op.f3_arr[dist_index, soln_index] = IOD.f3

                custom_prnt("\n")
                str = "f1 = %f cm" % (IOD.f1)
                custom_prnt(str)
                str = "f1 = %f D" % (convert_cm2dpt(IOD.f1))
                custom_prnt(str)
                str = "f2 = %f cm" % (IOD.f2)
                custom_prnt(str)
                str = "f2 = %f D" % (convert_cm2dpt(IOD.f2))
                custom_prnt(str)
                str = "f3 = %f cm" % (IOD.f3)
                custom_prnt(str)
                str = "f3 = %f D" % (convert_cm2dpt(IOD.f3))
                custom_prnt(str)

                # Verify that TA ~= TT
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()

                TA_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
                str = "Actual Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TA_l)

                TT_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
                str = "Target Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TT_l)

                OO_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
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

                op.mag_arr[dist_index, soln_index] = IOD.rw_magnification

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
                diff_l = []
                mag_l = []
                for ncurr_dist in dists:
                    ncurr_dist_index = dists.index(ncurr_dist)

                    # IOD.populate_d_eye(ncurr_dist)
                    IOD.d_W_f1 = ncurr_dist

                    IOD.propagate_rw_all(IOD.d_W_f1) # Assume that ncurr_dist = d_W_f1

                    diff_dist = IOD.d_WI_f4 + IOD.d_W_f1
                    diff_l.append(diff_dist)
                    op.img_dist[dist_index, soln_index, ncurr_dist_index] = diff_dist

                    mag_l.append(IOD.rw_magnification)
                    op.mag_arr[dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification



                mag_l_rounded = [round(elem,2) for elem in mag_l]
                # print(mag_l_rounded)
                mag_arr = np.array(mag_l, dtype=np.float64)

                diff_l_rounded = [round(elem,2) for elem in diff_l]
                # print(diff_l_rounded)
                diff_arr = np.array(diff_l, dtype=np.float64)

                str = 'Avg(mag): %0.2f  Std(mag): %0.2f  Avg(dif): %0.2f  Std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
                print(str)
                print('\n')

                # Collect the average difference of the better solution

    # graph_outputs(op, dists, soln_l, outputs_dir, ylabels, ylim_arr)

'''
f1 = f4
f2 = f3
d12 = d34
'''
def main4():
    IOD = OD()
    op = outputs()
    diop_diff = 0.6
    max_dist = convert_m2cm(10)
    num_dist = 5
    dists = calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)

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

    ylabels = ["f1", "d_W_f1", "d_f1_LCoS", "f2", "norm", "I1", "d_WI_f4", "d_OM_f4", "mag_arr", "img_dist"]
    ylim_arr = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [0,-200]]
    output_arrays_resized = 'False'

    for IOD.d_f2_f3 in range(15, 18):
        str = "d_f2_f3 = %0.2f ========================" % (IOD.d_f2_f3)
        print(str)
        for curr_dist in dists:
            str = "d_vip_eye = %0.2f ========================" % (curr_dist)
            print(str)
            dist_index = dists.index(curr_dist)
            IOD.d_vip_eye = curr_dist # Should be > 23
            str = "d_vip_eye = %f" % curr_dist
            custom_prnt(str)

            IOD.populate_d_eye(curr_dist)
            IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
            op.f1_arr[dist_index] = IOD.f1
            sym_f2 = Symbol('f_2')
            IOD.f2 = sym_f2
            IOD.prototype_v4_populate_dependent_focalLengths()
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()

            II = Matrix([[1,0], [0,1]])
            IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
            S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
            TT = II
            IOD.TT = II

            IOD.calc_TA_diff_TT()
            OO = IOD.OO

            # custom_prnt_matrix(OO)
            OO_l = OO.tolist()
            flat_OO_l = []
            conv_lol_flat_l(OO_l, flat_OO_l)

            # Getting solutions for all equations together
            soln_l = list(nonlinsolve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f2]))
            # END Getting solutions for all equations together

            # Converting from sympy set,tuple to python list
            soln_l2 = []
            conv_lol_flat_l(soln_l, soln_l2)
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

            for curr_soln in unique_soln_l:
                str = "curr_soln_index = %d ========================" % (unique_soln_l.index(curr_soln))
                print(str)

                # if(curr_soln < 4):
                #     continue

                soln_index = unique_soln_l.index(curr_soln)
                IOD.f2 = curr_soln
                op.f2_arr[dist_index, soln_index] = IOD.f2

                custom_prnt("\n")
                str = "f1 = %f cm" % (IOD.f1)
                custom_prnt(str)
                str = "f1 = %f D" % (convert_cm2dpt(IOD.f1))
                custom_prnt(str)
                str = "f2 = %f cm" % (IOD.f2)
                custom_prnt(str)
                str = "f2 = %f D" % (convert_cm2dpt(IOD.f2))
                custom_prnt(str)

                # Verify that TA ~= TT
                IOD.prototype_v4_populate_dependent_focalLengths()
                IOD.calc_ABCD_matrices()
                IOD.calc_TA()
                IOD.calc_TA_diff_TT()
                IOD.calc_OO_norm()

                TA_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
                str = "Actual Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TA_l)

                TT_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
                str = "Target Transfer matrix:"
                custom_prnt(str)
                custom_prnt(TT_l)

                OO_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
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
                print(IOD.d_W_f1)
                IOD.propagate_rw_all(IOD.d_W_f1)
                # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
                op.I1_arr[dist_index, soln_index] = IOD.I1
                op.d_f1_LCoS_arr[dist_index] = IOD.d_f1_LCoS
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
                diff_l = []
                mag_l = []
                for ncurr_dist in dists:
                    ncurr_dist_index = dists.index(ncurr_dist)

                    # IOD.populate_d_eye(ncurr_dist)
                    IOD.d_W_f1 = ncurr_dist

                    IOD.propagate_rw_all(IOD.d_W_f1) # Assume that ncurr_dist = d_W_f1

                    diff_dist = IOD.d_WI_f4 + IOD.d_W_f1
                    diff_l.append(diff_dist)
                    op.img_dist[dist_index, soln_index, ncurr_dist_index] = diff_dist

                    mag_l.append(IOD.rw_magnification)
                    op.mag_arr[dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification



                mag_l_rounded = [round(elem,2) for elem in mag_l]
                # print(mag_l_rounded)
                mag_arr = np.array(mag_l, dtype=np.float64)

                diff_l_rounded = [round(elem,2) for elem in diff_l]
                # print(diff_l_rounded)
                diff_arr = np.array(diff_l, dtype=np.float64)

                str = 'Avg(mag): %0.2f  Std(mag): %0.2f  Avg(dif): %0.2f  Std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
                print(str)
                print('\n')

    # graph_outputs(op, dists, soln_l, outputs_dir, ylabels, ylim_arr)


'''
f3 = f4 = 5
d12 = d34
'''
def main9():
    IOD = OD()
    op = outputs()
    diop_diff = 0.5
    min_dist = 25
    num_dist = 8
    dists = calc_perceptually_useful_distances(min_dist, diop_diff, num_dist)

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

    ylabels = ["f1", "d_W_f1", "d_f1_LCoS", "f2", "norm", "I1", "d_WI_f4", "d_OM_f4", "mag_arr", "img_dist"]
    ylim_arr = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1.5, 1.5], [0,-200]]
    output_arrays_resized = 'False'

    for curr_dist in dists:
        dist_index = dists.index(curr_dist)
        IOD.d_vip_eye = curr_dist # Should be > 23
        str = "d_vip_eye = %f" % curr_dist
        custom_prnt(str)

        IOD.populate_d_eye(curr_dist)
        IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
        op.f1_arr[dist_index] = IOD.f1
        sym_f2 = Symbol('f_2')
        IOD.f2 = sym_f2
        IOD.f3 = 5
        IOD.f4 = 5
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()

        II = Matrix([[1,0], [0,1]])
        IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
        S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
        TT = II
        IOD.TT = II

        IOD.calc_TA_diff_TT()
        OO = IOD.OO

        OO_l = OO.tolist()
        flat_OO_l = []
        conv_lol_flat_l(OO_l, flat_OO_l)

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
            str = "f1 = %f D" % (convert_cm2dpt(IOD.f1))
            custom_prnt(str)
            str = "f2 = %f cm" % (IOD.f2)
            custom_prnt(str)
            str = "f2 = %f D" % (convert_cm2dpt(IOD.f2))
            custom_prnt(str)

            # Verify that TA ~= TT
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()
            IOD.calc_TA_diff_TT()
            IOD.calc_OO_norm()

            TA_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TA)
            str = "Actual Transfer matrix:"
            custom_prnt(str)
            custom_prnt(TA_l)

            TT_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.TT)
            str = "Target Transfer matrix:"
            custom_prnt(str)
            custom_prnt(TT_l)

            OO_l = convert_sympy_mutableDenseMatrix_custom_prntableList(IOD.OO)
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
            for ncurr_dist in dists:
                ncurr_dist_index = dists.index(ncurr_dist)
                IOD.populate_d_eye(ncurr_dist)
                IOD.propagate_rw_all(IOD.d_W_f1)
                custom_prnt(IOD.rw_magnification)
                op.mag_arr[dist_index, soln_index, ncurr_dist_index] = IOD.rw_magnification
                op.img_dist[dist_index, soln_index, ncurr_dist_index] = IOD.d_WI_f4
        
    graph_outputs(op, dists, soln_l, outputs_dir, ylabels, ylim_arr)

'''
Modelling the optical system from
Howlett-Smithwick-SID2017-Perspective correct occlusion-capable augmented reality displays using cloaking optics constraints
'''
def main10():
    IOD = OD()
    op = outputs()
    diop_diff = 0.3
    max_dist = convert_m2cm(10)
    num_dist = 15
    dists = calc_perceptually_useful_distances(max_dist, diop_diff, num_dist)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    std_output_arr = np.zeros(num_dist)
    op.mag_arr = np.copy(std_output_arr)
    op.img_dist = np.copy(std_output_arr)

    ylabels = ["mag_arr", "img_dist"]
    ylim_arr = [[-1.5, 1.5], [-1, -1]]

    IOD.f1 = 4
    IOD.f2 = 4
    IOD.f3 = 4
    IOD.f4 = 4

    howlett_d = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
    # howlett_t = howlett_d + 4*IOD.f1
    howlett_t = 4*IOD.f1

    IOD.d_f2_f3 = howlett_t

    IOD.calc_ABCD_matrices()
    IOD.calc_TA()
    II = Matrix([[1,0], [0,1]])
    IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
    S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
    TT = II
    IOD.TT = II

    IOD.calc_TA_diff_TT()
    OO = IOD.OO
    OO_l = OO.tolist()
    flat_OO_l = []
    conv_lol_flat_l(OO_l, flat_OO_l)

    str = "Magnification at all distances:"
    custom_prnt(str)
    diff_l = []
    mag_l = []
    for ncurr_dist in dists:
        ncurr_dist_index = dists.index(ncurr_dist)

        # IOD.populate_d_eye(ncurr_dist)
        IOD.d_W_f1 = ncurr_dist

        IOD.propagate_rw_all(IOD.d_W_f1) # Assume that ncurr_dist = d_W_f1

        diff_dist = IOD.d_WI_f4 + IOD.d_W_f1
        diff_l.append(diff_dist)

        mag_l.append(IOD.rw_magnification)


    mag_l_rounded = [round(elem,2) for elem in mag_l]
    # print(mag_l_rounded)
    mag_arr = np.array(mag_l, dtype=np.float64)

    diff_l_rounded = [round(elem,2) for elem in diff_l]
    # print(diff_l_rounded)
    diff_arr = np.array(diff_l, dtype=np.float64)

    str = 'Avg(mag): %0.2f  Std(mag): %0.2f  Avg(dif): %0.2f  Std(dif): %0.2f' % (np.mean(mag_arr), np.std(mag_arr), np.mean(diff_arr), np.std(diff_arr))
    print(str)
    print('\n')
 
    # soln_l = []
    # graph_outputs(op, dists, soln_l, outputs_dir, ylabels, ylim_arr)

if __name__ == '__main__':
    main5()

