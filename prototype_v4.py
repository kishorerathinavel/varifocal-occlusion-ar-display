from common_functions import *
from sympy import *
import numpy as np
from numpy import linalg as LA
#from sympy.core.symbol import symbols
#from sympy.solvers.solveset import nonlinsolve

class OD(): # Short for Implemented Optical Design
    def __init__(self):
        self.d_f1_LCoS = 4.0 # Minimum possible
        self.d_LCoS_f2 = 4.0 # Minimum possible
        self.d_f1_f2 = self.d_f1_LCoS + self.d_LCoS_f2
        self.d_f2_f3 = 3.0 # Guess
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

    def populate_d_eye(self):
        self.d_W_eye = self.d_vip_eye
        self.d_W_f1 = self.d_W_eye - self.d_f4_eye - self.d_f3_f4 - self.d_f2_f3 - self.d_f1_f2

    def propagate_rw_all(self, ncurr_dist):
        self.d_vip_eye = ncurr_dist
        self.populate_d_eye()
        self.O1 = self.d_W_f1
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
    print(nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y]))

def convert_sympy_mutableDenseMatrix_printableList(denseMatrix):
    denseMatrix_np = np.array(denseMatrix.tolist()).astype(np.float64)
    r_denseMatrix_np = np.round(denseMatrix_np, 2)
    denseMatrix_l = r_denseMatrix_np.tolist()
    return denseMatrix_l

def calc_perceptually_useful_distances(min_dist, diop_diff, num_dist):
    # Assuming that min_dist is specified in cm
    min_diop_dist = convert_cm2dpt(min_dist)
    prev_diop_dist = min_diop_dist
    dists = []
    dists.append(min_dist)
    for iter in range(2,num_dist):
        next_diop_dist = prev_diop_dist - diop_diff
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

def main5():
    IOD = OD()
    op = outputs()
    diop_diff = 0.5
    min_dist = 25
    num_dist = 9
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
        print(str)

        IOD.populate_d_eye()
        IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
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
        IOD.TT = S14

        IOD.calc_TA_diff_TT()
        OO = IOD.OO

        # print_matrix(OO)
        OO_l = OO.tolist()
        flat_OO_l = []
        conv_lol_flat_l(OO_l, flat_OO_l)

        # Getting solutions for all equations together
        soln_l = list(nonlinsolve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f2, sym_f3]))
        print("Solutions")
        print(soln_l)
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
        # print(type(rounded_norm_l))
        print("Norm of solutions")
        print(rounded_norm_l)
        # END Get the norm of OO = TT - TA for each solution

        # Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices
        num_soln = len(soln_l)
        if(prev_num_soln < num_soln):
            num_new_colns = num_soln - prev_num_soln
            new_cols = np.zeros((num_dist, num_new_colns))
            op.f2_arr = np.hstack((op.f2_arr, new_cols))
            op.f3_arr = np.hstack((op.f2_arr, new_cols))
            op.norm_arr = np.hstack((op.norm_arr, new_cols))
            op.I1_arr = np.hstack((op.I1_arr, new_cols))
            op.d_f1_LCoS_arr = np.hstack((op.d_f1_LCoS_arr, new_cols))
            op.d_WI_f4_arr = np.hstack((op.d_WI_f4_arr, new_cols))
            op.d_OM_f4_arr = np.hstack((op.d_OM_f4_arr, new_cols))
            op.d_W_f1_arr = np.hstack((op.d_W_f1_arr, new_cols))
            op.mag_arr = np.hstack((op.mag_arr, new_cols))
        # END Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices

        for curr_soln in soln_l:
            soln_index = soln_l.index(curr_soln)
            IOD.f2 = curr_soln[0]
            IOD.f3 = curr_soln[1]
            op.f2_arr[dist_index, soln_index] = IOD.f2
            op.f3_arr[dist_index, soln_index] = IOD.f3

            print("\n")
            str = "f1 = %f cm" % (IOD.f1)
            print(str)
            str = "f1 = %f D" % (convert_cm2dpt(IOD.f1))
            print(str)
            str = "f2 = %f cm" % (IOD.f2)
            print(str)
            str = "f2 = %f D" % (convert_cm2dpt(IOD.f2))
            print(str)
            str = "f3 = %f cm" % (IOD.f3)
            print(str)
            str = "f3 = %f D" % (convert_cm2dpt(IOD.f3))
            print(str)

            # Verify that TA ~= TT
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()
            IOD.calc_TA_diff_TT()
            IOD.calc_OO_norm()

            TA_l = convert_sympy_mutableDenseMatrix_printableList(IOD.TA)
            str = "Actual Transfer matrix:"
            print(str)
            print(TA_l)

            TT_l = convert_sympy_mutableDenseMatrix_printableList(IOD.TT)
            str = "Target Transfer matrix:"
            print(str)
            print(TT_l)

            OO_l = convert_sympy_mutableDenseMatrix_printableList(IOD.OO)
            str = "residual matrix"
            print(str)
            print(OO_l)
            rounded_norm = round(IOD.norm, 2)
            op.norm_arr[dist_index, soln_index] = rounded_norm
            str = "norm = %f" % (rounded_norm)
            print(str)
            # END Verify that TT ~= TA

            # Calculate where image of real world is formed
            # Calculate d_WI_f4
            IOD.propagate_rw_all(IOD.d_vip_eye)
            # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            op.I1_arr[dist_index, soln_index] = IOD.I1
            op.d_f1_LCoS_arr[dist_index, soln_index] = IOD.d_f1_LCoS
            # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            str = "d_WI_f4 = %f" % (IOD.d_WI_f4)
            print(str)
            op.d_WI_f4_arr[dist_index, soln_index] = IOD.d_WI_f4
            # END Calculate where image of real world is formed

            op.mag_arr[dist_index, soln_index] = IOD.rw_magnification

            # Calculate where image of occlusion mask
            if (abs(IOD.I1 - IOD.d_f1_LCoS) < 0.01):
                print("No need to propagate OM separately because rw at vip formed at LCoS")
                IOD.d_OM_f4 = IOD.d_WI_f4
            else:
                print("RW at vip did not form at LCoS")
                IOD.propagate_om()
            # END Calculate where image of occlusion mask
            str = "d_OM_f4 = %f" %(IOD.d_OM_f4)
            print(str)
            str = "d_W_f1 = %f" % (-IOD.d_W_f1)
            print(str)

            op.d_OM_f4_arr[dist_index, soln_index] = IOD.d_OM_f4
            op.d_W_f1_arr[dist_index, soln_index] = -IOD.d_W_f1

            str = "Magnification at all distances:"
            print(str)
            for ncurr_dist in dists:
                IOD.propagate_rw_all(ncurr_dist)
                # print(IOD.d_W_f1)
                # print(IOD.d_WI_f4)
                print(IOD.rw_magnification)
        
            print("-----------------")

def main4():
    IOD = OD()
    op = outputs()
    diop_diff = 0.5
    min_dist = 25
    num_dist = 9
    dists = calc_perceptually_useful_distances(min_dist, diop_diff, num_dist)

    # Assume that num_solns = 2
    # All output matrices have num_dist rows and num_soln columns
    num_soln = 2
    prev_num_soln = 2
    std_output_arr = np.zeros((num_dist, num_soln))
    op.f2_arr = np.copy(std_output_arr)
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
        print(str)

        IOD.populate_d_eye()
        IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
        sym_f2 = Symbol('f_2')
        IOD.f2 = sym_f2
        IOD.prototype_v4_populate_dependent_focalLengths()
        IOD.calc_ABCD_matrices()
        IOD.calc_TA()

        II = Matrix([[1,0], [0,1]])
        IOD.d_f1_f4 = IOD.d_f1_f2 + IOD.d_f2_f3 + IOD.d_f3_f4
        S14 = makeFreeSpacePropagationMatrix(IOD.d_f1_f4)
        TT = S14
        IOD.TT = S14

        IOD.calc_TA_diff_TT()
        OO = IOD.OO

        # print_matrix(OO)
        OO_l = OO.tolist()
        flat_OO_l = []
        conv_lol_flat_l(OO_l, flat_OO_l)

        # Getting solutions for all equations together
        soln_l = list(nonlinsolve([OO[0,1], OO[0,0], OO[1,0], OO[1,1]], [sym_f2]))
        # END Getting solutions for all equations together

        # Getting solutions for each equation separately
        # soln_l = []
        # for eqn in flat_OO_l:
        #     curr_soln = nonlinsolve([eqn], [sym_f2])
        #     soln_l.append(list(curr_soln))
        # print(soln_l)
        # END Getting solutions for each equation separately

        # Converting from sympy set,tuple to python list
        soln_l2 = []
        conv_lol_flat_l(soln_l, soln_l2)
        # END Converting from sympy set to python list

        # Extracting unique solutions
        rounded_soln_l = [round(elem,2) for elem in soln_l2]
        unique_soln_l = list(set(rounded_soln_l))
        # print(soln_l)
        # print(rounded_soln_l)
        # print(type(unique_soln_l))
        print("Unique solutions")
        print(unique_soln_l)
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
        # print(type(rounded_norm_l))
        print("Norm of solutions")
        print(rounded_norm_l)
        # END Get the norm of OO = TT - TA for each solution

        # Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices
        num_soln = len(unique_soln_l)
        if(prev_num_soln < num_soln):
            num_new_colns = num_soln - prev_num_soln
            new_cols = np.zeros((num_dist, num_new_colns))
            op.f2_arr = np.hstack((op.f2_arr, new_cols))
            op.norm_arr = np.hstack((op.norm_arr, new_cols))
            op.I1_arr = np.hstack((op.I1_arr, new_cols))
            op.d_f1_LCoS_arr = np.hstack((op.d_f1_LCoS_arr, new_cols))
            op.d_WI_f4_arr = np.hstack((op.d_WI_f4_arr, new_cols))
            op.d_OM_f4_arr = np.hstack((op.d_OM_f4_arr, new_cols))
            op.d_W_f1_arr = np.hstack((op.d_W_f1_arr, new_cols))
            op.mag_arr = np.hstack((op.mag_arr, new_cols))
        # END Check if number of solutions is more than previously assumed num_solns. If yes, expand all matrices

        for curr_soln in unique_soln_l:
            if(curr_soln < 4):
                continue
            
            soln_index = unique_soln_l.index(curr_soln)

            print("\n")
            str = "f1 = %f cm" % (IOD.f1)
            print(str)
            str = "f1 = %f D" % (convert_cm2dpt(IOD.f1))
            print(str)
            str = "f2 = %f cm" % (curr_soln)
            print(str)
            str = "f2 = %f D" % (convert_cm2dpt(curr_soln))
            print(str)

            op.f2_arr[dist_index, soln_index] = curr_soln

            IOD.f2 = curr_soln
            IOD.f3 = IOD.f2
            IOD.f4 = IOD.f1

            # Verify that TA ~= TT
            IOD.calc_ABCD_matrices()
            IOD.calc_TA()
            IOD.calc_TA_diff_TT()
            IOD.calc_OO_norm()

            TA_l = convert_sympy_mutableDenseMatrix_printableList(IOD.TA)
            str = "Actual Transfer matrix:"
            print(str)
            print(TA_l)

            TT_l = convert_sympy_mutableDenseMatrix_printableList(IOD.TT)
            str = "Target Transfer matrix:"
            print(str)
            print(TT_l)

            OO_l = convert_sympy_mutableDenseMatrix_printableList(IOD.OO)
            str = "residual matrix"
            print(str)
            print(OO_l)
            rounded_norm = round(IOD.norm, 2)
            op.norm_arr[dist_index, soln_index] = rounded_norm
            str = "norm = %f" % (rounded_norm)
            print(str)
            # END Verify that TT ~= TA

            # Calculate where image of real world is formed
            # Calculate d_WI_f4
            IOD.propagate_rw_all(IOD.d_vip_eye)
            # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            op.I1_arr[dist_index, soln_index] = IOD.I1
            op.d_f1_LCoS_arr[dist_index, soln_index] = IOD.d_f1_LCoS
            # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            str = "d_WI_f4 = %f" % (IOD.d_WI_f4)
            print(str)
            op.d_WI_f4_arr[dist_index, soln_index] = IOD.d_WI_f4
            # END Calculate where image of real world is formed

            op.mag_arr[dist_index, soln_index] = IOD.rw_magnification

            # Calculate where image of occlusion mask
            if (abs(IOD.I1 - IOD.d_f1_LCoS) < 0.01):
                print("No need to propagate OM separately because rw at vip formed at LCoS")
                IOD.d_OM_f4 = IOD.d_WI_f4
            else:
                print("RW at vip did not form at LCoS")
                IOD.propagate_om()
            # END Calculate where image of occlusion mask
            str = "d_OM_f4 = %f" %(IOD.d_OM_f4)
            print(str)
            str = "d_W_f1 = %f" % (-IOD.d_W_f1)
            print(str)

            op.d_OM_f4_arr[dist_index, soln_index] = IOD.d_OM_f4
            op.d_W_f1_arr[dist_index, soln_index] = -IOD.d_W_f1

            str = "Magnification at all distances:"
            print(str)
            for ncurr_dist in dists:
                IOD.propagate_rw_all(ncurr_dist)
                # print(IOD.d_W_f1)
                # print(IOD.d_WI_f4)
                print(IOD.rw_magnification)
        
            print("-----------------")
    print("End of code")
        # Calculate d_WI_eye for W at d_vip_eye
        # Calculate magnification
    
if __name__ == '__main__':
    main5()

