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
        self.d_f4_eye = 4.0

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

class outputs():
    pass

def nomain():
    x, y, z = symbols('x, y, z', real=True)
    print(nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y]))

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

def main4():
    IOD4 = OD()
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
        IOD4.d_vip_eye = curr_dist # Should be > 23
        str = "d_vip_eye = %f" % curr_dist
        print(str)

        IOD4.populate_d_eye()
        IOD4.f1 = calculate_focal_length(IOD4.d_W_f1, IOD4.d_f1_LCoS)
        M1 = makeLensMatrix(1/IOD4.f1)
        M4 = M1
        S12 = makeFreeSpacePropagationMatrix(IOD4.d_f1_f2)
        S34 = S12
        S23 = makeFreeSpacePropagationMatrix(IOD4.d_f2_f3)
        II = Matrix([[1,0], [0,1]])
        sym_f2 = Symbol('f_2')
        M2 = makeLensMatrix(1/sym_f2)
        M3 = M2
        TA = M4*S34*M3*S23*M2*S12*M1
        IOD4.d_f1_f4 = IOD4.d_f1_f2 + IOD4.d_f2_f3 + IOD4.d_f3_f4
        S14 = makeFreeSpacePropagationMatrix(IOD4.d_f1_f4)
        TT = S14
        OO = TT - TA

        print_matrix(OO)
        OO_l = OO.tolist()
        flat_OO_l = []
        conv_lol_flat_l(OO_l, flat_OO_l)
        soln_l = []
        for eqn in flat_OO_l:
            curr_soln = nonlinsolve([eqn], [sym_f2])
            soln_l.append(list(curr_soln))

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
            M2 = makeLensMatrix(1/curr_soln)
            M3 = M2
            TA = M4*S34*M3*S23*M2*S12*M1
            OO = TT - TA
            OO2 = np.array(OO.tolist()).astype(np.float64)
            curr_norm = LA.norm(OO2)
            norm_l.append(curr_norm)
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
            soln_index = unique_soln_l.index(curr_soln)

            print("\n")
            str = "f2 = %f" % (curr_soln)
            print(str)

            op.f2_arr[dist_index, soln_index] = curr_soln

            IOD4.f2 = curr_soln
            IOD4.f3 = IOD4.f2
            IOD4.f4 = IOD4.f1

            # Verify that TA ~= TT
            M1 = makeLensMatrix(1/IOD4.f1)
            M2 = makeLensMatrix(1/IOD4.f2)
            M3 = makeLensMatrix(1/IOD4.f3)
            M4 = makeLensMatrix(1/IOD4.f4)
            S12 = makeFreeSpacePropagationMatrix(IOD4.d_f1_f2)
            S23 = makeFreeSpacePropagationMatrix(IOD4.d_f2_f3)
            S34 = makeFreeSpacePropagationMatrix(IOD4.d_f3_f4)
            TA = M4*S34*M3*S23*M2*S12*M1

            str = "Actual Transfer matrix:"
            print(str)
            TA_np = np.array(TA.tolist()).astype(np.float64)
            r_TA_np = np.round(TA_np, 2)
            TA_l = r_TA_np.tolist()
            print(TA_l)

            str = "Target Transfer matrix:"
            print(str)
            TT_np = np.array(TT.tolist()).astype(np.float64)
            r_TT_np = np.round(TT_np, 2)
            TT_l = r_TT_np.tolist()
            print(TT_l)

            OO = TT - TA
            OO_np = np.array(OO.tolist()).astype(np.float64)
            r_OO_np = np.round(OO_np, 2)
            OO_l = r_OO_np.tolist()
            str = "residual matrix"
            print(str)
            print(OO_l)
            curr_norm = LA.norm(OO_np)
            rounded_norm = round(curr_norm, 2)
            op.norm_arr[dist_index, soln_index] = rounded_norm
            str = "norm = %f" % (rounded_norm)
            print(str)
            # END Verify that TT ~= TA

            # Calculate where image of real world is formed
            # Calculate d_WI_f4
            O1 = IOD4.d_W_f1
            I1 = calculate_image_distance(O1, IOD4.f1)
            m1 = I1/O1
            # Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            str = "I1 = %f" %(I1)
            print(str)
            str = "d_f1_LCoS = %f" % (IOD4.d_f1_LCoS)
            print(str)

            op.I1_arr[dist_index, soln_index] = I1
            op.d_f1_LCoS_arr[dist_index, soln_index] = IOD4.d_f1_LCoS
            # END Verify that the image of real world at d_W_f1 is coming to focus at the LCoS
            O2 = IOD4.d_f1_f2 - I1
            I2 = calculate_image_distance(O2, IOD4.f2)
            m2 = I2/O2
            O3 = IOD4.d_f2_f3 - I2
            I3 = calculate_image_distance(O3, IOD4.f3)
            m3 = I3/O3
            O4 = IOD4.d_f3_f4 - I3
            I4 = calculate_image_distance(O4, IOD4.f4)
            m4 = I4/O4
            str = "d_WI_f4 = %f" % (I4)
            print(str)
            op.d_WI_f4_arr[dist_index, soln_index] = I4
            # END Calculate where image of real world is formed

            mT = m1*m2*m3*m4
            op.mag_arr[dist_index, soln_index] = mT
            
            # Calculate where image of occlusion mask
            # Calculate d_OM_f4
            O2 = IOD4.d_LCoS_f2
            I2 = calculate_image_distance(O2, IOD4.f2)
            O3 = IOD4.d_f2_f3 - I2
            I3 = calculate_image_distance(O3, IOD4.f3)
            O4 = IOD4.d_f3_f4 - I3
            I4 = calculate_image_distance(O4, IOD4.f4)
            IOD4.d_OM_f4 = I4
            str = "d_OM_f4 = %f" %(IOD4.d_OM_f4)
            print(str)
            str = "d_W_f1 = %f" % (-IOD4.d_W_f1)
            print(str)
            str = "magnification = %f" % (mT)
            print(str)

            op.d_OM_f4_arr[dist_index, soln_index] = IOD4.d_OM_f4
            op.d_W_f1_arr[dist_index, soln_index] = -IOD4.d_W_f1
            # END Calculate where image of occlusion mask

            print("-----------------")
    print("End of code")
        # Calculate d_WI_eye for W at d_vip_eye
        # Calculate magnification
    
if __name__ == '__main__':
    main4()

