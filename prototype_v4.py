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
        self.d_f2_f3 = 10.0 # Guess
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

def nomain():
    x, y, z = symbols('x, y, z', real=True)
    print(nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y]))

def main4():
    IOD4 = OD()
    IOD4.d_vip_eye = 15
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
    OO = II - M4*S34*M3*S23*M2*S12*M1
    TT = M4*S34*M3*S23*M2*S12*M1

    # Not correct to solve equation by equation like this.
    soln_l = nonlinsolve([OO[0,0], OO[0,1], OO[1,0], OO[1,1]], [sym_f2])
    soln_l2 = []
    for elem in soln_l:
        elem2 = elem[0]
        soln_l2.append(elem2)
    rounded_soln_l = [round(elem,2) for elem in soln_l2]
    unique_soln_l = list(set(rounded_soln_l))
    # print(soln_l)
    # print(rounded_soln_l)
    # print(type(unique_soln_l))
    print(unique_soln_l)
    norm_l = []
    for curr_soln in unique_soln_l:
        M2 = makeLensMatrix(1/curr_soln)
        M3 = M2
        OO = II - M4*S34*M3*S23*M2*S12*M1
        OO2 = np.array(OO.tolist()).astype(np.float64)
        # print_matrix(OO2)
        curr_norm = LA.norm(OO2)
        norm_l.extend([curr_norm])
    rounded_norm_l = [round(elem,2) for elem in norm_l]
    # print(type(rounded_norm_l))
    print(rounded_norm_l)
    min_index = rounded_norm_l.index(min(rounded_norm_l))
    IOD4.f2 = unique_soln_l[min_index]
    IOD4.f3 = IOD4.f2
    IOD4.f4 = IOD4.f1

    # Calculate d_OM_eye
    O2 = IOD4.d_LCoS_f2
    I2 = calculate_image_distance(O2, IOD4.f2)
    O3 = IOD4.d_f2_f3 - I2
    I3 = calculate_image_distance(O3, IOD4.f3)
    O4 = IOD4.d_f3_f4 - I3
    I4 = calculate_image_distance(O4, IOD4.f4)
    IOD4.d_OM_eye = I4 + IOD4.d_f4_eye
    print(IOD4.d_vip_eye)
    print(IOD4.d_OM_eye)

    # Calculate d_WI_eye for W at d_vip_eye
    # Calculate magnification
    
if __name__ == '__main__':
    main4()

