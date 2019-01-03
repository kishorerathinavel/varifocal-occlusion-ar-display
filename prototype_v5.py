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

def main():
    IOD = OD()
    IOD.d_vip_eye = 15
    IOD.populate_d_eye()
    IOD.f1 = calculate_focal_length(IOD.d_W_f1, IOD.d_f1_LCoS)
    print(IOD.f1)
    M1 = makeLensMatrix(1/IOD.f1)
    M4 = M1
    S12 = makeFreeSpacePropagationMatrix(IOD.d_f1_f2)
    S34 = S12
    S23 = makeFreeSpacePropagationMatrix(IOD.d_f2_f3)
    II = Matrix([[1,0], [0,1]])
    sym_f2 = Symbol('F_2')
    M2 = makeLensMatrix(1/sym_f2)
    M3 = M2
    OO = II - M4*S34*M3*S23*M2*S12*M1
    TT = M4*S34*M3*S23*M2*S12*M1

    soln_l = []
    curr_soln = solve(OO[0,0])
    soln_l.extend(curr_soln)
    curr_soln = solve(OO[0,1])
    soln_l.extend(curr_soln)
    curr_soln = solve(OO[1,0])
    soln_l.extend(curr_soln)
    curr_soln = solve(OO[1,1])
    soln_l.extend(curr_soln)

    rounded_soln_l = [round(elem,2) for elem in soln_l]
    unique_soln_l = list(set(rounded_soln_l))
    # print(soln_l)
    # print(rounded_soln_l)
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
    print(rounded_norm_l)
    
if __name__ == '__main__':
    main()

