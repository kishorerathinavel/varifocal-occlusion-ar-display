from sympy import *
from common_functions import *
# from sympy.abc import x, y
from numpy.linalg import inv
import numpy as np
import inspect

def simulateHowlett():
    sym_f = Symbol('f')
    Mf = makeLensMatrix(sym_f)

    sym_2f = 2*sym_f
    S2f = makeFreeSpacePropagationMatrix(sym_2f)
    sym_t = Symbol('t')
    St = makeFreeSpacePropagationMatrix(sym_t)

    sym_d = Symbol('d')
    Sd = makeFreeSpacePropagationMatrix(sym_d)

    # Sd = Mf*S2f*Mf*St*Mf2*S2f*Mf
    # Mf2 = (Mf*S2f*Mf*St)^-1 Sd (S2f*Mf)^-1
    I1 = Mf*S2f*Mf*St
    I2 = S2f*Mf
    M2 = I1.inv()*Sd*I2.inv()
    M2_arr = np.array(M2)
    print(M2_arr[0,0])
    print(M2_arr[0,1])
    print(M2_arr[1,0])
    print(M2_arr[1,1])

   
def print_matrix_all_formats(mat):
    formats = ['lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex', 'old']
    # ilex, igrlex, igrevlex is good.
    # ilex is better
    for format in formats:
        print(latex(mat[0,0], order=format))
        # Refer to https://github.com/sympy/sympy/issues/5203
    
def main():
    # b = Symbol('b')
    # mat_A = Matrix([[3,0], [4,1]])
    # mat_B = Matrix([[2,9], [2,b]])
    # print(np.array(mat_A*mat_B))
    # print(np.array(mat_B))
    # print(np.array(mat_B.inv()))

    # simulateHowlett()
    
    sym_f1 = Symbol('F_1^{(t)}')
    M1 = makeLensMatrix(sym_f1)
    sym_f2 = Symbol('F_2^{(t)}')
    M2 = makeLensMatrix(sym_f2)
    sym_f3 = Symbol('F_2^{(t)}')
    M3 = makeLensMatrix(sym_f3)
    sym_f4 = Symbol('F_1^{(t)}')
    M4 = makeLensMatrix(sym_f4)

    sym_d12 = Symbol('d_{12}')
    S12 = makeFreeSpacePropagationMatrix(sym_d12)
    sym_d23 = Symbol('d_{23}')
    S23 = makeFreeSpacePropagationMatrix(sym_d23)
    sym_d34 = Symbol('d_{12}')
    S34 = makeFreeSpacePropagationMatrix(sym_d34)

    II = M4*S34*M3*S23*M2*S12*M1
    print_matrix(II)

    II = Matrix([[1,0], [0,1]])
    I1 = M4*S34*M3*S23
    I2 = S12*M1
    M2 = I1.inv()*II*I2.inv()
    print_matrix(M2)

if __name__ == '__main__':
    main()

