from sympy import *
from numpy.linalg import inv
import numpy as np

b = Symbol('b')
mat_A = Matrix([[3,0], [4,1]])
mat_B = Matrix([[2,9], [2,b]])
print(np.array(mat_A*mat_B))
print(np.array(mat_B))
print(np.array(mat_B.inv()))
