import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
# from scipy.optimize import shgo # Need scipy=1.2.0 for this
# from scipy.optimize import dual_annealing # Need scipy=1.2.0 for this
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1
import common_functions as cf

'''
Need to minimize:
1. Error in magnification of real world objects
2. Error in depth difference for real world objects

Assume:
fixed d_f1_f2 = 0

Arguements to optimize:
f1, f2

Inputs to energy function:
f1, f2, d_f1_f2

Outpus of energy function:
norm of the vector of errors mentioned above
'''

def propagate(f, d_f1_f2, dist):
    O1 = dist
    I1 = cf.calculate_image_distance(O1, f[0])
    m1 = -I1/O1
    O2 = d_f1_f2 - I1
    I2 = cf.calculate_image_distance(O2, f[1])
    m2 = -I2/O2
    mT = m1*m2
    curr_err_dist = dist + I2
    curr_err_mag = mT - 1.0
    return (curr_err_dist, curr_err_mag)

''' Propagate using Lensmaker's equations '''
def energy_function(f, d_f1_f2): 
    dists_l = [1000.0, 250.0, 64.0, 16.0]
    err_mag_l = []
    err_dist_l = []
    for dist in dists_l:
        [curr_err_dist, curr_err_mag] = propagate(f, d_f1_f2, dist)
        err_dist_l.append(curr_err_dist)
        err_mag_l.append(curr_err_mag)

    err_dist_arr = np.squeeze(np.array(err_dist_l))
    err_mag_arr  = np.squeeze(np.array(err_mag_l))

    energy_dist = np.dot(err_dist_arr, err_dist_arr)/len(err_dist_l)
    energy_mag = np.dot(err_mag_arr, err_mag_arr)/len(err_mag_l)

    combined_energy = energy_dist + energy_mag
    return combined_energy

def using_nelder_mead():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        res = minimize(energy_function, f0, args=(d_f1_f2), method='nelder-mead', options={'xtol':1e-5, 'disp':False})
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))
    
def using_trust_constr():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        bounds = Bounds([-10, -10], [10, 10])

        linear_constraint = LinearConstraint([[1, 0],[0, 1]], [-10, -10], [10, 10])
        # Using the insight that f[0] = -f[1]
        # linear_constraint = LinearConstraint([[1, 1],[1, 0],[0, 1]], [0, -10, -10], [0, 10, 10])

        res = minimize(energy_function, f0, args=(d_f1_f2), method = 'trust-constr', jac='2-point', hess=SR1(), constraints=linear_constraint, options={'verbose':False, 'max_trust_radius':2.0},bounds=bounds)
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))

def using_differential_evolution():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        res = differential_evolution(energy_function, bounds=[(-10.0, 10.0), (-10.0, 10.0)], args=[d_f1_f2])
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))

class MyBounds(object):
    def __init__(self, xmax=[10.0,10.0], xmin=[-10.0,-10.0] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def using_basinhopping():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        mybounds = MyBounds()
        res = basinhopping(energy_function, f0, minimizer_kwargs={'args':[d_f1_f2]}, accept_test=mybounds)
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))

def using_shgo():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        res = shgo(energy_function, bounds=[(-10.0, 10.0), (-10.0, 10.0)], args=[d_f1_f2])
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))

def using_dual_annealing():
    for trial in range(0, 10):
        f0 = np.squeeze(10*(np.random.rand(2,1) - 0.5*np.ones((2,1))))
        if(trial == 9):
            f0 = [5.0, -6.1]
        d_f1_f2 = 0.0
        res = dual_annealing(energy_function, bounds=[(-10.0, 10.0), (-10.0, 10.0)], args=[d_f1_f2])
        [curr_err_dist, curr_err_mag] = propagate(res.x, d_f1_f2, 1000.0)
        print('f0: %6.2f %6.2f | f: %6.2f %6.2f | err: %6.2f %6.2f'% (f0[0], f0[1], res.x[0], res.x[1], curr_err_dist, curr_err_mag))
    

if __name__ == '__main__':
    # using_trust_constr()
    # using_nelder_mead()
    using_differential_evolution()
    # using_basinhopping()
    # using_shgo()
    # using_dual_annealing()

