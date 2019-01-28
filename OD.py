import common_functions as cf
import numpy as np
from numpy import linalg as LA

'''
Refer to abcd.svg for the diagram of the optical designs modelled here
'''

class lens():
    def __init__(self):
        self.focal_length = 0.0
        self.d_prev_lens = 0.0
        self.d_object = 0.0
        self.d_image = 0.0
        self.magnification = 0.0
        self.tunable = False
        self.M = []
        self.S = []
        
    
class optical_design(): 
    def __init__(self):
        self.lens_l = []
        self.min_d_f_LCoS = 4.0
        self.min_d_f_eye = 2.0
        self.num_lenses = 0
        self.num_lenses_om = 0
        self.d_vip_eye = 0.0
        self.magnification = 0.0

    def populate_focal_lengths(self, f):
        counter = 0
        for lens in self.lens_l:
            if(lens.tunable == True):
                lens.focal_length = f[counter]
                counter = counter + 1

    def propagate_rw_all(self, curr_dist):
        self.magnification = 1.0
        for lens in self.lens_l:
            lens.d_image = cf.calculate_image_distance(lens.d_object, lens.focal_length)
            lens.magnification = -lens.d_image/lens.d_object
            self.magnification = self.magnification * lens.magnification

    def propagate_om(self):
        self.magnification = 1.0
        for lens in self.lens_l[self.num_lenses - self.num_lenses_om:]:
            lens.d_image = cf.calculate_image_distance(lens.d_object, lens.focal_length)
            lens.magnification = -lens.d_image/lens.d_object
            self.magnification = self.magnification * lens.magnification

    def calc_ABCD_matrices(self):
        for lens in self.lens_l:
            lens.M = cf.makeLensMatrix(lens.focal_length)
            lens.S = cf.makeFreeSpacePropagationMatrix(lens.d_prev_lens)

    def calc_TA(self):
        self.TA = Matrix([[1,0], [0,1]])
        for lens in self.lens_l:
            self.TA = self.TA * lens.M * lens.S

    def prototype_v4_populate_dependent_focalLengths(self):
        self.f3 = self.f2
        self.f4 = self.f1

    def calc_TA_diff_TT(self):
        self.OO = self.TT - self.TA

    def calc_OO_norm(self):
        OO_np = np.array(self.OO.tolist()).astype(np.float64)
        self.norm = LA.norm(OO_np)
    
