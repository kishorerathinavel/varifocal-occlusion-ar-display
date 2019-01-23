import common_functions as cf
import numpy as np
from numpy import linalg as LA

'''
Refer to abcd.svg for the diagram of the optical designs modelled here
'''

class optical_design(): 
    def __init__(self):
        self.d_f1_LCoS = 4.0 # Minimum possible
        self.d_LCoS_f2 = 4.0 # Minimum possible
        self.d_f1_f2 = self.d_f1_LCoS + self.d_LCoS_f2
        self.d_f2_f3 = 12 # Guess
        self.d_f3_f4 = self.d_f1_f2
        self.d_f4_eye = 2.0
        self.num_lenses = 0
        self.include_all = False

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

    def populate_d_eye(self, curr_dist):
        self.d_W_eye = curr_dist
        self.d_W_f1 = self.d_W_eye - self.d_f4_eye - self.d_f3_f4 - self.d_f2_f3 - self.d_f1_f2

    def populate_focal_lengths(self, f):
        common_f2_f3 = 3.43
        if(self.num_lenses == 2):
            self.f1 = f[0]
            self.f2 = common_f2_f3
            if(self.num_lenses > 2 or self.include_all == True):
                self.f3 = common_f2_f3
            if(self.num_lenses > 3 or self.include_all == True):
                self.f4 = f[1]
        elif(self.num_lenses == 3):
            self.f1 = f[0]
            self.f2 = f[1]
            if(self.num_lenses > 2 or self.include_all == True):
                self.f3 = f[1]
            if(self.num_lenses > 3 or self.include_all == True):
                self.f4 = f[2]

    def propagate_rw_all(self, curr_dist):
        self.rw_magnification = 1.0
        self.O1 = curr_dist
        self.I1 = cf.calculate_image_distance(self.O1, self.f1)
        self.m1 = -self.I1/self.O1
        self.rw_magnification = self.rw_magnification * self.m1
        self.O2 = self.d_f1_f2 - self.I1
        self.I2 = cf.calculate_image_distance(self.O2, self.f2)
        self.final_I = self.I2
        self.m2 = -self.I2/self.O2
        self.rw_magnification = self.rw_magnification * self.m2
        if(self.num_lenses > 2 or self.include_all == True):
            self.O3 = self.d_f2_f3 - self.I2
            self.I3 = cf.calculate_image_distance(self.O3, self.f3)
            self.final_I = self.I3
            self.m3 = -self.I3/self.O3
            self.rw_magnification = self.rw_magnification *self.m3
            if(self.num_lenses > 3 or self.include_all == True):
                self.O4 = self.d_f3_f4 - self.I3
                self.I4 = cf.calculate_image_distance(self.O4, self.f4)
                self.final_I = self.I4
                self.m4 = -self.I4/self.O4
                self.rw_magnification = self.rw_magnification *self.m4
        # self.rw_magnification = self.m1*self.m2*self.m3*self.m4
        # self.d_WI_f4 = self.I4
        # self.d_WI_f1 = self.d_WI_f4 + self.d_f1_f2 + self.d_f2_f3 + self.d_f3_f4

    def propagate_om(self):
        self.O2 = self.d_LCoS_f2
        self.I2 = cf.calculate_image_distance(self.O2, self.f2)
        self.final_I = self.I2
        if(self.num_lenses > 2 or self.include_all):
            self.O3 = self.d_f2_f3 - self.I2
            self.I3 = cf.calculate_image_distance(self.O3, self.f3)
            self.final_I = self.I3
            if(self.num_lenses > 3 or self.include_all == True):
                self.O4 = self.d_f3_f4 - self.I3
                self.I4 = cf.calculate_image_distance(self.O4, self.f4)
                self.final_I = self.I4

    def calc_ABCD_matrices(self):
        self.M1 = cf.makeLensMatrix(self.f1)
        self.M2 = cf.makeLensMatrix(self.f2)
        self.M3 = cf.makeLensMatrix(self.f3)
        self.M4 = cf.makeLensMatrix(self.f4)
        self.S12 = cf.makeFreeSpacePropagationMatrix(self.d_f1_f2)
        self.S23 = cf.makeFreeSpacePropagationMatrix(self.d_f2_f3)
        self.S34 = cf.makeFreeSpacePropagationMatrix(self.d_f3_f4)

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
    
