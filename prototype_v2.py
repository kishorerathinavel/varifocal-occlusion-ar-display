import numpy as np
import sys
from common_functions import *

# All values are in centimeters
class OD(): # Short for Implemented Optical Design
    def __init__(self):
        self.d_f1_f2 = 2.9 # Measured. Might have estimated the center of f2 wrongly
        self.d_f2_cube = 2.0 # Measured. Might have estimated the center of f2 wrongly. 
        self.d_cube_LCoS = 2.0 # Measured. Parallax error possible.
        self.d_LCoS_cube = self.d_cube_LCoS 
        self.d_cube_f3 = self.d_f2_cube
        self.d_f3_f4 = 1.9 # Measured. Parallax error possible.
        self.d_f4_cube = 3.2 # Measured. Parallax error possible.
        self.d_cube_eye = 1.5 # Guestimate.
        self.f1 = convert_dpt2cm(15)
        self.f4 = convert_dpt2cm(10)

        # Uninitialized
        self.f2 = 0.0
        self.f3 = 0.0
        self.d_W_f1 = 0.0 
        self.d_W_eye = 0.0
        self.d_WI_eye = 0.0
        self.d_OI_eye = 0.0
        self.magnification = 0.0

    def calc_d_W_eye(self):
        self.d_W_eye = self.d_W_f1 + self.d_f1_f2 + self.d_f2_cube + self.d_cube_eye

    def calc_d_W_f1(self):
        self.d_W_f1 = self.d_W_eye  - self.d_cube_eye - self.d_f2_cube - self.d_f1_f2

    def infocus_mode(self):
        self.calc_d_W_eye()
        self.d_WI_eye = self.d_W_eye
        self.d_OI_eye = self.d_W_eye

    def calc_f3(self):
        I_f4 = (self.d_f4_cube + self.d_cube_eye) - self.d_WI_eye
        O_f4 = calculate_image_distance(I_f4, self.f4)
        I_f3 = self.d_f3_f4 - O_f4
        O_f3 = self.d_LCoS_cube + self.d_cube_f3
        self.f3 = calculate_focal_length(O_f3, I_f3)

    def calc_f2(self):
        self.calc_d_W_f1()
        O_f1 = self.d_W_f1
        I_f1 = calculate_image_distance(O_f1, self.f1)
        O_f2 = self.d_f1_f2 - I_f1
        I_f2 = self.d_f2_cube + self.d_cube_LCoS
        self.f2 = calculate_focal_length(O_f2, I_f2)

    def calc_magnification(self):
        self.magnification = 1.0
        
def main():
    IOD = OD()
    dist_arr = [30, 100, 270]
    magnification_arr = []
    for om_dist in dist_arr:
        IOD.d_WI_eye = om_dist
        IOD.calc_f3()
        for rw_dist in dist_arr:
            IOD.d_W_eye = rw_dist
            IOD.calc_d_W_f1()
            IOD.calc_f2()
            IOD.calc_magnification()
            line = [IOD.f3, IOD.f2, IOD.magnification]
            magnification_arr.append(line)
    magnification_array = np.array(magnification_arr)
    print(magnification_array)
if __name__ == '__main__':
    main()

