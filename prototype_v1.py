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
        self.f1 = convert_dpt2cm(13)
        self.f4 = convert_dpt2cm(10)

        # Uninitialized
        self.d_W_f1 = 0.0 
        self.d_W_eye = 0.0
        self.d_WI_eye = 0.0
        self.d_OI_eye = 0.0
        self.f2 = 0.0
        self.f3 = 0.0

    def calc_d_W_eye(self):
        self.d_W_eye = self.d_W_f1 + self.d_f1_f2 + self.d_f2_cube + self.d_cube_eye

    def calc_d_W_f1(self):
        self.d_W_f1 = self.d_WI_eye  - self.d_cube_eye - self.d_f2_cube - self.d_f1_f2

    def infocus_mode(self):
        self.calc_d_W_eye()
        self.d_WI_eye = self.d_W_eye
        self.d_OI_eye = self.d_W_eye

def main():
    IOD = OD()
    calibrate_phase1 = False
    calibrate_phase2 = True

    if(calibrate_phase1):
        # Model the case where the display can act as a VR display
        IOD.d_WI_eye = 267 # E-Tech board distance
        I_f4 = (IOD.d_f4_cube + IOD.d_cube_eye) - IOD.d_WI_eye
        O_f4 = calculate_image_distance(I_f4, IOD.f4)
        I_f3 = IOD.d_f3_f4 - O_f4
        O_f3 = IOD.d_LCoS_cube + IOD.d_cube_f3
        IOD.f3 = calculate_focal_length(O_f3, I_f3)
        print("\nValues for E-Tech Board")
        print(I_f4)
        print(O_f4)
        print(I_f3)
        print(O_f3)
        print(IOD.f3)
        print(convert_cm2dpt(IOD.f3))
        print("Should be 12.00")
        # Actual value: 12.00 dpt
        # Simulation value: 8.29 cm
        # Simulation value: 12.07 dpt

        IOD.d_WI_eye = 32 # references distance
        I_f4 = (IOD.d_f4_cube + IOD.d_cube_eye) - IOD.d_WI_eye
        O_f4 = calculate_image_distance(I_f4, IOD.f4)
        I_f3 = IOD.d_f3_f4 - O_f4
        O_f3 = IOD.d_LCoS_cube + IOD.d_cube_f3
        IOD.f3 = calculate_focal_length(O_f3, I_f3)
        print("\nValues for near object")
        print(I_f4)
        print(O_f4)
        print(I_f3)
        print(O_f3)
        print(IOD.f3)
        print(convert_cm2dpt(IOD.f3))
        print("Should be ~6.54")
        # Actual value: 5 to 6 dpt for 28 cm
        # Actual value: ~6.5 dpt for 41 cm
        # Simulation value: 18.58 cm
        # Simulation value: 5.38 dpt

# | Distance to Eye |    f2 |   f3 |
# |             267 |   5.2 |   12 |
# |              33 | 12.04 |  7.0 |

    if(calibrate_phase2):
        # Model the see-through case
        IOD.d_WI_eye = 267 # E-Tech board distance
        IOD.calc_d_W_f1()
        O_f1 = IOD.d_W_f1
        I_f1 = calculate_image_distance(O_f1, IOD.f1)
        O_f2 = IOD.d_f1_f2 - I_f1
        I_f2 = IOD.d_f2_cube + IOD.d_cube_LCoS
        IOD.f2 = calculate_focal_length(O_f2, I_f2)
        print("\nValues for E-Tech Board")
        print(O_f1)
        print(I_f1)
        print(O_f2)
        print(I_f2)
        print(IOD.f2)
        print(convert_cm2dpt(IOD.f2))
        print("Should be 5.2")
        # Actual value: 5.2 dpt
        # Simulation value:  cm
        # Simulation value:  dpt

        IOD.d_WI_eye = 32 # references distance
        IOD.calc_d_W_f1()
        O_f1 = IOD.d_W_f1
        I_f1 = calculate_image_distance(O_f1, IOD.f1)
        O_f2 = IOD.d_f1_f2 - I_f1
        I_f2 = IOD.d_f2_cube + IOD.d_cube_LCoS
        IOD.f2 = calculate_focal_length(O_f2, I_f2)
        print("\nValues for near object")
        print(O_f1)
        print(I_f1)
        print(O_f2)
        print(I_f2)
        print(IOD.f2)
        print(convert_cm2dpt(IOD.f2))
        print("Should be 12.04")
        # Actual value: 12.04 dpt
        # Simulation value:  cm
        # Simulation value:  dpt
    
if __name__ == '__main__':
    main()
    # sys.exit(main())

