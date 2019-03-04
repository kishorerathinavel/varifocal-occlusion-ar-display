#!/usr/bin/env python

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox,
                             QVBoxLayout, QSlider, QSpinBox, QWidget, QLabel)
from lens import Lens
import sys
import time
import winsound


# f1_l = [5.10, 5.40, 6.56, 8.12]
# f4_l = [11.26, 10.96, 9.80, 8.24]

#f1_orig = [5.00, 7.39]
f1_orig = [18.0, 11.8]
f4_orig = [4.8, 6.9]

f1_l = f1_orig.copy()
f4_l = f4_orig.copy()

offset = 0.0
beep_frequency = 2500
beep_duration = 100

# f1_l = [5.10, 8.12]
# f4_l = [11.26, 8.24]

index = 1

lens1 = Lens('COM9', debug=False)
lens4 = Lens('COM8', debug=False)

def calc_f4_l(offset):
    global f4_l
    f4_l = [f4_orig[0] + offset, f4_orig[1] - offset]
    print(f4_l)
    
def connect_lenses():
    global lens1, lens2

    print('Lens 1:')
    print('Lens serial number:', lens1.lens_serial)
    print('Lens temperature:', lens1.get_temperature())
    min_fp, max_fp = lens1.to_focal_power_mode()
    print('Minimal diopter:', min_fp)
    print('Maximum diopter:', max_fp)
    print(lens1.set_temperature_limits(20,45))
    lens1.set_diopter(f1_l[index])

    print('Lens 1:')
    print('Lens serial number:', lens4.lens_serial)
    print('Lens temperature:', lens4.get_temperature())
    min_fp, max_fp = lens4.to_focal_power_mode()
    print('Minimal diopter:', min_fp)
    print('Maximum diopter:', max_fp)
    print(lens4.set_temperature_limits(20,45))
    lens4.set_diopter(f4_l[index])

def update_lens(f1=True,f2=True):
    global lens1, lens2
    str = ''
    if(f1 == True):
        lens1.set_diopter(f1_l[index])
        str = '%s L1[%d]:%0.2f' %(str, index, f1_l[index])
    if(f2 == True):
        lens4.set_diopter(f4_l[index])
        str = '%s L4[%d]:%0.2f' %(str, index, f4_l[index])
    print('%s' % (str))
    
def increment_index(f1=True, f2=True):
    global index
    index = index + 1
    if(index > len(f1_l) - 1):
        index = len(f1_l) - 1
    update_lens(f1,f2)
    
def decrement_index(f1=True, f2=True):
    global index
    index = index - 1
    if(index < 0):
        index = 0
    update_lens(f1,f2)

def set_fl_middle():
    global f1_l, f4_l
    f1_middle = (f1_l[0] + f1_l[1])/2.0
    f1_l = [f1_middle, f1_middle]
    f4_middle = (f4_l[0] + f4_l[1])/2.0
    f4_l = [f4_middle, f4_middle]
    update_lens(True, True)

def set_fl_absolute_middle():
    global f1_l, f4_l
    f1_middle = (7.0 + 27.0)/2.0
    f1_l = [f1_middle, f1_middle]
    f4_middle = (4.0 + 11.0)/2.0
    f4_l = [f4_middle, f4_middle]
    update_lens(True, True)

def reset_fl():
    global f1_l, f4_l
    f1_l = f1_orig.copy() 
    f4_l = f4_orig.copy()
    update_lens(True, True)
    
def modify_current_fl(lens_num, delta):
    global f1_l, f4_l
    f1 = False
    f4 = False
    if(lens_num == 1):
        f1_l[index] = f1_l[index] + delta
        if(f1_l[index] > 27.0):
            f1_l[index] = 27.0
            winsound.Beep(beep_frequency, beep_duration)
        if(f1_l[index] < 7.0):
            f1_l[index] = 7.0
            winsound.Beep(beep_frequency, beep_duration)
        f1 = True
    if(lens_num == 4):
        f4_l[index] = f4_l[index] + delta
        if(f4_l[index] > 11.0):
            f4_l[index] = 11.0
            winsound.Beep(beep_frequency, beep_duration)
        if(f4_l[index] < 4.0):
            f4_l[index] = 4.0
            winsound.Beep(beep_frequency, beep_duration)
        f4 = True
    update_lens(f1, f4)

class SlidersGroup(QGroupBox):
    valueChanged = pyqtSignal(int)
    def __init__(self, title, parent=None):
        super(SlidersGroup, self).__init__(title, parent)

        minimumValue = 0
        maximumValue = 1
        spinBoxValue = 1

        orientation = Qt.Horizontal
        self.slider = QSlider(orientation)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)

        self.spinBox_c = QSpinBox()
        self.spinBox_c.setRange(minimumValue, maximumValue)
        self.spinBox_c.setSingleStep(1)
        self.slider.valueChanged.connect(self.spinBox_c.setValue)
        self.spinBox_c.valueChanged.connect(self.slider.setValue)
        self.spinBox_c.setValue(spinBoxValue)

        self.spinBox_b = QSpinBox()
        self.spinBox_b.setRange(-100, 100)
        self.spinBox_b.setSingleStep(1)
        self.spinBox_b.valueChanged.connect(self.setMinimum)
        self.spinBox_b.setValue(minimumValue)

        self.spinBox_e = QSpinBox()
        self.spinBox_e.setRange(-100, 100)
        self.spinBox_e.setSingleStep(1)
        self.spinBox_e.valueChanged.connect(self.setMaximum)
        self.spinBox_e.setValue(maximumValue)

        slidersLayout = QGridLayout()
        slidersLayout.addWidget(self.spinBox_b, 0, 0, 1, 1)
        slidersLayout.addWidget(self.spinBox_e, 0, 3, 1, 1)
        slidersLayout.addWidget(self.slider, 1, 0, 1, 4)
        slidersLayout.addWidget(self.spinBox_c, 1, 4, 1, 1)

        self.setLayout(slidersLayout)

    def setMinimum(self, value):    
        self.slider.setMinimum(value)
        self.spinBox_c.setMinimum(value)

    def setMaximum(self, value):    
        self.slider.setMaximum(value)
        self.spinBox_c.setMaximum(value)

class OutputsGroup(QGroupBox):
    def __init__(self, title, parent=None):
        super(OutputsGroup, self).__init__(title, parent)

        defaultImageDistance = -1
        defaultMagnification = 0
        self.imagedistancetext = QLabel("Image Distance")
        self.imagedistance = QLabel(str(defaultImageDistance))
        self.magnificationtext = QLabel("Magnification")
        self.magnification = QLabel(str(defaultMagnification))
        outputsLayout = QGridLayout()
        outputsLayout.addWidget(self.imagedistancetext, 0, 0, 1, 1)
        outputsLayout.addWidget(self.imagedistance, 0, 1, 1, 1)
        outputsLayout.addWidget(self.magnificationtext, 1, 0, 1, 1)
        outputsLayout.addWidget(self.magnification, 1, 1, 1, 1)
        self.setLayout(outputsLayout)

    def updateImageDistance(self, value):
        self.imagedistance.setText(str(value))

    def updateMagnification(self, value):
        self.magnification.setText(str(value))

class OD():
    def __init__(self):
        self.d_f1_LCoS = 4.0 # Constrained by 30 mm cube
        self.d_LCoS_f2 = 4.0 # Constrained by 30 mm cube
        self.d_f2_f3 = 2 # Guess
        self.d_f3_f4 = 10.0 # Nikor 4f system
        self.d_f4_eye = 2 # Guess
        self.f4 = 5 # Assume that one of the lenses is fixed and the other is adjustable

    def calculateFocalLengths(self, d_vip_eye):
        self.d_vip_f1 = self.d_vip_eye - self.d_f4_eye - self.d_f3_f4 - self.d_f2_f3 - self.d_LCoS_f2 - self.d_f1_LCoS
        self.f1 = calculate_focal_length(self.d_vip_f1, self.d_f1_LCoS)
        self.magnification_f1 = -self.d_f1_LCoS/self.d_vip_f1
        self.d_vip_f2 = -(self.d_vip_f1 + self.d_f1_LCoS + self.d_LCoS_f2)
        self.f2 = calculate_focal_length(self.d_LCoS_f2, self.d_vip_f2)
        self.magnification_f2 = -self.d_vip_f2/self.d_LCoS_f2
        self.magnification_f1_f2 = self.magnification_f1 * self.magnification_f2
        self.magnification_f3_f4 = -1
        correction = -self.f4((1/self.magnification_f1_f2) + 1)
        self.f3 = self.f4 + correction
        self.magnification_f3_f4 = -self.f3/self.f4

'''
Remaining letters:
1234567890
  ERTYUI P
    GH  L
    BN
'''
class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.slider1 = SlidersGroup("Virtual Image Distance")
        self.outputs1 = OutputsGroup("Before Correction")
        self.outputs2 = OutputsGroup("After Correction")
        self.offset = 0.0

        self.slider1.slider.valueChanged.connect(self.calculateOutputs)
        
        layout = QVBoxLayout()
        layout.addWidget(self.slider1)
        layout.addWidget(self.outputs1)
        layout.addWidget(self.outputs2)
        self.setLayout(layout)
        self.setWindowTitle("Sliders")

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_O:
            increment = True
            while True:
                prev_time = time.time()
                while(time.time() - prev_time < 2):
                    pass
                if(increment == True):
                    increment_index()
                    print(index)
                    if(index == len(f1_l) - 1):
                        print('Increment is changed to False')
                        increment = False
                    else:
                        self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() + 1)
                else:
                    decrement_index()
                    print(index)
                    if(index == 0):
                        print('Increment is changed to True')
                        increment = True
                    else:
                        self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() - 1)
        if e.key() == Qt.Key_Q:
            if(self.slider1.spinBox_c.minimum() < self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() - 1)
                decrement_index()
        if e.key() == Qt.Key_W:
            if(self.slider1.spinBox_c.maximum() > self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() + 1)
                increment_index()
        if e.key() == Qt.Key_A:
            if(self.slider1.spinBox_c.minimum() < self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() - 1)
                decrement_index(True, False)
        if e.key() == Qt.Key_S:
            if(self.slider1.spinBox_c.maximum() > self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() + 1)
                increment_index(True, False)
        if e.key() == Qt.Key_Z:
            if(self.slider1.spinBox_c.minimum() < self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() - 1)
                decrement_index(False, True)
        if e.key() == Qt.Key_X:
            if(self.slider1.spinBox_c.maximum() > self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() + 1)
                increment_index(False, True)
        if e.key() == Qt.Key_J:
            self.offset = self.offset - 0.1
            calc_f4_l(self.offset)
        if e.key() == Qt.Key_K:
            self.offset = self.offset + 0.1
            calc_f4_l(self.offset)
        if e.key() == Qt.Key_Escape:
            self.close()
        if e.key() == Qt.Key_N:
            set_fl_absolute_middle()
        if e.key() == Qt.Key_M:
            set_fl_middle()
        if e.key() == Qt.Key_R:
            reset_fl()
        if e.key() == Qt.Key_D:
            modify_current_fl(1, -0.1)
        if e.key() == Qt.Key_F:
            modify_current_fl(1, 0.1)
        if e.key() == Qt.Key_C:
            modify_current_fl(4, -0.1)
        if e.key() == Qt.Key_V:
            modify_current_fl(4, 0.1)

    def calculateOutputs(self):
        self.outputs1.updateImageDistance(self.slider1.slider.value())
        self.outputs1.updateMagnification(self.slider1.slider.value())
        self.outputs2.updateImageDistance(self.slider1.slider.value())
        self.outputs2.updateMagnification(self.slider1.slider.value())


if __name__ == '__main__':
    connect_lenses()
    app = QApplication(sys.argv)
    window = Window()
    window.show()

sys.exit(app.exec_())

