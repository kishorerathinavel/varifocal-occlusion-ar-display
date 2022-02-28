#!/usr/bin/env python

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox,
                             QVBoxLayout, QSlider, QSpinBox, QWidget, QLabel)

class SlidersGroup(QGroupBox):
    valueChanged = pyqtSignal(int)
    def __init__(self, title, parent=None):
        super(SlidersGroup, self).__init__(title, parent)

        minimumValue = 0
        maximumValue = 20
        spinBoxValue = 5

        orientation = Qt.Horizontal
        self.slider = QSlider(orientation)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
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
    
class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.slider1 = SlidersGroup("Virtual Image Distance")
        self.outputs1 = OutputsGroup("Before Correction")
        self.outputs2 = OutputsGroup("After Correction")

        self.slider1.slider.valueChanged.connect(self.calculateOutputs)
        
        layout = QVBoxLayout()
        layout.addWidget(self.slider1)
        layout.addWidget(self.outputs1)
        layout.addWidget(self.outputs2)
        self.setLayout(layout)
        self.setWindowTitle("Sliders")

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Q:
            if(self.slider1.spinBox_c.minimum() < self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() - 1)
        if e.key() == Qt.Key_W:
            if(self.slider1.spinBox_c.maximum() > self.slider1.spinBox_c.value()):
                self.slider1.spinBox_c.setValue(self.slider1.spinBox_c.value() + 1)
        if e.key() == Qt.Key_Escape:
            self.close()

    def calculateOutputs(self):
        self.outputs1.updateImageDistance(self.slider1.slider.value())
        self.outputs1.updateMagnification(self.slider1.slider.value())
        self.outputs2.updateImageDistance(self.slider1.slider.value())
        self.outputs2.updateMagnification(self.slider1.slider.value())


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.show()

sys.exit(app.exec_())

