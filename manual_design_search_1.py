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

        self.currValueLabel = QLabel(str(spinBoxValue))
        self.slider.valueChanged.connect(self.updateLabel)

        slidersLayout = QGridLayout()
        slidersLayout.addWidget(self.spinBox_b, 0, 0, 1, 1)
        slidersLayout.addWidget(self.spinBox_e, 0, 3, 1, 1)
        slidersLayout.addWidget(self.slider, 1, 0, 1, 4)
        slidersLayout.addWidget(self.spinBox_c, 1, 4, 1, 1)
        slidersLayout.addWidget(self.currValueLabel, 1, 5, 1, 1)

        self.setLayout(slidersLayout)

    def setMinimum(self, value):    
        self.slider.setMinimum(value)
        self.spinBox_c.setMinimum(value)

    def setMaximum(self, value):    
        self.slider.setMaximum(value)
        self.spinBox_c.setMaximum(value)

    def updateLabel(self, value):
        self.currValueLabel.setText(str(value))

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.slider1 = SlidersGroup("Horizontal1")
        self.slider2 = SlidersGroup("Horizontal2")
        layout = QVBoxLayout()
        layout.addWidget(self.slider1)
        layout.addWidget(self.slider2)
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


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.show()

sys.exit(app.exec_())

