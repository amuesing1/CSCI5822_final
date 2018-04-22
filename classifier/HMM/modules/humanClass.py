"""
**********************************************
File: humanClass.py
Author: Luke Burks
Date: November 2017

Making an interface for human aided
classification of point targets

**********************************************
"""
import numpy as np
import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *


sys.path.append('../../../scenario_simulator/modules')
from DynamicsProfiles import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Window(QtGui.QWidget):

    def __init__(self):
        super(Window,self).__init__()

        from classification import HumanTester
        self.human = HumanTester()

        np.random.seed(2)

        self.time = 0
        self.data = self.human.species.intensityModel
        self.data = self.data+np.random.normal(0,0.25,(len(self.data)))
        for k in range(0,len(self.data)):
            self.data[k] = max(self.data[k],1)
        self.data = self.data[0:100]
        self.plottedData = []
        self.colors = []
        self.updateTimer = QtCore.QTimer(self)
        self.interval = 1000
        self.updateTimer.setInterval(self.interval)

        self.updateTimer.timeout.connect(self.updateTable)

        self.matplotlibWidget = MatplotlibWidget(self)
        self.matplotlibWidget.move(200,200)

        self.pause = False

        self.initUI()


    def initUI(self):
        qbtn = QtGui.QPushButton('Quit',self)
        qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(0,250)



        self.tableWidget = QTableWidget(5,4,self)
        self.famNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']
        self.genusNames = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']

        for f in self.famNames:
            newItem = QTableWidgetItem(f)
            self.tableWidget.setItem(self.famNames.index(f),0,newItem)
        for g in self.genusNames:
            newItem = QTableWidgetItem(g)
            self.tableWidget.setItem(self.genusNames.index(g),2,newItem)

        #self.tableWidget.resize(self.tableWidget.sizeHint())
        self.tableWidget.resize(420,175)

        self.setGeometry(350,350,1050,850)
        self.setWindowTitle('Human-AI classification')


        updateButton = QtGui.QPushButton('Update Table',self)
        updateButton.clicked.connect(self.updateTable)
        updateButton.resize(updateButton.sizeHint())
        updateButton.move(100,250)


        self.pauseButton = QtGui.QPushButton("Play",self)
        self.pauseButton.clicked.connect(self.pausing)
        self.pauseButton.resize(self.pauseButton.sizeHint())
        self.pauseButton.setStyleSheet("background-color:green")
        self.pauseButton.move(100,300)

        speedUpButton = QtGui.QPushButton("Faster",self)
        speedUpButton.clicked.connect(self.speedUp)
        speedUpButton.resize(speedUpButton.sizeHint())
        speedUpButton.move(100,350)

        speedDownButton = QtGui.QPushButton("Slower",self)
        speedDownButton.clicked.connect(self.speedDown)
        speedDownButton.resize(speedDownButton.sizeHint())
        speedDownButton.move(0,350)

        self.layoutStratiform = QtGui.QHBoxLayout()
        self.widgetStratiform = QtGui.QWidget(self)
        self.widgetStratiform.setLayout(self.layoutStratiform)
        self.widgetStratiform.move(475,50)
        #self.lStratiform = QLabel('Stratiform            ')
        self.lStratiform = QLabel('Genus 0:')
        self.layoutStratiform.addWidget(self.lStratiform)
        self.layoutStratiform.addSpacing(10)
        self.stratiformGroup = QtGui.QButtonGroup(self.widgetStratiform)
        self.rStratiformYes = QtGui.QRadioButton('Yes')
        self.stratiformGroup.addButton(self.rStratiformYes)
        self.rStratiformNull = QtGui.QRadioButton('Null')
        self.rStratiformNull.setChecked(True)
        self.stratiformGroup.addButton(self.rStratiformNull)
        self.rStratiformNo = QtGui.QRadioButton('No')
        self.stratiformGroup.addButton(self.rStratiformNo)
        self.layoutStratiform.addWidget(self.rStratiformYes)
        self.layoutStratiform.addWidget(self.rStratiformNull)
        self.layoutStratiform.addWidget(self.rStratiformNo)

        self.layoutCirriform = QtGui.QHBoxLayout()
        self.widgetCirriform = QtGui.QWidget(self)
        self.widgetCirriform.setLayout(self.layoutCirriform)
        self.widgetCirriform.move(475,75)
        #self.lCirriform = QLabel('Cirriform               ')
        self.lCirriform = QLabel('Genus 1:')
        self.layoutCirriform.addWidget(self.lCirriform)
        self.layoutCirriform.addSpacing(10)
        self.CirriformGroup = QtGui.QButtonGroup(self.widgetCirriform)
        self.rCirriformYes = QtGui.QRadioButton('Yes')
        self.CirriformGroup.addButton(self.rCirriformYes)
        self.rCirriformNull = QtGui.QRadioButton('Null')
        self.rCirriformNull.setChecked(True)
        self.CirriformGroup.addButton(self.rCirriformNull)
        self.rCirriformNo = QtGui.QRadioButton('No')
        self.CirriformGroup.addButton(self.rCirriformNo)
        self.layoutCirriform.addWidget(self.rCirriformYes)
        self.layoutCirriform.addWidget(self.rCirriformNull)
        self.layoutCirriform.addWidget(self.rCirriformNo)

        self.layoutStratoCumuliform = QtGui.QHBoxLayout()
        self.widgetStratoCumuliform = QtGui.QWidget(self)
        self.widgetStratoCumuliform.setLayout(self.layoutStratoCumuliform)
        self.widgetStratoCumuliform.move(475,100)
        #self.lStratoCumuliform = QLabel('StratoCumuliform ')
        self.lStratoCumuliform = QLabel('Genus 2:')
        self.layoutStratoCumuliform.addWidget(self.lStratoCumuliform)
        self.layoutStratoCumuliform.addSpacing(10)
        self.StratoCumuliformGroup = QtGui.QButtonGroup(self.widgetStratoCumuliform)
        self.rStratoCumuliformYes = QtGui.QRadioButton('Yes')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformYes)
        self.rStratoCumuliformNull = QtGui.QRadioButton('Null')
        self.rStratoCumuliformNull.setChecked(True)
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNull)
        self.rStratoCumuliformNo = QtGui.QRadioButton('No')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformYes)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNull)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNo)


        self.layoutCumuliform = QtGui.QHBoxLayout()
        self.widgetCumuliform = QtGui.QWidget(self)
        self.widgetCumuliform.setLayout(self.layoutCumuliform)
        self.widgetCumuliform.move(475,125)
        #self.lCumuliform = QLabel('Cumuliform          ')
        self.lCumuliform = QLabel('Genus 3:')
        self.layoutCumuliform.addWidget(self.lCumuliform)
        self.layoutCumuliform.addSpacing(10)
        self.CumuliformGroup = QtGui.QButtonGroup(self.widgetCumuliform)
        self.rCumuliformYes = QtGui.QRadioButton('Yes')
        self.CumuliformGroup.addButton(self.rCumuliformYes)
        self.rCumuliformNull = QtGui.QRadioButton('Null')
        self.rCumuliformNull.setChecked(True)
        self.CumuliformGroup.addButton(self.rCumuliformNull)
        self.rCumuliformNo = QtGui.QRadioButton('No')
        self.CumuliformGroup.addButton(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(self.rCumuliformYes)
        self.layoutCumuliform.addWidget(self.rCumuliformNull)
        self.layoutCumuliform.addWidget(self.rCumuliformNo)


        self.layoutCumulonibiform = QtGui.QHBoxLayout()
        self.widgetCumulonibiform = QtGui.QWidget(self)
        self.widgetCumulonibiform.setLayout(self.layoutCumulonibiform)
        self.widgetCumulonibiform.move(475,150)
        # self.lCumulonibiform = QLabel('Cumulonibiform   ')
        self.lCumulonibiform = QLabel('Genus 4:')
        self.layoutCumulonibiform.addWidget(self.lCumulonibiform)
        self.layoutCumulonibiform.addSpacing(10)
        self.CumulonibiformGroup = QtGui.QButtonGroup(self.widgetCumulonibiform)
        self.rCumulonibiformYes = QtGui.QRadioButton('Yes')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformYes)
        self.rCumulonibiformNull = QtGui.QRadioButton('Null')
        self.rCumulonibiformNull.setChecked(True)
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNull)
        self.rCumulonibiformNo = QtGui.QRadioButton('No')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformYes)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNull)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNo)

        #self.updateTimer.start()

        self.show()

    def speedUp(self):
        self.interval -= 100
        self.interval = max(1,self.interval)
        self.updateTimer.setInterval(self.interval)

    def speedDown(self):
        self.interval += 100
        self.updateTimer.setInterval(self.interval)

    def pausing(self):

        if(self.pause):
            #play
            self.pauseButton.setText("Play")
            self.pauseButton.setStyleSheet("background-color:green")
            self.updateTimer.stop()
            self.pause = False
        else:
            self.pause = True
            self.pauseButton.setText("Pause")
            self.pauseButton.setStyleSheet("background-color:red")
            self.updateTimer.start()


    def updateTable(self):

        if(self.time >= len(self.data)):
            return

        #probs = {'Stratiform':np.random.random(),"Cirriform":np.random.random(),'Stratocumuliform':np.random.random(),'Cumuliform':np.random.random(),'Cumulonibiform':np.random.random()}
        obs = []
        if(self.rStratiformYes.isChecked()):
            obs.append(0)
            self.rStratiformNull.setChecked(True)
        elif(self.rStratiformNo.isChecked()):
            obs.append(2)
            self.rStratiformNull.setChecked(True)

        if(self.rCirriformYes.isChecked()):
            obs.append(3)
            self.rCirriformNull.setChecked(True)
        elif(self.rCirriformNo.isChecked()):
            obs.append(5)
            self.rCirriformNull.setChecked(True)

        if(self.rStratoCumuliformYes.isChecked()):
            obs.append(6)
            self.rStratoCumuliformNull.setChecked(True)
        elif(self.rStratoCumuliformNo.isChecked()):
            obs.append(8)
            self.rStratoCumuliformNull.setChecked(True)

        if(self.rCumuliformYes.isChecked()):
            obs.append(9)
            self.rCumuliformNull.setChecked(True)
        elif(self.rCumuliformNo.isChecked()):
            obs.append(11)
            self.rCumuliformNull.setChecked(True)

        if(self.rCumulonibiformYes.isChecked()):
            obs.append(12)
            self.rCumulonibiformNull.setChecked(True)
        elif(self.rCumulonibiformNo.isChecked()):
            obs.append(14)
            self.rCumulonibiformNull.setChecked(True)

        #Simulate random dropping of packets
        r = np.random.random()
        if(r < .1):
            self.plottedData.append(0)
            self.plottedData.append(0)
            self.plottedData.append(0)
            self.plottedData.append(0)
            self.colors.append('r')
            self.colors.append('r')
            self.colors.append('r')
            self.colors.append('r')
            self.time += 4
            return
        else:
            probs = self.human.singleUpdate(self.data[self.time])
            probsGenus = self.human.singleUpdateGenus(self.data[self.time],obs)
            self.plottedData.append(self.data[self.time])
            cers = 'y'
            #self.colors.append('y')
            if(probsGenus['Cumuliform0'] > .98):
                cers = 'g'
            self.colors.append(cers)

            self.matplotlibWidget.axis.scatter([i for i in range(0,len(self.plottedData))],self.plottedData,c=self.colors)
            self.matplotlibWidget.axis.set_ylim([0,11])
            self.matplotlibWidget.axis.set_title('Intensity Time Series')
            self.matplotlibWidget.axis.set_xlabel("Time Step")
            self.matplotlibWidget.axis.set_ylabel("Intensity")

            self.matplotlibWidget.canvas.draw()
            self.time += 1

            for p in probs.keys():
                newItem = QTableWidgetItem("{0:.2f}".format(probs[p]))
                self.tableWidget.setItem(self.famNames.index(p),1,newItem)
            for p in probsGenus.keys():
                newItem = QTableWidgetItem("{0:.2f}".format(probsGenus[p]))
                self.tableWidget.setItem(self.genusNames.index(p),3,newItem)
            #self.tableWidget.resize(self.tableWidget.sizeHint())
            self.tableWidget.resize(420,175)



class MatplotlibWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.axis = self.figure.add_subplot(111)

        self.layoutVertical = QtGui.QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)



def main():
    app = QtGui.QApplication(sys.argv)
    ex=Window()

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
