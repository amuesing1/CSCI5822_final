from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import sys
import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../../../scenario_simulator/modules')
from regions import Region
from DynamicsProfiles import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Window(QtGui.QWidget):
    def __init__(self,total_data,prior):
        super(Window,self).__init__()
        self.frame=0
        self.all_data=total_data
        self.table=prior
        self.prev_obs=0
        self.matplotlibWidget=MatplotlibWidget(self)
        self.matplotlibWidget2=MatplotlibWidget2(self)
        self.matplotlibWidget.move(25,25)
        self.matplotlibWidget2.move(550,300)
        means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
        new_means=np.reshape(np.array(means),(5,15,15))
        self.matplotlibWidget2.axis2.clear()

        _x=np.arange(15)
        _y=np.arange(15)
        _xx,_yy=np.meshgrid(_x,_y)
        x,y=_xx.ravel(),_yy.ravel()
        top=new_means[0,:,:]
        top=np.reshape(top,(1,225))[0]
        bottom=np.zeros_like(top)
        width=depth=1
        self.matplotlibWidget2.axis2.set_xlabel('Previous Obs')
        self.matplotlibWidget2.axis2.set_ylabel('Next Obs')
        self.matplotlibWidget2.axis2.set_title('Likelihoods of Genus 0')
        self.matplotlibWidget2.axis2.bar3d(x,y,bottom,width,depth,top,shade=True)
        self.matplotlibWidget2.canvas2.draw()

        self.updateTimer=QtCore.QTimer(self)
        self.interval = 1000
        self.updateTimer.setInterval(self.interval)

        self.updateTimer.start()
        self.updateTimer.timeout.connect(self.updateGraph)

        self.setGeometry(350,350,1250,900)



        self.updatebutton=QtGui.QPushButton("Update",self)
        self.updatebutton.move(775,200)
        self.updatebutton.clicked.connect(self.updateDir)
        self.layoutStratiform = QtGui.QHBoxLayout()
        self.widgetStratiform = QtGui.QWidget(self)
        self.widgetStratiform.setLayout(self.layoutStratiform)
        self.widgetStratiform.move(700,50)
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
        self.widgetCirriform.move(700,75)
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
        self.widgetStratoCumuliform.move(700,100)
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
        self.widgetCumuliform.move(700,125)
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
        self.widgetCumulonibiform.move(700,150)
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



        self.show()

    def updateGraph(self):
        maxsig=np.amax(self.all_data)
        self.matplotlibWidget.axis.imshow(self.all_data[self.frame],vmax=maxsig,cmap='Greys_r')
        self.matplotlibWidget.canvas.draw()
        self.frame+=1

    def updateDir(self):
        obs=[]
        if(self.rStratiformYes.isChecked()):
            obs.append(0)
            self.rStratiformNull.setChecked(True)
        elif(self.rStratiformNull.isChecked()):
            obs.append(1)
            self.rStratiformNull.setChecked(True) 
        elif(self.rStratiformNo.isChecked()):
            obs.append(2)
            self.rStratiformNull.setChecked(True)

        if(self.rCirriformYes.isChecked()):
            obs.append(3)
            self.rCirriformNull.setChecked(True)
        elif(self.rCirriformNull.isChecked()):
            obs.append(4)
            self.rCirriformNull.setChecked(True) 
        elif(self.rCirriformNo.isChecked()):
            obs.append(5)
            self.rCirriformNull.setChecked(True)

        if(self.rStratoCumuliformYes.isChecked()):
            obs.append(6)
            self.rStratoCumuliformNull.setChecked(True)
        elif(self.rStratoCumuliformNull.isChecked()):
            obs.append(7)
            self.rStratoCumuliformNull.setChecked(True) 
        elif(self.rStratoCumuliformNo.isChecked()):
            obs.append(8)
            self.rStratoCumuliformNull.setChecked(True)

        if(self.rCumuliformYes.isChecked()):
            obs.append(9)
            self.rCumuliformNull.setChecked(True)
        elif(self.rCumuliformNull.isChecked()):
            obs.append(10)
            self.rCumuliformNull.setChecked(True) 
        elif(self.rCumuliformNo.isChecked()):
            obs.append(11)
            self.rCumuliformNull.setChecked(True)

        if(self.rCumulonibiformYes.isChecked()):
            obs.append(12)
            self.rCumulonibiformNull.setChecked(True)
        elif(self.rCumulonibiformNull.isChecked()):
            obs.append(13)
            self.rCumulonibiformNull.setChecked(True) 
        elif(self.rCumulonibiformNo.isChecked()):
            obs.append(14)
            self.rCumulonibiformNull.setChecked(True)

        self.table[0,self.prev_obs,obs]+=.01
        means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
        new_means=np.reshape(np.array(means),(5,15,15))
        self.matplotlibWidget2.axis2.clear()

        _x=np.arange(15)
        _y=np.arange(15)
        _xx,_yy=np.meshgrid(_x,_y)
        x,y=_xx.ravel(),_yy.ravel()
        top=new_means[0,:,:]
        top=np.reshape(top,(1,225))[0]
        bottom=np.zeros_like(top)
        width=depth=1
        self.matplotlibWidget2.axis2.set_xlabel('Previous Obs')
        self.matplotlibWidget2.axis2.set_ylabel('Next Obs')
        self.matplotlibWidget2.axis2.set_title('Likelihoods of Genus 0')
        self.matplotlibWidget2.axis2.bar3d(x,y,bottom,width,depth,top,shade=True)
        self.matplotlibWidget2.canvas2.draw()
        #  ax1.bar3d(x,y,bottom,width,depth,top,shade=True)
        #  plt.show()

        self.prev_obs=random.choice(obs)

class MatplotlibWidget2(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget2,self).__init__(parent)
        self.figure2=Figure()
        self.canvas2=FigureCanvasQTAgg(self.figure2)
        self.axis2=self.figure2.add_subplot(111,projection='3d')
        self.layoutVertical2=QtGui.QVBoxLayout(self)
        self.layoutVertical2.addWidget(self.canvas2)

class MatplotlibWidget(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget,self).__init__(parent)
        self.figure=Figure()
        self.canvas=FigureCanvasQTAgg(self.figure)
        self.axis=self.figure.add_subplot(111)
        self.layoutVertical=QtGui.QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)

def make_some_data():
    img_path="../imgs/boulder.png"
    region_coordinates={'latmin':0,'latmax':0,'lonmin':0,'lonmax':0}
    Boulder=Region('Boulder',img_path,region_coordinates)
    Boulder.initPointTargets()
    Boulder.generateLayers()
    total_targets=np.zeros((100,100,100))
    for i, (gt_layer,sb_layer,pb_layer) in enumerate(zip(Boulder.ground_truth_layers,Boulder.shake_base_layers,Boulder.pixel_bleed_layers)):
        total_targets=total_targets+gt_layer+sb_layer+pb_layer
    total=total_targets+Boulder.noise_layer+Boulder.structured_noise_layer+Boulder.shotgun_noise_layer
    return total

def DirPrior():
    table=np.zeros((5,15,15))
    base_table=np.array([[0.0817438692,0.1634877384,0.0136239782,0.0544959128,0.1634877384,0.0272479564,0.0272479564,0.0953678474,0.0544959128,0.0408719346,0.068119891,0.0544959128,0.0326975477,0.0544959128,0.068119891],
            [0.0476190476,0.1428571429,0.0238095238,0.0714285714,0.1428571429,0.0119047619,0.0357142857,0.0833333333,0.0714285714,0.0476190476,0.119047619,0.0238095238,0.0238095238,0.0476190476,0.1071428571],
            [0.047318612,0.1261829653,0.0630914826,0.0378548896,0.0630914826,0.094637224,0.1261829653,0.047318612,0.0157728707,0.0630914826,0.141955836,0.0315457413,0.0315457413,0.047318612,0.0630914826],
            [0.0338983051,0.0847457627,0.0508474576,0.0508474576,0.2033898305,0.0338983051,0.0169491525,0.0338983051,0.1355932203,0.0508474576,0.1186440678,0.0169491525,0.0338983051,0.1016949153,0.0338983051],
            [0.0282258065,0.0483870968,0.0967741935,0.0201612903,0.0483870968,0.1612903226,0.0403225806,0.060483871,0.060483871,0.0201612903,0.0403225806,0.1814516129,0.1008064516,0.0806451613,0.0120967742]])
    for i in range(15):
        table[:,:,i]=base_table
    table*=0.1
    for i in range(15):
        table[:,i,i]*=10
    #  print table
    return table


if __name__ == '__main__':
    total=make_some_data()
    prior=DirPrior()
    app=QtGui.QApplication(sys.argv)
    ex=Window(total,prior)
    sys.exit(app.exec_())
