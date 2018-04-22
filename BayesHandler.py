"""
**********************************************
File: BayesHandler.py
Author: Luke Burks, Michael Iuzzolino
Date: December 2017

General Interface for maintaining a belief over
targets

**********************************************
"""
from __future__ import division
from copy import deepcopy
import numpy as np
import sys
import json
sys.path.append("../scenario_simulator/modules/")
sys.path.append("./HMM/modules")
sys.path.append("./RNN/modules")
sys.path.append("./human/data")
from DynamicsProfiles import *
from classification import HumanTester as HMMClassifier
from RNN_Classifiers import RNN
import time

class BayesHandler(object):

    def __init__(self, MLC, skill = "novice"):
        self.MLClassifier = MLC

        #  try:
        #      self.saved_observations=np.load('backend_development/classifier/human/data/human_observations.npy')
        #  except:
        self.saved_observations=np.array([])
        self.saved_observations_filename='backend_development/classifier/human/data/human_observations_'+str(int(time.time()))+'.npy'
        #  self.hdf=pd.HDFStore('human_observations.h5')
        #  try:
        #      self.saved_observations=read_hdf('human_observations.h5','data',mode='r+')
        #  except:
        #      self.saved_observations=pd.DataFrame()

        self.genus_names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']

        # Jeremy's Approximation of a sperical human in a vacuum
        # [P(Yes_0|0),P(Null_0|0),P(No_0|0),P(Yes_1|0),P(Null_1|0),...]
        # [P(Yes_0|1),P(Null_0|1),P(No_0|1),P(Yes_1|1),P(Null_1|1),...]
        if skill=="novice":
            self.human_observation_model = {}
            self.human_observation_model['Cumuliform0'] = [0.0817438692,0.1634877384,0.0136239782,0.0544959128,0.1634877384,0.0272479564,0.0272479564,0.0953678474,0.0544959128,0.0408719346,0.068119891,0.0544959128,0.0326975477,0.0544959128,0.068119891]
            self.human_observation_model['Cumuliform1'] = [0.0476190476,0.1428571429,0.0238095238,0.0714285714,0.1428571429,0.0119047619,0.0357142857,0.0833333333,0.0714285714,0.0476190476,0.119047619,0.0238095238,0.0238095238,0.0476190476,0.1071428571]
            self.human_observation_model['Cumuliform2'] = [0.047318612,0.1261829653,0.0630914826,0.0378548896,0.0630914826,0.094637224,0.1261829653,0.047318612,0.0157728707,0.0630914826,0.141955836,0.0315457413,0.0315457413,0.047318612,0.0630914826]
            self.human_observation_model['Cumuliform3'] = [0.0338983051,0.0847457627,0.0508474576,0.0508474576,0.2033898305,0.0338983051,0.0169491525,0.0338983051,0.1355932203,0.0508474576,0.1186440678,0.0169491525,0.0338983051,0.1016949153,0.0338983051]
            self.human_observation_model['Cumuliform4'] = [0.0282258065,0.0483870968,0.0967741935,0.0201612903,0.0483870968,0.1612903226,0.0403225806,0.060483871,0.060483871,0.0201612903,0.0403225806,0.1814516129,0.1008064516,0.0806451613,0.0120967742]
        elif skill=="expert":
            #Jeremy
            # self.human_observation_model = {}
            # self.human_observation_model['Cumuliform0'] = [0.125,0.1388888889,0.0138888889,0.0416666667,0.1388888889,0.0555555556,0.0277777778,0.0833333333,0.0694444444,0.0277777778,0.0694444444,0.0277777778,0.0277777778,0.0555555556,0.0972222222]
            # self.human_observation_model['Cumuliform1'] = [0.024691358,0.1358024691,0.037037037,0.0987654321,0.1234567901,0.012345679,0.0296296296,0.0740740741,0.0864197531,0.0320987654,0.0987654321,0.049382716,0.024691358,0.049382716,0.1234567901]
            # self.human_observation_model['Cumuliform2'] = [0.03125,0.109375,0.09375,0.03125,0.0625,0.109375,0.125,0.046875,0.015625,0.046875,0.125,0.046875,0.03125,0.046875,0.078125]
            # self.human_observation_model['Cumuliform3'] = [0.0317460317,0.0793650794,0.0634920635,0.0317460317,0.1746031746,0.0476190476,0.0158730159,0.0317460317,0.126984127,0.0793650794,0.0952380952,0.0158730159,0.0317460317,0.0952380952,0.0793650794]
            # self.human_observation_model['Cumuliform4'] = [0.0266159696,0.0456273764,0.0912547529,0.0190114068,0.0456273764,0.1520912548,0.0380228137,0.0570342205,0.0760456274,0.0190114068,0.0380228137,0.1711026616,0.1330798479,0.0760456274,0.0114068441]

            #Luke alterations
            self.human_observation_model = {}
            self.human_observation_model['Cumuliform0'] = [0.325,0.1388888889,0.0138888889,0.0416666667,0.1388888889,0.0555555556,0.0277777778,0.0833333333,0.0694444444,0.0277777778,0.0694444444,0.0277777778,0.0277777778,0.0555555556,0.0972222222]
            self.human_observation_model['Cumuliform1'] = [0.024691358,0.1358024691,0.037037037,0.3987654321,0.1234567901,0.012345679,0.0296296296,0.0740740741,0.0864197531,0.0320987654,0.0987654321,0.049382716,0.024691358,0.049382716,0.1234567901]
            self.human_observation_model['Cumuliform2'] = [0.03125,0.109375,0.09375,0.03125,0.0625,0.109375,0.325,0.046875,0.015625,0.046875,0.125,0.046875,0.03125,0.046875,0.078125]
            self.human_observation_model['Cumuliform3'] = [0.0317460317,0.0793650794,0.0634920635,0.0317460317,0.1746031746,0.0476190476,0.0158730159,0.0317460317,0.126984127,0.3793650794,0.0952380952,0.0158730159,0.0317460317,0.0952380952,0.0793650794]
            self.human_observation_model['Cumuliform4'] = [0.0266159696,0.0456273764,0.0912547529,0.0190114068,0.0456273764,0.1520912548,0.0380228137,0.0570342205,0.0760456274,0.0190114068,0.0380228137,0.1711026616,0.3330798479,0.0760456274,0.0114068441]

            for key in self.human_observation_model.keys():
                suma = sum(self.human_observation_model[key]); 
                for i in range(0,len(self.human_observation_model[key])):
                    self.human_observation_model[key][i] /= suma; 



        #print("Initializing Bayes Handler...")

        # Set uniform initial proability
        self.probabilities = { genus : 0.2 for genus in self.genus_names }

    def update(self, data=None, human_observation=None ,track_id=100, region=100):
        if track_id!=100:
            track_id=int(track_id[-1])
        if region!=100:
            region=['boulder','israel','north_korea'].index(region)

        if data is not None:
            self.probabilities = self.MLClassifier.updateML(self.probabilities, data)

        if human_observation: 
            if len(self.saved_observations)!=0:
                self.saved_observations=np.vstack((self.saved_observations,
                    np.append(np.array([region,track_id]),np.array(human_observation))))
            else:
                self.saved_observations=np.append(np.array([region,track_id]),np.array(human_observation))
            print self.saved_observations
            np.save(self.saved_observations_filename,self.saved_observations)
            self.probabilities = self.humanUpdate(human_observation)

        # Find argmax genus
        self.best_genus = self.probabilities.keys()[np.argmax([prob for prob in self.probabilities.values()])]

        return self.probabilities, self.best_genus

    def humanUpdate(self, observations):
        # Apply bayes rule for each observation
        for observation in observations:
            for genus in self.genus_names:
                self.probabilities[genus] *= self.human_observation_model[genus][observation]

        # Normalize probabilities
        suma = sum(self.probabilities.values())

        for genus in self.genus_names:
            self.probabilities[genus] /= suma

        return self.probabilities

    def reset(self):
        self.probabilities = { genus : 0.2 for genus in self.genus_names }
        self.MLClassifier.reset(self.probabilities)

if __name__ == '__main__':

    # HMM
    # -------------------------------------------------
    Bayes_handler_hmm = BayesHandler(HMMClassifier('./HMM/data/histModels_fams.npy'))
    # -------------------------------------------------

    # RNN
    # -------------------------------------------------
    with open("RNN/models/architecture_build/rnn_params.json", "r") as infile:
        rnn_params = json.load(infile)

    rnn_params["model_filepath"] = "RNN/models/final_model.ckpt"

    RNN_clf = RNN(**rnn_params)
    Bayes_handler_rnn = BayesHandler(RNN_clf)
    # -------------------------------------------------

    data_sequence = []
    try:
        while True:
            new_data_point = int(raw_input("New data: "))
            data_sequence.append(new_data_point)
            print("Data: {}".format(data_sequence))

            print("HMM")
            Bayes_handler_hmm.update(new_data_point, [1])

            print("RNN")
            Bayes_handler_rnn.update(new_data_point, [1])

    except KeyboardInterrupt:
        print("Exiting.")
