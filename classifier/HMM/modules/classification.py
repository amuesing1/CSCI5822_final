from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np
import h5py
from hmmlearn import hmm
import warnings
from copy import deepcopy
from gaussianMixtures import GM, Gaussian
import os
import time

from classifier_testing_utils import *

class HumanTester():

    def __init__(self, modelFileName='../data/histModels_fams.npy',genus=0):

        self.models = loadHistoricalModels(modelFileName)
        self.species = Cumuliform(genus=genus, weather=False)
        self.family_names = ['Stratiform', 'Cirriform', 'Stratocumuliform', 'Cumuliform', 'Cumulonibiform']
        self.genus_names = ['Cumuliform0', 'Cumuliform1', 'Cumuliform2', 'Cumuliform3', 'Cumuliform4']

        self.reset()

    def continueForward(self, newData, model, prevAlpha=[-1,-1]):
        x0 = model['prior']
        pxx = model['transition']
        pyx = model['obs']

        numStates = len(x0)
        if prevAlpha[0] == -1:
            prevAlpha=x0

        newAlpha = [-1]*numStates
        for xcur in range(numStates):
            newAlpha[xcur] = 0
            for xprev in range(numStates):
                newAlpha[xcur] += prevAlpha[xprev]*pxx[xcur][xprev]
            newAlpha[xcur] = newAlpha[xcur]*pyx[xcur].pointEval(newData)
        return newAlpha

    def humanTesting(self):
        modelFileName = '../data/histModels_fams.npy'
        models = loadHistoricalModels(modelFileName)

        genus = 'Cirriform0'
        species = Cirriform(genus = 0,weather=False)
        data = species.intensityModel

        famNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']

        obsMod = {}
        obsMod['Stratiform'] = [.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        obsMod['Cirriform'] = [.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        obsMod['Stratocumuliform'] = [.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6]
        obsMod['Cumuliform'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6]
        obsMod['Cumulonibiform'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2]

        alphas = {}
        for f in famNames:
            alphas[f] = [-1,-1]

        probs = {}
        for f in famNames:
            probs[f] = 1

        #for each bit of data
        for d in data:
            #update classification probs
            for f in famNames:
                alphas[f] = self.continueForward(d, models[f], alphas[f])
                probs[f] = probs[f]*sum(alphas[f])

            #normalize probs
            suma = sum(probs.values())
            for f in famNames:
                probs[f] = probs[f]/suma

            #show to human
            #get human observation
            #print(probs)
            #ob = int(raw_input("Which observation would you like to make?"))

            ob = -1
            #print('up')
            if ob != -1:
                #apply bayes rule
                for f in famNames:
                    probs[f] = probs[f]*obsMod[f][ob]

            #normalize probs
            suma = sum(probs.values())
            for f in famNames:
                probs[f] = probs[f]/suma

            self.probabilities = probs
            # #update alphas
            # for f in famNames:
            # 	for i in range(len(alphas[f])):
            # 		alphas[f][i] = alphas[f][i]*probs[f]

    def singleUpdate(self, data_i, obs=-1):

        #update classification probs
        for f in self.family_names:
            self.alphas[f] = self.continueForward(data_i, self.models[f], self.alphas[f])
            self.probabilities[f] = self.probabilities[f] * sum(self.alphas[f])

        #normalize probs
        suma = sum(self.probabilities.values())
        for f in self.family_names:
            self.probabilities[f] = self.probabilities[f]/suma

        #show to human
        #get human observation
        #print(probs)
        #ob = int(raw_input("Which observation would you like to make?"))

        if obs != -1:
            #apply bayes rule
            for ob in obs:
                for f in self.family_names:
                    self.probabilities[f] = self.probabilities[f]*self.obsMod[f][ob]

        #normalize probs
        suma = sum(self.probabilities.values())
        for f in self.family_names:
            self.probabilities[f] = self.probabilities[f]/suma

        return self.probabilities

    def singleUpdateGenus(self, data_i, obs=-1):

        #update classification probs
        for f in self.genus_names:
            self.genus_alphas[f] = self.continueForward(data_i, self.models[f], self.genus_alphas[f])
            self.genus_probabilities[f] = self.genus_probabilities[f] * sum(self.genus_alphas[f])

        #normalize probsGenus
        suma = sum(self.genus_probabilities.values())
        for f in self.genus_names:
            self.genus_probabilities[f] = self.genus_probabilities[f] / suma

        #show to human
        #get human observation
        #print(probsGenus)
        #ob = int(raw_input("Which observation would you like to make?"))
        #print(obs)

        # ************************************************
        # NOTE: RNN PLUGS IN HERE
        # ************************************************

        if obs != -1:
            #apply bayes rule
            for ob in obs:
                for f in self.genus_names:
                    self.genus_probabilities[f] = self.genus_probabilities[f] * self.obsModGenus[f][ob]

        #normalize probsGenus
        suma = sum(self.genus_probabilities.values())
        for f in self.genus_names:
            self.genus_probabilities[f] = self.genus_probabilities[f] / suma

        return self.genus_probabilities

    def updateML(self, prior, data_i):

        # Initialize genus probabilities with prior
        self.genus_probabilities = prior

        # Update classification probabilities
        for genus in self.genus_names:
            self.genus_alphas[genus] = self.continueForward(data_i, self.models[genus], self.genus_alphas[genus])
            self.genus_probabilities[genus] *= sum(self.genus_alphas[genus])

        # Normalize genus probabilities
        # suma = np.sum(np.exp(self.genus_probabilities.values()))
        # for genus, logit in zip(self.genus_names, self.genus_probabilities.values()):
        #     self.genus_probabilities[genus] = np.exp(logit) / float(suma + 1e-9)

        # Normalize genus probabilities
        suma = sum(self.genus_probabilities.values())

        for genus in self.genus_names:
            self.genus_probabilities[genus] /= suma

        return self.genus_probabilities

    def reset(self, init_probabilities=None):
        self.alphas = { f : [-1, -1] for f in self.family_names }
        self.probabilities = { f : 1 for f in self.family_names }
        self.genus_alphas = { f : [-1, -1] for f in self.genus_names }
        self.genus_probabilities = init_probabilities if init_probabilities else { f: 1 for f in self.genus_names }

if __name__ == "__main__":

    TESTING = False

    sys.path.append('../../../scenario_simulator/modules')
    from DynamicsProfiles import *

    if TESTING:
        makingAndTestingStuff()
    else:
        a = HumanTester()
        a.humanTesting()

else:
    sys.path.append('../../scenario_simulator/modules')
    from DynamicsProfiles import *
