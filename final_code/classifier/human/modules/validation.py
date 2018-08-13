from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import gamma, psi, polygamma
import random
import copy
import sys
import itertools
import warnings
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
np.set_printoptions(precision=2)

#  sys.path.append('../../../scenario_simulator/modules')
sys.path.append('../../HMM/modules')

from gaussianMixtures import GM,Gaussian
#  from regions import Region
#  from DynamicsProfiles import *

warnings.filterwarnings("ignore",category=RuntimeWarning)

class Validation():
    def __init__(self):

        self.table=self.DirPrior()
        self.theta1=np.zeros((5,10))
        self.theta2_correct=np.zeros((50,10))
        self.table_ind=np.mean(self.table,axis=1)
        for X in range(5):
            self.theta1[X,:]=scipy.stats.dirichlet.mean(alpha=np.mean(self.table,axis=1)[X,:])
        self.table_real[self.table_real<0]=0.1
        for X in range(5):
            for prev_obs in range(10):
                self.theta2_correct[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.table_real[X,prev_obs,:])

    def DirPrior(self):
        table=np.zeros((5,10,10))
        base_table=np.ones((5,10))
        #  base_table*=0.1
        for i in range(5):
            base_table[i,2*i]*=2
            for j in range(5):
                if i==j:
                    base_table[i,2*j+1]*=0.5
                else:
                    base_table[i,2*j+1]*=2
                    base_table[i,2*j]*=0.5
        for i in range(10):
            table[:,:,i]=base_table
        for i in range(10):
            table[:,i,i]*=3
        table=np.swapaxes(table,1,2)

        table*=5
        #  self.table_real=np.random.normal(table,1)
        table_real=np.zeros((5,10,10))
        #  base_table_real=np.random.rand(5,10)
        base_table_real=np.ones((5,10))
        base_table_real*=5
        for i in range(5):
            base_table_real[i,2*i]*=1.67
            for j in range(5):
                if i==j:
                    base_table_real[i,2*j+1]*=0.7
                else:
                    base_table_real[i,2*j+1]*=1.67
                    base_table_real[i,2*j]*=0.7
        for i in range(10):
            table_real[:,:,i]=base_table_real
        for i in range(10):
            table_real[:,i,i]*=50
        table_real=np.swapaxes(table_real,1,2)
        table_real+=np.random.uniform(-1,1,(5,10,10))
        self.table_real=table_real
        #  print table_real
        return table

    def updateProbs_ind(self,real_target):
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        obs_names=['Yes-0','No-0','Yes-1','No-1','Yes-2','No-2','Yes-3','No-3','Yes-4','No-4']
        
        # initialize Dir sample
        theta1_static=np.zeros((5,10))
        postX=copy.copy(self.probs_ind)
        all_post=np.zeros((1,5))
        all_theta1=np.zeros((1,5,10))
        all_theta1[:]=np.nan
        for X in range(5):
            theta1_static[X,:]=scipy.stats.dirichlet.mean(alpha=self.table_ind[X,:])
        #  tic=time.time()
        theta1=copy.copy(theta1_static)
        #  print "Observation: %s" % obs_names[self.obs[-1]]
        for n in range(2000):
            for i in names:
                likelihood=theta1[names.index(i),self.obs[-1]]
                #  print likelihood
                postX[i]=self.probs_ind[i]*likelihood
            suma=sum(postX.values())
            # normalize
            for i in names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            if n%5==0:
                all_post=np.append(all_post,[postX.values()],axis=0)
            # sample from X
            X=np.random.choice(range(5),p=postX.values())
            alphas=copy.copy(self.table_ind)
            theta1=copy.copy(theta1_static)
            theta1_to_store=np.zeros((5,10))
            theta1_to_store[:]=np.nan
            alphas[X,self.obs[-1]]+=1
            theta1[X,:]=np.random.dirichlet(alphas[X,:])
            if n%5==0:
                theta1_to_store[X,:]=theta1[X,:]
                all_theta1=np.append(all_theta1,[theta1_to_store],axis=0)
        if len(self.obs)>0:
            theta1_change=(theta1_static-np.nan_to_num(np.nanmean(all_theta1,axis=0)))/theta1_static
            theta1_change[theta1_change==1]=0
            if np.amin(theta1_change)<0:
                rows=list(set(np.where(theta1_change<0)[0]))
                #  print rows
                for i in rows:
                    smallest=min(theta1_change[i,:])
                    theta1_change[i,:]+=np.abs(smallest)
            theta1_change=theta1_change.reshape((5,10))
            self.table_ind+=((theta1_change)/np.sum(theta1_change))
            #  print self.table
        post_probs=np.mean(all_post[1:],axis=0)
        for i in names:
            self.probs_ind[i]=post_probs[names.index(i)]

    def updateProbs(self,real_target):
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        obs_names=['Yes-0','No-0','Yes-1','No-1','Yes-2','No-2','Yes-3','No-3','Yes-4','No-4']
        
        # initialize Dir sample
        num_samples=5000
        sample_check=[]
        theta2_static=np.empty((50,10))
        postX=copy.deepcopy(self.probs)
        all_post=np.zeros((int(num_samples/5),1,5))
        all_theta2=np.zeros((int(num_samples/5),50,10))
        #  all_theta2[:]=np.nan
        for X in range(5):
            for prev_obs in range(10):
                #  print self.table[X,prev_obs,:]
                theta2_static[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.table[X,prev_obs,:])
                #  print theta2_static[X*10+prev_obs,:]
        if len(self.obs)>0:
            prev_obs=self.obs[-1]
            self.obs.append(np.random.choice(range(10),p=self.theta2_correct[real_target*10+prev_obs,:]))
        else:
            self.obs.append(np.random.choice(range(10),p=self.theta1[real_target,:]))
        # confusion matrix for human
        if self.obs[-1]%2==0:
            self.pred_obs.append(0)
            if (self.obs[-1]/2)==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)
        else:
            self.pred_obs.append(1)
            if (int(self.obs[-1]/2))==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)

        theta2=copy.deepcopy(theta2_static)
        #  print "Observation: %s" % obs_names[self.obs[-1]]
        for n in range(num_samples):
            for i in names:
                likelihood=self.theta1[names.index(i),self.obs[0]]
                # sample from theta2
                if len(self.obs)>1:
                    for value in self.obs[1:]:
                        likelihood*=theta2[names.index(i)*10+self.obs[self.obs.index(value)-1],value]
                        #  print "Alpha value: %f" % self.table[names.index(i),self.obs[self.obs.index(value)-1],value]
                        #  print "Theta value: %f" % theta2[names.index(i)*10+self.obs[self.obs.index(value)-1],value]
                #  print likelihood
                postX[i]=self.probs[i]*likelihood
            suma=sum(postX.values())
            # normalize
            for i in names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            if n%5==0:
                all_post[int(n/5),:,:]=postX.values()
            # sample from X
            X=np.random.choice(range(5),p=postX.values())
            alphas=copy.deepcopy(self.table)
            theta2=copy.deepcopy(theta2_static)
            if len(self.obs)>1:
                alphas[X,self.obs[-2],self.obs[-1]]+=1
                #  print X,self.obs[y-1],value
                #  print "alphas",alphas[X,self.obs[y-1],:]
                #  print "start",theta2[X*10+self.obs[y-1],:]
                theta2[X*10+self.obs[-2],:]=np.random.dirichlet(alphas[X,self.obs[-2],:])
                #  if (X==0) and (self.obs[y-1]==1):
                #      print value,alphas[X,self.obs[y-1],:]
                #      print value,theta2[X*10+self.obs[y-1],:]
                #  print "end",theta2[X*10+self.obs[y-1],:]
                if n%5==0:
                    all_theta2[int(n/5),X*10+self.obs[-2],:]=theta2[X*10+self.obs[-2],:]

        if len(self.obs)>1:
            #  print all_theta2.shape
            #  print "alphas start",self.table[0,1,:]
            for n in range(all_theta2.shape[1]):
                pk_top_list=[]
                sum_alpha=sum(self.table[int(n/10),n%10,:])
                for k in range(all_theta2.shape[2]):
                    samples=all_theta2[np.nonzero(all_theta2[:,n,k]),n,k]
                    #  if (n==0) and (k==0) and (len(samples[0])>0):
                    #      print "obs" ,self.obs, "sample mean",np.mean(all_theta2[np.nonzero(all_theta2[:,0,:]),0,:],axis=1)[0]
                    if len(samples[0])==0:
                        pass
                    else:
                        #  print np.mean(samples[0])
                        #  print "real:",scipy.stats.dirichlet.mean(alpha=self.table_real[int(n/10),n%10,:])[k]
                        pk_top_list.append(np.mean(samples[0]))
                        current_alpha=self.table[int(n/10),n%10,k]
                        #  print "result:",np.mean(samples)
                        #  print "real:",self.theta2_correct[n,k]
                        for x in range(5):
                            #  N_numerator=1
                            #  for k in range(all_theta2.shape[2]):
                            #      N_numerator=N_numerator*gamma(self.table[int(n/10),n%10,k])
                            sum_alpha_old=sum_alpha-current_alpha+self.table[int(n/10),n%10,k]
                            #  N=N_numerator/gamma(sum_alpha)
                            logpk=np.sum(np.log(samples[0]))/len(samples[0])
                            #  print logpk
                            y=psi(sum_alpha_old)+logpk
                            #  print "total alpha:",sum_alpha
                            #  print "logpk:",logpk
                            #  print "y:",y
                            if y>=-2.22:
                                alphak=np.exp(y)+0.5
                            else:
                                alphak=-1/(y+psi(1))
                            #  print "start:",alphak
                            for w in range(5):
                                alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                            self.table[int(n/10),n%10,k]=alphak
                            #  if (n==1) and (k==9) and (len(samples[0])>0) and (x==4):
                            #      print "theta start",theta2_static[1,:]
                            #      print "end theta",scipy.stats.dirichlet.mean(alpha=self.table[0,1,:])
                                #  print "real theta",scipy.stats.dirichlet.mean(alpha=self.table_real[0,1,:])
                            #      print "alphas end",self.table[0,1,:]
                                #  print "alphas real",self.table_real[0,1,:]
                                #  sys.exit()
                        #  print "result:",alphak
                            #  print "result y:",psi(alphak)
                            #  print "what y should be:",psi(self.table_real[int(n/10),n%10,m])
                            #  print "what logpk should be:",psi(self.table_real[int(n/10),n%10,m])-psi(sum_alpha)
                            #  print "what N should be:",(np.sum(np.log(pklist)))/(psi(self.table_real[int(n/10),n%10,m])-psi(sum_alpha))
                        #  print "real:",self.table_real[int(n/10),n%10,k]
                            #  sys.exit()
                #  if (n==0) and (len(samples[0])>0):
                #      print "result:",self.table[int(n/10),n%10,:]
                #      print "real:",self.table_real[int(n/10),n%10,:]
                #      print "sample:",["%.2f" % e for e in pk_top_list]
                #      print "sample:",["%.2f" % e for e in pk_top_list]
                    #  print "result:",scipy.stats.dirichlet.mean(alpha=self.table[int(n/10),n%10,:])
                    #  print "real:",scipy.stats.dirichlet.mean(alpha=self.table_real[int(n/10),n%10,:])
                #      sys.exit()

        post_probs=np.mean(all_post,axis=0)
        for i in names:
            self.probs[i]=post_probs[0][names.index(i)]

def KLD(mean_i,mean_j,var_i,var_j):

    dist=.5*((var_i**2/var_j**2)+var_j**2*(mean_j-mean_i)**2-1+np.log(var_j**2/var_i**2))

    return np.absolute(dist)


if __name__ == '__main__':
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])
    num_tar=int(commands[0])

    sim=Validation()
    total_difference=np.empty([1,50,10])
    theta_real_mean=np.empty((50,10))
    theta_calc_mean=np.empty((50,10))
    theta_real_var=np.empty((50,10))
    theta_calc_var=np.empty((50,10))
    true_tar=[]
    pred_tar=[]
    pred_percent=[]
    sim.pred_obs=[]
    sim.real_obs=[]
    #  theta_real_ind=np.empty((5,10))
    #  theta_calc_ind=np.empty((5,10))
    names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
    correct=[0]*num_tar
    correct_percent=[]
    correct_ml=[0]*num_tar
    correct_percent_ml=[]
    #  correct_ind=[0]*num_tar
    #  correct_percent_ind=[]
    for n in tqdm(range(num_tar),ncols=100):
    #  for n in range(num_tar):
    #      print "%d /50" % n
        # initialize target type
        genus=np.random.randint(5)

        if commands[1]=='uniform':
            sim.probs={}
            for i in names:
                sim.probs[i]=.2
            correct_ml[n]=np.random.choice([0,0,0,0,1],p=sim.probs.values())
            correct_percent_ml.append(sum(correct_ml)/(n+1))
        elif commands[1]=='assist':
            sim.probs={}
            for i in names:
                if names.index(i)==genus:
                    sim.probs[i]=np.random.normal(.75,.25)
                else:
                    sim.probs[i]=np.random.normal(.25,.25)
                if sim.probs[i]<0:
                    sim.probs[i]=0.01
            for i in names:
                sim.probs[i]/=sum(sim.probs.values())
            chosen_ml=max(sim.probs.values())
            if genus==sim.probs.values().index(chosen_ml):
                correct_ml[n]=1
            correct_percent_ml.append(sum(correct_ml)/(n+1))

        sim.obs=[]
        # 5 observations per target
        #  for j in range(int(np.random.uniform(1,6))):
        for j in range(5):
            sim.updateProbs(genus)
            #  sim.updateProbs_ind(genus)
        chosen=max(sim.probs.values())
        pred_percent.append(chosen)
        true_tar.append(genus)
        pred_tar.append(sim.probs.values().index(chosen))
        #  check_probs_ind=sim.probs_ind.values()
        if genus==sim.probs.values().index(chosen):
            correct[n]=1
        #  if genus==check_probs_ind.index(max(check_probs_ind)):
        #      correct_ind[n]=1
        correct_percent.append(sum(correct)/(n+1))
        #  correct_percent_ind.append(sum(correct_ind)/(n+1))
        for X in range(5):
            #  theta_real_ind[X,:]=scipy.stats.dirichlet.mean(alpha=np.mean(sim.table_real,axis=1)[X,:])
            #  theta_calc_ind[X,:]=scipy.stats.dirichlet.mean(alpha=sim.table_ind[X,:])
            for prev_obs in range(10):
                theta_real_mean[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=sim.table_real[X,prev_obs,:])
                theta_calc_mean[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=sim.table[X,prev_obs,:])
                theta_real_var[X*10+prev_obs,:]=scipy.stats.dirichlet.var(alpha=sim.table_real[X,prev_obs,:])
                theta_calc_var[X*10+prev_obs,:]=scipy.stats.dirichlet.var(alpha=sim.table[X,prev_obs,:])
        if (n%int((num_tar/10))==0) or n==(num_tar-1):
            difference=np.empty([50,10])
            #  print theta_real_mean[10,2],theta_calc_mean[10,2]
            #  print theta_real_var[10,2],theta_calc_var[10,2]
            for i in range(difference.shape[0]):
                for j in range(difference.shape[1]):
                    difference[i,j]=KLD(theta_real_mean[i,j],theta_calc_mean[i,j],theta_real_var[i,j],theta_calc_var[i,j])
            total_difference=np.append(total_difference,np.expand_dims(difference,axis=0),axis=0)

    plt.figure(1)
    plt.subplot(221)
    cm=confusion_matrix(true_tar,pred_tar)
    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm,cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Given Label')
    plt.title('Target Classification Confusion Matrix')
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

    plt.subplot(222)
    cm=confusion_matrix(sim.real_obs,sim.pred_obs)
    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm,cmap='Blues')
    plt.ylabel('True Value')
    plt.xlabel('Given Obs')
    plt.title('Human Observations Confusion Matrix')
    plt.xticks([0,1],['pos','neg'])
    plt.yticks([0,1],['pos','neg'])
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

    plt.subplot(223)
    plt.plot([n+5 for n in range(num_tar-5)],correct_percent[5:], label="w/Human Total Correct")
    plt.plot([n+5 for n in range(num_tar-5)],correct_percent_ml[5:], label="wo/Human Total Correct")
    plt.legend()
    plt.xlabel('Number of Targets')
    plt.ylabel('Percent Correct')
    #  plt.legend(loc='best')
    plt.title('Correct Classification')

    plt.subplot(224)
    precision, recall, _ =precision_recall_curve(correct,pred_percent)
    plt.step(recall,precision,where='post')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precission')
    plt.title('Precision Recall Curve')

    plt.figure(2)
    d=np.abs(total_difference[1:,:,:]-np.median(total_difference[1:,:,:]))
    mdev=np.median(d)
    vmax=2*mdev+np.median(total_difference[1:,:,:])
    for i in range(11):
        plt.subplot(1,11,i+1)
        plt.imshow(total_difference[i+1],cmap='hot',vmin=np.min(total_difference[1:,:,:]),vmax=vmax)
        #  plt.imshow(np.log(total_difference[i+1]),cmap='hot',vmin=np.min(np.log(total_difference[1:,:,:])),vmax=np.max(np.log(total_difference[1:,:,:])))
        plt.xticks([])
        plt.yticks([])
        #  plt.imshow(total_difference[i],cmap='hot')
        plt.xlabel('%d Targets' % (int(num_tar/10)*i))
        if i==5:
            plt.title('KLD Distance for Dirichlet Distributions')
    cax=plt.axes([0.93,0.25,0.025,0.5])
    plt.colorbar(cax=cax)

    plt.show()

