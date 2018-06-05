# Lasso_Regression_Classifier
import numpy as np
from sklearn import linear_model
from numpy import linalg as LA
from collections import Counter
from scipy.spatial import distance

class Lasso_Regression_Classifier(object):

  def __init__(self,D,L,train_label,average_win):
      self.D = D # train_samples as dictionary (feat x samples)
      self.L = L # regularization parameter
      self.num_genre=4
      self.train_label=train_label
      self.train_label_0=np.where(train_label==0, 1, 0)
      self.train_label_1=np.where(train_label==1, 1, 0)
      self.train_label_2=np.where(train_label==2, 1, 0)
      self.train_label_3=np.where(train_label==3, 1, 0)
      self.samples_per_music=1200/average_win

  def sparse_code_w(self,i):
      ####STUDENT CODE####
      clf = linear_model.Lasso(alpha=self.L, tol=0.001, selection='random')
      clf.fit(self.D,self.test_samples[i])
      w=clf.coef_
      return w

  def classifier(self,w,i):
      
      ####STUDENT CODE####
      dists=np.array([])

      dists = np.append(dists,LA.norm(self.test_samples[i]-np.dot(self.D,np.multiply(w,self.train_label_0))))
      dists = np.append(dists,LA.norm(self.test_samples[i]-np.dot(self.D,np.multiply(w,self.train_label_1))))
      dists = np.append(dists,LA.norm(self.test_samples[i]-np.dot(self.D,np.multiply(w,self.train_label_2))))
      dists = np.append(dists,LA.norm(self.test_samples[i]-np.dot(self.D,np.multiply(w,self.train_label_3))))
      genre_estimate = np.argmin(dists)
      return genre_estimate
  
  def most_common(self,lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]



  def predict(self,test_samples):
      ##sparse code learning##
      ##classifier##
      self.test_samples=test_samples  
      
      sample_prediction=[]
      for i in range(test_samples.shape[0]): # number of samples
          w=self.sparse_code_w(i)
          genre_estimate=self.classifier(w,i)
          sample_prediction=np.append(sample_prediction,genre_estimate)
   
      ###majority voting###
      print len(sample_prediction)
      steps=len(sample_prediction)/self.samples_per_music
      print(steps)
      for step in range(steps):
          majority_label=self.most_common(sample_prediction[step*self.samples_per_music:step*self.samples_per_music+self.samples_per_music])
          sample_prediction[step*self.samples_per_music:step*self.samples_per_music+self.samples_per_music]=majority_label*np.ones(self.samples_per_music)
       
      
      return sample_prediction
