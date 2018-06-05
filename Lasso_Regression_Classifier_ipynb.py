
# coding: utf-8

# # Music Genre Classifier by using Lasso regression.
# 
# 

# In[6]:


import numpy as np

Music_Genre=["classical","jazz","pop","rock"]

#load train and val data set
train_data=np.load("./feature_data/train/train_sample.npy")
train_label=np.load("./feature_data/train/train_label.npy")
val_data=np.load("./feature_data/val/val_sample.npy")
val_label=np.load("./feature_data/val/val_label.npy")

print len(train_data[0])
# In[7]:


L = [0.1,1,10,13,15,17,21,30,100]


# # Please implement Lasso_Regression_Classifier 
# Please implement Classifier in the file : Lasso_Regression_Classifier.py
from cs536_3.models import Lasso_Regression_Classifier

average_win=20 #Student will change this#
print "Average win is : %d"%(average_win)
list_val_accuracy = np.array([])

for l in L: 
    LRC = Lasso_Regression_Classifier(train_data.transpose(),l,train_label,average_win)
    val_pred=LRC.predict(val_data)
    val_acc = np.sum(val_pred==val_label)/(1.0*(len(val_pred)))

    list_val_accuracy=np.append(list_val_accuracy,val_acc)
    
    print("Regularization_L:".format(),"Validation Accuracy is".format(),l,val_acc)
# # Please select My L and check the test performance.

print list_val_accuracy
chosen_l = L[np.argmax(list_val_accuracy)]

print "chosen_l %d"%(chosen_l)
# In[8]:


## Test data load
test_data=np.load("./feature_data/test/test_sample.npy")
test_label=np.load("./feature_data/test/test_label.npy")


# In[9]:


My_L = chosen_l #student answer
LRC = Lasso_Regression_Classifier(train_data.transpose(),My_L,train_label,average_win)
test_pred=LRC.predict(test_data)
test_acc = np.sum(test_pred==test_label)/(1.0*len(test_pred))
print("Regularization_L:".format(),"Test Accuracy is".format(),My_L,test_acc)

