#!/usr/bin/env python
# coding: utf-8

# Random Forest Classifier from Decision Tree
# 
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
clf = DecisionTreeClassifier()
train = pd.read_csv("loans_train.csv")
test = pd.read_csv("loans_test.csv")



Xtr = train.drop(columns = ['default','Unnamed: 0'])
# print(Xtr)
print(" x train shape: ",Xtr.shape)
Ytr = train['default'] 

xtst = test.drop(columns = ['default','Unnamed: 0'])
ytst = test['default']
print(" x test shape: ",xtst.shape)

clf = clf.fit(Xtr,Ytr)
y_predtest = clf.predict(xtst)
print(y_predtest.shape)
print("y test shape",ytst.shape)
print("testing data Set accuracy", accuracy_score(ytst, y_predtest))
ypredtrain = clf.predict(Xtr)
print("Training data accuracy",accuracy_score(Ytr, ypredtrain))
print("Validation",cross_val_score(clf,Xtr,Ytr,cv= 5,scoring ='accuracy').mean())
#tree.plot_tree(clf)
training_accuracy = []
tree_depth =[]
for depth in range(2,10):
    clf = DecisionTreeClassifier(max_depth = depth)
    clf.fit(Xtr,Ytr)
    ypredtrain = clf.predict(Xtr)
    training_acc = accuracy_score(Ytr,ypredtrain)


# In[2]:


# datafrxtst = pd.DataFrame(xtst)
# print(datafrxtst)


# 

# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

def cs360_cv(your_knn_model,x_data,y_data,num_folds=5,avg_acc=False):
    x_data=np.array(x_data)
    y_data = np.array(y_data)
    kf=KFold(n_splits=num_folds,shuffle=True)
    fold_accuracies = []
    for train_index, test_index in kf.split(x_data,y_data):
        cross_val_X_train_data = x_data[train_index]
        cross_val_X_test_data = x_data[test_index]
        cross_val_y_train_data = y_data[train_index]
        cross_val_y_test_data = y_data[test_index]
        your_knn_model.fit(cross_val_X_train_data,cross_val_y_train_data)
        preds = your_knn_model.predict(cross_val_X_test_data)
#         print(len(cross_val_y_test_data))
#         print(len(y_data))
        fold_accuracies.append(sum([1 for i,j in zip(preds,cross_val_y_test_data) if i==j])/len(y_data)*num_folds)
    if avg_acc:
        return sum(fold_accuracies)/len(fold_accuracies)
    return fold_accuracies 

class randomForest:
    def __init__(self,num_estimator,max_depth,pct_ft):
        self.num_estimator = num_estimator
        self.max_depth = max_depth
        self.pct_ft = pct_ft
        self.clf = [DecisionTreeClassifier(max_depth = self.max_depth,criterion = 'entropy') for x in range(0,self.num_estimator)]
        self.y_pred = [np.array(()) for i in range(0, self.num_estimator)]
        self.sel_col = [pd.DataFrame() for i in range(0,num_estimator)]
#         self.tree = tree

            
    def fit(self,Xtr,Ytr):
#         print(self.clf)
        y_predtest =[]
#         Xtr = train.drop(columns = ['default','Unnamed: 0'])
#         Ytr = train['default'] 
        
#         for tree in self.clf:
        for i in range(0,len(self.clf)):
#             print(type(Xtr))
            Xtr=pd.DataFrame((Xtr))
#             print(type(Xtr))
            self.sel_col[i] = Xtr.sample(frac = self.pct_ft ,axis = 1)     # for choosing random col
            self.clf[i] = self.clf[i].fit(self.sel_col[i],Ytr)

    def predict(self,xtest):
        i = 0
#         print(xtest.shape)
        xtest = pd.DataFrame(xtest)
#         for cols in xtest.columns:
#             print(cols)
        for tree in self.clf:
    #            
            xtest_some_cols = xtest[:][self.sel_col[i].columns]
#             print(xtest_some_cols.shape)
            z = tree.predict(xtest_some_cols)

            self.y_pred[i] = z.reshape(-1,1)
    #             print(len(self.y_pred[:][i]))
            i+=1
        b =  np.sum(self.y_pred,axis = 0)
#         print(b)

        cc = [1 if x >= 3 else 0 for x in b]
        return cc

for depth in range(2,10):
    a = randomForest(5,depth, 0.5)
#     a.fit()
#     Ypr = a.predict()
    print('depth ', depth)
    print(cs360_cv(a,Xtr,Ytr,num_folds=5,avg_acc=False))
# print(a.num_estimator)


# In[ ]:





# In[4]:


import random
f = [1,2,3,4,5]
for i in f:
    sampled_list = random.sample(f, 3)
    print(sampled_list)


# In[ ]:




