
# coding: utf-8

# In[117]:


import sklearn
from sklearn import  svm
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
#with open(sys.argv[1], 'r') as f:
#sys.stdout=open("Model2.txt","w")
f=open("house-votes-84.data.txt","r")

data=(pd.read_csv(f,header=None))
#print(data)
data.values.tolist()
data.replace({'y':1,'n':-1,'?':0,'republican':100,'democrat':200}, inplace=True)
data.values.tolist()
print('Done')
#print(data)


# In[118]:


democrat=data[data[0]==200]
republican=data[data[0]==100]
#print(democrat)
#print(republican)


# In[139]:


[DA,DB,DC,DD]=np.array_split(democrat,4)
[RA,RB,RC,RD]=np.array_split(republican,4)
A=DA.append(RA)
A_C=A.ix[:,0]
A = A.drop([0], axis=1)
print(A)
B=DB.append(RB)
B_C=B.ix[:,0]
B = B.drop([0], axis=1)
C=DC.append(RC)
C_C=C.ix[:,0]
C = C.drop([0], axis=1)
D=DD.append(RD)
D_C=D.ix[:,0]
D = D.drop([0], axis=1)
#print(A)
#print(B)
#print(C)
#print(D)
#print(A_C)


# In[150]:


train1=A.append(B)
train1=np.asarray(train1)
test1=C
c1=A_C.append(B_C)
c1=np.asarray(c1)
#print(train1)
train2=B.append(C)
test2=A
#print(train2)
train3=C.append(A)
test3=B
#print(train3)
print('done')


# In[147]:


c=list(np.arange(0.1,5,0.5))
gamma=list(np.arange(0.1,1.1,0.1))
acc = []
classes=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
print(train1.shape)
print(A_C.shape)
for each in c:
    
    acc=[]
    accu=[]
    model = svm.SVC(kernel='linear', C=c)
    model.fit(train1, c1)
    y_pred = model.predict(test1)
    acc.append(a)
print("The C value is",c[acc.index(max(acc))])
print("1 FOLD")

