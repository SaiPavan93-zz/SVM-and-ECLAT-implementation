from __future__ import division
import sys
from implementation import fim,eclat
from sklearn import  svm
from sklearn import metrics
import numpy as np

f=open("house-votes-84.data.txt","r")
file=f.readlines()
paru=[]
hundred=[]
twohundred=[]
for line in file:
    strpline = line.rstrip()
    arr = strpline.split(',')
    newline = [];
    for i in range(len(arr)):
        if (arr[i] == 'y'):
            newline.append(i)
    if (arr[0] == 'republican'):
        newline.append(100)
    else:
        newline.append(200)
    sys.stdout = open("assign4b-iusaichoud.txt", "w")
    #print(*newline, sep="," )
    a=[str(x) for x in newline]
    paru.append(a)
    #a=" ".join(str(x) for x in newline)
print("The data is :")
print(paru)

first=fim(paru,'s',supp=20)
print("\nFrequent item sets are",*first,sep='\n')
print("\nFrequent Item sets with 20% support\n",len(first))
values=[]
for i in range(len(first)):
    values.append(first[i][1])
values=sorted(list(set(values)),reverse=True)
values=values[0:10]
ans=[]
for each in values:
    for i in range(len(first)):
        if (first[i][1]==each):
            ans.append(first[i])
print("\nTop Ten Item sets with Highest Support value are\n",*ans,sep='\n')

normal=[]
for each in first:
    l=[int(x) for x in each[0]]
    l=(sorted(l))
    l.append(each[1])
    if(l[-2]==100):
        hundred.append(l)
    elif(l[-2]==200):
        twohundred.append(l)
    else:
        normal.append(l)
print("\nFrequent itemsets with 100 :",len(hundred),"\n\nFrequent itemsets with 200 :\n",len(twohundred))
hundred1=[[str(x) for x in y]for y in hundred]
for each in hundred1:
    each[-1]=int(each[-1])
twohundred1=[[str(x) for x in y]for y in twohundred]
for each in twohundred1:
    each[-1]=int(each[-1])
normal1=[[str(x) for x in y]for y in normal]
for each in normal1:
    each[-1]=int(each[-1])


def fifth(hundred1,normal1):
    sub=[]
    #print(normal1)
    for each in hundred1:
        a=each[:-2]
        for element in normal1:
            if(element[:-1]==a):
                sub.append((element[:-1],"----->100",each[-1]/element[-1]))
    return(sub)

def seventh(twohundred1,normal1):
    sub=[]
    #print(normal1)
    for each in twohundred1:
        a=each[:-2]
        for element in normal1:
            if(element[:-1]==a):
                sub.append((element[:-1],"----->200",each[-1]/element[-1]))
    return(sub)

def getKey1(item):
    return item[2]

ans=fifth(hundred1,normal1)
third_sorted = (sorted(ans, key=getKey1, reverse=True))
print("Top 10 association rules where the rule head is 100",*third_sorted[0:10],sep='\n')
greaterthan75=[]
for each in third_sorted:
    if(each[2]>0.75):
        greaterthan75.append(each)
print("\n\nRules with head 100 with confidence greater than 75% :",*greaterthan75,sep='\n')
print("\nNo.of Rules with head 100 with confidence greater than 75% :",len(greaterthan75))

ans1=seventh(twohundred1,normal1)
fourth_sorted = (sorted(ans1, key=getKey1, reverse=True))
print("Top 10 association rules where the rule head is 200",*fourth_sorted[0:10],sep='\n')
greaterthan751=[]
for each in fourth_sorted:
    if(each[2]>0.75):
        greaterthan751.append(each)

print("\n\nRules with head 200 with confidence greater than 75% :",*greaterthan751,sep='\n')
print("\nNo.of Rules with head 200 with confidence greater than 75% :",len(greaterthan751))

def generate(a):
    k=[]
    for each in a:
        l = [0] * 16
        for j in each:
            for n, i in enumerate(l):
                if (n == j - 1):
                    l[n] = 1
                elif (i == 1):
                    l[n] = 1
                else:
                    l[n] = 0
        k.append(l)
    return(k)

#greaterthan75=100,greaterthan751=200
gd1=[[int(y)for y in x[0]]for x in greaterthan75]
i1=generate(gd1)
gd2=[[int(y)for y in x[0]]for x in greaterthan751]
i2=generate(gd2)
for each in i1:
    each.insert(0,100)
for each in i2:
    each.insert(0,200)


def svm_implementation(train,cl,test,cls,c):
    acc=[]
    accu=[]
    clf = svm.SVC(kernel='linear',C=c)
    clf.fit(train, cl)
    y_pred = clf.predict(test)
    #print(y_pred,"\n",len(y_pred))
    #print("Accuracy:", metrics.accuracy_score(cls, y_pred))
    a=(metrics.accuracy_score(cls, y_pred))
    return  a





def folds(i1,i2):
    lt1=int((len(i1)-1)/4)
    lt2=int((len(i2)+1)/4)
    idata1=i1[0:lt1]
    idata1.extend(i2[0:lt2])
    idata2=i1[lt1:(2*lt1)]
    idata2.extend(i2[lt2:(2*lt2)])
    idata3=i1[(2*lt1):(3*lt1)]
    idata3.extend(i2[(2*lt2):(3*lt2)])
    idata4=i1[(3*lt1):]
    idata4.extend(i2[(3*lt2):])
    traindata1=np.array(idata2+idata3)
    testdata1=np.array(idata4)
    traindata2=np.array(idata3+idata4)
    testdata2=np.array(idata2)
    traindata3=np.array(idata2+idata4)
    testdata3=np.array(idata3)
    c = list(np.arange(0.1, 5, 0.5))
    acc = []
    validdata=np.array(idata1)
    accu=[]
    for each in c:
        a = svm_implementation(traindata1[:,1:17], traindata1[:,0], validdata[:,1:17], validdata[:,0], each)
        acc.append(a)
    #print(acc)
    print("The C value is", c[acc.index(max(acc))])
    print("1 FOLD")
    C = c[acc.index(max(acc))]
    first = svm_implementation(traindata1[:,1:17], traindata1[:,0], testdata1[:,1:17], testdata1[:,0], C)
    accu.append(first)
    acc = []
    for each in c:
        a = svm_implementation(traindata2[:,1:17], traindata2[:,0], validdata[:,1:17], validdata[:,0], each)
        acc.append(a)
        # print(max(acc),acc.index(max(acc)))
    print("The C value is", c[acc.index(max(acc))])
    print("2 FOLD")
    C = c[acc.index(max(acc))]
    second = svm_implementation(traindata2[:,1:17], traindata2[:,0], testdata2[:,1:17], testdata2[:,0], C)
    accu.append(second)
    acc = []
    for each in c:
        a = svm_implementation(traindata3[:,1:17], traindata3[:,0], validdata[:,1:17], validdata[:,0], each)
        acc.append(a)
        # print(max(acc),acc.index(max(acc)))
    print("The C value is", c[acc.index(max(acc))])
    print("3 FOLD")
    C = c[acc.index(max(acc))]
    third = svm_implementation(traindata3[:,1:17], traindata3[:,0], testdata3[:,1:17], testdata3[:,0], C)
    accu.append(third)
    print("Accuracy over 3 folds\n", accu)
    print("mean is\n",np.mean(accu),"\nstd is\n", np.std(accu))


folds(i1,i2)












