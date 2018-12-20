from __future__ import division
import numpy as np
import sys
from sklearn import  svm
from sklearn import metrics

def preprocess(demo,repub):
    arr_demo=[]
    classes_demo=[]
    arr_repub=[]
    classes_repub=[]
    for row in demo:
        tempRow = []
        for each in row:
            if(each=='y'):
                each=1
            elif(each=='n'):
                each=-1
            elif(each=='?'):
                each=0
            elif(each=='democrat'):
                each=2
            elif(each=='republican'):
                each=3
            tempRow.append(each)
        classes_demo.append(tempRow[0])
        tempRow=tempRow[1:]
        arr_demo.append(tempRow)
    for row in repub:
        tempRow1 = []
        for each in row:
            if(each=='y'):
                each=1
            elif(each=='n'):
                each=-1
            elif(each=='?'):
                each=0
            elif(each=='democrat'):
                each=2
            elif(each=='republican'):
                each=3
            tempRow1.append(each)
        classes_repub.append(tempRow1[0])
        tempRow1=tempRow1[1:]
        arr_repub.append(tempRow1)
    return (arr_demo,classes_demo,arr_repub,classes_repub)

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

def svm_implementation_rbf(train,cl,test,cls,c,g):
    acc = []
    accu = []
    clf = svm.SVC(kernel='rbf', C=c,gamma=g)
    clf.fit(train, cl)
    y_pred = clf.predict(test)
    #print(y_pred, "\n", len(y_pred))
    #print("Accuracy for gaussian :", metrics.accuracy_score(cls, y_pred))
    a=(metrics.accuracy_score(cls,y_pred))
    return a

def folds(arr_demo,classes_demo,arr_repub,classes_repub):

    accu=[]
    accu1=[]
    nof_demo=int(len(arr_demo)/4)
    nof_repub=int(len(arr_repub)/4)
    tdata=arr_demo[0:nof_demo]
    tdata.extend(arr_repub[0:nof_repub])
    classes=classes_demo[0:nof_demo]
    classes.extend(classes_repub[0:nof_repub])
    tdata1=arr_demo[nof_demo:(2*nof_demo)]
    tdata1.extend(arr_repub[nof_repub:(2*nof_repub)])
    classes1=classes_demo[nof_demo:(2*nof_demo)]
    classes1.extend(classes_repub[nof_repub:(2*nof_repub)])
    tdata2=arr_demo[(2*nof_demo):(3*nof_demo)]
    tdata2.extend(arr_repub[(2*nof_repub):(3*nof_repub)])
    classes2=classes_demo[(2*nof_demo):(3*nof_demo)]
    classes2.extend(classes_repub[(2*nof_repub):(3*nof_repub)])
    tdata3=arr_demo[(3*nof_demo):]
    tdata3.extend(arr_repub[(3*nof_repub):])
    classes3=classes_demo[(3*nof_demo):]
    classes3.extend(classes_repub[(3*nof_repub):])
    traindata1 = tdata1 + tdata2
    traindata1_classes=classes1+classes2
    traindata2 = tdata2 + tdata3
    traindata2_classes=classes2+classes3
    traindata3 = tdata1 + tdata3
    traindata3_classes=classes1+classes3
    testdata1 = tdata3
    testdata2 = tdata1
    testdata3 = tdata2
    #c=[0.01,0.05,0.1,0.5,1,1.5,5]
    c=list(np.arange(0.1,5,0.5))
    gamma=list(np.arange(0.1,1.1,0.1))
    acc = []
    for each in c:

        a=svm_implementation(traindata1, traindata1_classes, tdata, classes,each)
        acc.append(a)
    #print(max(acc),acc.index(max(acc)))
    print("The C value is",c[acc.index(max(acc))])
    print("1 FOLD")
    C=c[acc.index(max(acc))]
    first = svm_implementation(traindata1, traindata1_classes, testdata1, classes3, C)
    accu.append(first)
    acc=[]
    for each in c:
        a = svm_implementation(traindata2, traindata2_classes, tdata, classes, each)
        acc.append(a)
        # print(max(acc),acc.index(max(acc)))
    print("The C value is", c[acc.index(max(acc))])
    print("2 FOLD")
    C = c[acc.index(max(acc))]
    second = svm_implementation(traindata2, traindata2_classes, testdata2, classes1, C)
    accu.append(second)

    acc=[]
    for each in c:
        a = svm_implementation(traindata3, traindata3_classes, tdata, classes, each)
        acc.append(a)
        # print(max(acc),acc.index(max(acc)))
    print("The C value is", c[acc.index(max(acc))])
    print("3 FOLD")
    C = c[acc.index(max(acc))]
    third = svm_implementation(traindata3, traindata3_classes, testdata3, classes2, C)
    accu.append(third)
    #print(*accu,sep='\n')
    print("mean is :",np.mean(accu),"\n std is :", np.std(accu))
    #print(c,gamma)
    acc=[]

    for g,cc in zip(gamma,c):
        ff= svm_implementation_rbf(traindata1, traindata1_classes, tdata, classes, cc,g)
        acc.append(ff)

    print("The C value is", c[acc.index(max(acc))])
    print("the value of G is",gamma[acc.index(max(acc))])
    print("1 FOLD")
    C = c[acc.index(max(acc))]
    G=gamma[acc.index(max(acc))]
    ff = svm_implementation_rbf(traindata1, traindata1_classes, testdata1, classes3, C,G)
    accu1.append(ff)

    acc = []

    for g, cc in zip(gamma, c):
        ff = svm_implementation_rbf(traindata2, traindata2_classes, tdata, classes, cc, g)
        acc.append(ff)

    print("The C value is", c[acc.index(max(acc))])
    print("the value of G is", gamma[acc.index(max(acc))])
    print("2 FOLD")
    C = c[acc.index(max(acc))]
    G = gamma[acc.index(max(acc))]
    ss = svm_implementation_rbf(traindata2, traindata2_classes, testdata2, classes1, C, G)
    accu1.append(ss)

    acc = []

    for g, cc in zip(gamma, c):
        ff = svm_implementation_rbf(traindata3, traindata3_classes, tdata, classes, cc, g)
        acc.append(ff)

    print("The C value is", c[acc.index(max(acc))])
    print("the value of G is", gamma[acc.index(max(acc))])
    print("3 FOLD")
    C = c[acc.index(max(acc))]
    G = gamma[acc.index(max(acc))]
    tt = svm_implementation_rbf(traindata3, traindata3_classes, testdata3, classes2, C,G)
    accu1.append(tt)
    print("mean gaussian :",np.mean(accu1),"\nstd gaussian", np.std(accu1))

def main():
    d=[]
    demo=[]
    repub=[]
    f=open("house-votes-84.data.txt","r")
    file=f.read()
    data=file.split("\n")
    for each in data:
        d.append(each.split(','))
    for each in d:
        if(each[0]=='democrat'):
            demo.append(each)
        elif(each[0]=='republican'):
            repub.append(each)
    sys.stdout = open("assign4a-iusaichoud.txt", "w")
    arr_demo,classes_demo,arr_repub,classes_repub=preprocess(demo,repub)
    folds(arr_demo,classes_demo,arr_repub,classes_repub)

if __name__=="__main__":
    main()