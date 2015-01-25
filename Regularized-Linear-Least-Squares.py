import numpy as np
from copy import *
import matplotlib.pyplot as plt
%matplotlib inline
import random

def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def makeLLS(X,T,lamb):
    (standardizeF, unstandardizeF) = makeStandardize(X)
    X = standardizeF(X)
    (nRows,nCols) = X.shape
    X = np.hstack((np.ones((X.shape[0],1)), X))    
    penalty = lamb * np.eye(nCols+1)
    penalty[0,0]  = 0  # don't penalize the bias weight
    w = np.linalg.lstsq(np.dot(X.T,X)+penalty, np.dot(X.T,T))[0]
    return (w, standardizeF, unstandardizeF)

def useLLS(model,X):
    w, standardizeF, _ = model
    X = standardizeF(X)
    X = np.hstack((np.ones((X.shape[0],1)), X))
    return np.dot(X,w)

!head slump_test.data
import pandas
d = pandas.read_csv('slump_test.data',delimiter=',')

df = d.iloc[:,1:].values
d = df

names =  ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','SLUMP(cm)','FLOW(cm)','Compressive Strength (28-day)(Mpa)']
plt.figure(figsize=(10,10))
nrow,ncol = d.shape
for c in range(ncol):
    plt.subplot(4,4, c+1)
    plt.plot(d[:,c])
    plt.ylabel(names[c])
    
Xnames =  ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']
Tnames = ['SLUMP(cm)','FLOW(cm)','Compressive Strength (28-day)(Mpa)']

nrows = X.shape[0]
nTrain = int(round(nrows*0.8))
nTest = nrows - nTrain

rows = np.arange(nrows)
np.random.shuffle(rows)

trainIndices = rows[:nTrain] #all rows of ntrain
testIndices = rows[nTrain:]

trainIndices,testIndices
np.intersect1d(trainIndices, testIndices)

Xtrain = X[trainIndices,:] #input for training
Ttrain = T[trainIndices,:] #output for training
Xtest = X[testIndices,:]
Ttest = T[testIndices,:]

ncols = Xtrain.shape[1]
samples = 1000
lamb = 0.0
result = []
models = []
lambdas = []
YAll = []
result = []
lambs = np.linspace(0,50,20)*samples
for lamb in lambs:
    lambdaI = lamb * np.eye(ncols+1)
    lambdaI[0,0] = 0.0
    model = makeLLS(Xtrain,Ttrain,lambdaI)
    models.append(model)
    
    w = model[0]
    
    predTrain = useLLS(model,Xtrain)
    predTest = useLLS(model,Xtest)
    result.append([lamb, np.sqrt(np.mean((predTrain-Ttrain)**2,axis=0)),
                   np.sqrt(np.mean((predTest-Ttest)**2,axis=0)),
                   list(w.flatten())])
    if lamb == 0:
        fitTrain = np.hstack((Ttrain,predTrain))
        fitTest = np.hstack((Ttest,predTest))
    if lamb == lambs[-1]:
        fitTrainLast = np.hstack((Ttrain,predTrain))
        fitTestLast = np.hstack((Ttest,predTest))

lambdas = [res[0] for res in result]
rmses = np.array([res[1:3] for res in result]) ## array here for plotting
ws = np.array( [res[3] for res in result] )
ws = ws.reshape (20,8,3)

for i in range(3):
    plt.figure(figsize=(10,10))
    plt.subplot(4,2,1)
    plt.plot(fitTrain[:,0+i],fitTrain[:,3+i],'o')
    a,b = max(np.min(fitTrain[:,[0+i,3+i]],axis=0)), min(np.max(fitTrain[:,[0+i,3+i]],axis=0)) 
    plt.plot([a,b],[a,b],'r',linewidth=3)
    plt.title('Training result, with lambda = 0')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    
    plt.subplot(4,2,3)
    plt.plot(fitTest[:,0+i],fitTest[:,3+i],'o')
    a,b = max(np.min(fitTest[:,[0+i,3+i]],axis=0)), min(np.max(fitTest[:,[0+i,3+i]],axis=0)) 
    plt.plot([a,b],[a,b],'r',linewidth=3)
    plt.title('Testing result, with lambda = 0')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')


    plt.subplot(4,2,2)
    plt.plot(fitTrainLast[:,0+i],fitTrainLast[:,3+i],'o')
    a,b = max(np.min(fitTrainLast[:,[0+i,3+i]],axis=0)), min(np.max(fitTrainLast[:,[0+i,3+i]],axis=0)) 
    plt.plot([a,b],[a,b],'r',linewidth=3)
    plt.title('Training result, with lambda = ' + str(lambs[-1]))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.subplot(4,2,4)
    plt.plot(fitTestLast[:,0+i],fitTestLast[:,3+i],'o')
    a,b = max(np.min(fitTestLast[:,[0+i,3+i]],axis=0)), min(np.max(fitTestLast[:,[0+i,3+i]],axis=0))
    plt.plot([a,b],[a,b],'r',linewidth=3)
    plt.title('Testing result, with lambda = ' + str(lambs[-1]))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    
    plt.subplot(4,2,(5,6))
    plt.plot(lambs,rmses[:,:,i],'o-')
    plt.legend(('train','test'))
    plt.ylabel('RMSE')
    plt.xlabel('$\lambda$')
    
    plt.subplot(4,2,(7,8))
    plt.plot(lambdas,ws[:,:,i],'o-')
    plt.plot([0,max(lambs)], [0,0], 'k--')
    plt.ylabel('weights')
    plt.xlabel('$\lambda$')
    
   
