'''
Created on Apr 30, 2016

@author: hehehehehe
'''

import numpy as np
from sklearn import svm
import csv


sampleCount = 70000


with open('training.csv', 'rb') as f:
    reader = csv.reader(f)
    data = np.array(list(reader))

train = np.delete(data,np.s_[0,199],1)
print 'train ',len(train)
temp1 = np.delete(train,0,0)
predictData = np.delete(temp1,np.s_[:sampleCount],0)
trainData = np.delete(temp1,np.s_[sampleCount:],0)

targetData = np.delete(data,np.s_[0:199],1)
targetData = np.delete(targetData,0,0)

trainTarget = [i[0] for i in targetData][:sampleCount]
predictTarget = [i[0] for i in targetData][sampleCount:]

# print target
print 'trainData ',len(trainData),len(trainData[0])
print 'trainTarget ',len(trainTarget),len(trainTarget[0])
print 'predictData ',len(predictData),len(predictData[0])
print 'predictTarget ', len(predictTarget) ,len(predictTarget[0])

trainX=np.array(trainData,dtype='float32')
trainY=np.array(trainTarget,dtype=int)
testX=np.array(predictData,dtype='float32')
testY=np.array(predictTarget,dtype=int)

# clf = svm.LinearSVC()
clf = svm.SVC()
'''
linear: \langle x, x'\rangle.
polynomial: (\gamma \langle x, x'\rangle + r)^d. d is specified by keyword degree, r by coef0.
rbf: \exp(-\gamma |x-x'|^2). \gamma is specified by keyword gamma, must be greater than 0.
sigmoid (\tanh(\gamma \langle x,x'\rangle + r)), where r is specified by coef0.

'''
print clf.set_params(kernel='linear').fit(trainX, trainY)  
res = clf.predict(testX)

print res,sum(res),len(res)
print testY,sum(testY),len(testY)

misCount = 0
matchCount = 0
unmatchCount = 0
for i in xrange(len(res)):
    if res[i] == 1 and testY[i] == 1:
        matchCount += 1
    if res[i] == 0 and testY[i] == 1:
        unmatchCount += 1
    if res[i] != testY[i]:
        misCount+= 1
print 'Res ',misCount
print 'matchCount ', matchCount
print 'unmatchCount', unmatchCount
print 'miss classification ',misCount * 1.0 / len(res)



    
