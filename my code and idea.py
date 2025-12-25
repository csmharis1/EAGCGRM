#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('UCDMyoVerseSubject0Data.npy')
np.savetxt('UCDMyoVerseSubject0Data.csv', data1, delimiter=',')
data2 = np.load('UCDMyoVerseSubject0Labels.npy')
np.savetxt('UCDMyoVerseSubject0Labels.csv', data2, delimiter=',')


# In[8]:


import numpy as np
import openpyxl

# Load the 3D array
data = np.load('UCDMyoVerseSubject0Data.npy')

# Save each 2D slice to a separate CSV or Excel file
for i, slice_2d in enumerate(data):
    np.savetxt(f'UCDMyoVerseSubject0Data{i}.csv', slice_2d, delimiter=',')
print('done')


# In[10]:


def splitMatrix(matrix):
    diagonalMatrix = np.diag(np.log(np.diag(matrix)))
    strictlyLowerTriangularMatrix = np.tril(matrix, k = -1)
    return diagonalMatrix, strictlyLowerTriangularMatrix


# In[11]:




def MEAN(matrix):
    numberMatrices, n, _ = np.shape(matrix)
    lowerMatrices = np.zeros((numberMatrices, n, n))
    diagonalMatrices = np.zeros((numberMatrices, n, n))
    for i in range(numberMatrices):
        chol = np.linalg.cholesky(matrix[i, :, :])
        cholD, cholL = splitMatrix(chol)
        lowerMatrices[i, :, :] = cholL
        diagonalMatrices[i, :, :] = cholD
    
    meanL = np.mean(lowerMatrices, axis = 0)
    meanD = np.diag(np.exp(np.diag(np.mean(diagonalMatrices, axis = 0))))

    meanF = meanL + meanD
    return meanF
print("done")


# In[12]:


def distanceMatrices(matrix1, matrix2):
    chol1 = np.linalg.cholesky(matrix1)
    chol2 = np.linalg.cholesky(matrix2)
    chol1D, chol1L = splitMatrix(chol1)
    chol2D, chol2L = splitMatrix(chol2)
    distanceL = np.square(np.linalg.norm(chol1L - chol2L, 'fro'))
    distanceD = np.square(np.linalg.norm(chol1D - chol2D, 'fro'))
    distance = np.sqrt(distanceL + distanceD)
    return distance


# In[13]:


covarianceMatrices = np.load("UCDMyoVerseSubject0Data.npy")
Labels = np.load("UCDMyoVerseSubject0Labels.npy")
print(covarianceMatrices.shape)
print(Labels.shape)
print(covarianceMatrices)


# In[14]:


print(Labels.shape)


# In[7]:




covarianceMatrices = np.load("UCDMyoVerseSubject0Data.npy")
Labels = np.load("UCDMyoVerseSubject0Labels.npy")
print(covarianceMatrices.shape)
print(Labels.shape)

numberGestures = 10
trialsPerGesture = 36
numberChannels = 12

Indices =  {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i in range(len(Labels)):
    Indices[Labels[i]].append(i)

covarianceMatrixByLabels = np.zeros((numberGestures, trialsPerGesture, numberChannels, numberChannels))
for i in range(numberGestures):
    for j in range(trialsPerGesture):
        covarianceMatrixByLabels[i, j] = covarianceMatrices[Indices[i][j]]

trainCentroid = np.zeros((numberGestures, numberChannels, numberChannels))
for i in range(numberGestures):
    trainCentroid[i, :, :] = MEAN(covarianceMatrixByLabels[i, :18])

testFeatures = np.zeros((numberGestures * trialsPerGesture // 2, numberChannels, numberChannels))
testLabels = np.zeros((numberGestures * trialsPerGesture // 2))
count = 0
for i in range(numberGestures):
    testFeatures[count:count + 18] = covarianceMatrixByLabels[i, 18:]
    testLabels[count:count + 18] = [i] * 18
    count += 18

predictLabels = np.zeros((numberGestures * trialsPerGesture // 2))
for k in range(numberGestures * trialsPerGesture // 2):
    distances = np.zeros((numberGestures))
    for m in range(numberGestures):
        distances[m] = distanceMatrices(testFeatures[k], trainCentroid[m]@trainCentroid[m].transpose())
    predictLabels[k] = np.argmin(distances)

correct = (predictLabels == testLabels)
print("Mean accuracy is: ", np.mean(correct))


# In[8]:




covarianceMatrices = np.load("ninaproSubject0Data.npy")
Labels = np.load("ninaproSubject0Labels.npy") - 1
print(covarianceMatrices.shape)
print(Labels.shape)

numberGestures = 17
numberChannels = 12
numberRepeat = 6

trainFeatures = np.zeros((4 * numberGestures, numberChannels, numberChannels))
trainLabels = np.zeros((4 * numberGestures))

testFeatures = np.zeros((2 * numberGestures, numberChannels, numberChannels))
testLabels = np.zeros((2 * numberGestures))

repeatTrain = [0, 2, 3, 5]
repeatTest = [1, 4]


index = 0
for gesture in range(numberGestures):
    for repeat in repeatTrain:
        trainFeatures[index] = covarianceMatrices[gesture * numberRepeat + repeat]
        trainLabels[index] = gesture
        index += 1


index = 0
for gesture in range(numberGestures):
    for repeat in repeatTest:
        testFeatures[index] = covarianceMatrices[gesture * numberRepeat + repeat]
        testLabels[index] = gesture
        index += 1

trainCentroid = np.zeros((numberGestures, numberChannels, numberChannels))
for i in range(numberGestures):
    trainCentroid[i, :, :] = MEAN(trainFeatures[i * 4: i * 4 + 4])


predictLabels = np.zeros((2 * numberGestures))
for k in range(2 * numberGestures):
    distances = np.zeros((numberGestures))
    for m in range(numberGestures):
        distances[m] = distanceMatrices(testFeatures[k], trainCentroid[m]@trainCentroid[m].transpose())
    predictLabels[k] = np.argmin(distances)

print("Mean accuracy is: ", np.mean(predictLabels == testLabels))

