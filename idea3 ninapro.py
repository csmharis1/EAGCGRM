

import numpy as np
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def splitMatrix(matrix):
    diagonalMatrix = np.diag(np.log(np.diag(matrix)))
    strictlyLowerTriangularMatrix = np.tril(matrix, k = -1)
    return diagonalMatrix, strictlyLowerTriangularMatrix


# In[3]:



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


# In[4]:


def distanceMatrices(matrix1, matrix2):
    chol1 = np.linalg.cholesky(matrix1)
    chol2 = np.linalg.cholesky(matrix2)
    chol1D, chol1L = splitMatrix(chol1)
    chol2D, chol2L = splitMatrix(chol2)
    distanceL = np.square(np.linalg.norm(chol1L - chol2L, 'fro'))
    distanceD = np.square(np.linalg.norm(chol1D - chol2D, 'fro'))
    distance = np.sqrt(distanceL + distanceD)
    return distance



covarianceMatrices = np.load("ninaproSubjectData.npy")
Labels = np.load("ninaproSubjectLabels.npy") - 1
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

