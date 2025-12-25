
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def splitMatrix(matrix):
    diagonalMatrix = np.diag(np.log(np.diag(matrix)))
    strictlyLowerTriangularMatrix = np.tril(matrix, k = -1)
    return diagonalMatrix, strictlyLowerTriangularMatrix


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

covarianceMatrices = np.load("UCDMyoVerseSubjectsData.npy")
Labels = np.load("UCDMyoVerseSubjectsLabels.npy")

unique_subjects = np.unique(Labels)

# Display the number of unique subjects
num_subjects = len(unique_subjects)
print(f"Number of unique subjects in the dataset: {num_subjects}")

# Optionally, display the unique subject IDs
print("Unique subject IDs:", unique_subjects)


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












predictProba = np.zeros((numberGestures * trialsPerGesture // 2, numberGestures))

for k in range(numberGestures * trialsPerGesture // 2):
    distances = np.zeros((numberGestures))
    for m in range(numberGestures):
        distances[m] = distanceMatrices(testFeatures[k], trainCentroid[m] @ trainCentroid[m].transpose())
    
    # Convert distances to probabilities
    inverse_distances = 1 / (distances + 1e-10)  # Avoid division by zero
    probabilities = inverse_distances / np.sum(inverse_distances)  # Normalize
    predictProba[k] = probabilities

predictLabels = np.argmax(predictProba, axis=1)  # Update predicted labels using probabilities





from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Confusion Matrix
cm = confusion_matrix(testLabels, predictLabels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(numberGestures))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=900)  # Save figure
plt.show()

# Binarize the test labels for multiclass evaluation
testLabels_bin = label_binarize(testLabels, classes=np.arange(numberGestures))

# Precision-Recall Curve
plt.figure(figsize=(12, 6))
for i in range(numberGestures):
    precision, recall, _ = precision_recall_curve(testLabels_bin[:, i], predictProba[:, i])
    plt.plot(recall, precision, label=f"Class {i}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Multiclass)")
plt.legend(loc="lower left")
plt.grid()
plt.savefig("precision_recall_curve.png", dpi=900)  # Save figure
plt.show()

# Receiver Operating Characteristic (ROC) Curve
plt.figure(figsize=(12, 6))
for i in range(numberGestures):
    fpr, tpr, _ = roc_curve(testLabels_bin[:, i], predictProba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multiclass)")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("roc_curve.png", dpi=900)  # Save figure
plt.show()




