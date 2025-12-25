import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, precision_score, recall_score, r2_score

# Function to split the matrix into diagonal and lower triangular parts
def splitMatrix(matrix):
    diagonalMatrix = np.diag(np.log(np.diag(matrix)))
    strictlyLowerTriangularMatrix = np.tril(matrix, k=-1)
    return diagonalMatrix, strictlyLowerTriangularMatrix

# Function to compute the Frechet mean of Cholesky matrices
def MEAN(matrix):
    numberMatrices, n, _ = np.shape(matrix)
    lowerMatrices = np.zeros((numberMatrices, n, n))
    diagonalMatrices = np.zeros((numberMatrices, n, n))
    for i in range(numberMatrices):
        chol = np.linalg.cholesky(matrix[i, :, :])
        cholD, cholL = splitMatrix(chol)
        lowerMatrices[i, :, :] = cholL
        diagonalMatrices[i, :, :] = cholD
    
    meanL = np.mean(lowerMatrices, axis=0)
    meanD = np.diag(np.exp(np.diag(np.mean(diagonalMatrices, axis=0))))
    meanF = meanL + meanD
    return meanF

# Function to compute distance between matrices
def distanceMatrices(matrix1, matrix2):
    chol1 = np.linalg.cholesky(matrix1)
    chol2 = np.linalg.cholesky(matrix2)
    chol1D, chol1L = splitMatrix(chol1)
    chol2D, chol2L = splitMatrix(chol2)
    distanceL = np.square(np.linalg.norm(chol1L - chol2L, 'fro'))
    distanceD = np.square(np.linalg.norm(chol1D - chol2D, 'fro'))
    distance = np.sqrt(distanceL + distanceD)
    return distance

# Load data
covarianceMatrices = np.load("UCDMyoVerseSubjectsData.npy")
Labels = np.load("UCDMyoVerseSubjectsLabels.npy")
print(covarianceMatrices.shape)
print(Labels.shape)

# Reshape covariance matrices into vectors for ML classifiers
n_samples = covarianceMatrices.shape[0]
data_flattened = covarianceMatrices.reshape(n_samples, -1)

# Split the data for ML classifiers
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(data_flattened, Labels, test_size=0.2, random_state=42)

# Initialize and train ML classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

# Dictionary to store metrics of each classifier
metrics_results = {}

# Evaluate each ML classifier
for name, clf in classifiers.items():
    clf.fit(X_train_ml, y_train_ml)
    y_pred_ml = clf.predict(X_test_ml)
    accuracy = accuracy_score(y_test_ml, y_pred_ml)
    nmi = normalized_mutual_info_score(y_test_ml, y_pred_ml)
    precision = precision_score(y_test_ml, y_pred_ml, average='macro')
    recall = recall_score(y_test_ml, y_pred_ml, average='macro')
    r2 = r2_score(y_test_ml, y_pred_ml)
    metrics_results[name] = [accuracy, nmi, precision, recall, r2]

numberGestures = 10
trialsPerGesture = 36
numberChannels = 12

Indices = {i: [] for i in range(numberGestures)}
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
        distances[m] = distanceMatrices(testFeatures[k], trainCentroid[m] @ trainCentroid[m].transpose())
    predictLabels[k] = np.argmin(distances)

macc = np.mean(predictLabels == testLabels)
nmi = normalized_mutual_info_score(testLabels, predictLabels)
precision = precision_score(testLabels, predictLabels, average='macro')
recall = recall_score(testLabels, predictLabels, average='macro')
r2 = r2_score(testLabels, predictLabels)

metrics_results["Proposed Method"] = [macc, nmi, precision, recall, r2]

# Convert results to DataFrame and save to CSV
metrics_df = pd.DataFrame(metrics_results, index=['Accuracy', 'NMI', 'Precision', 'Recall', 'R2']).T
metrics_df.to_csv('myverse_model_metrics.csv', index=True)

print("Results saved to 'myverse_model_metrics.csv'. Here are the metrics:")
print(metrics_df)
