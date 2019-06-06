"""
Pseudocode:

Inputs:
Training data set: 𝑋 = {𝐱1,…,𝐱𝑛}
Test data set: 𝑌 = {𝐲1,…,𝐲𝑠}
Given classifier: 𝐶
𝐾: number of nearest neighbors
𝜂 > 0: distance coefficient
𝛾 > 0: confident number

Outputs:
Corrected classification result 𝝁˜

Method:
for 𝑔=1 to 𝑠
step 1: Select the K-NN of 𝐲𝑔 from 𝑋;
step 2: Calculate the distance penalizing factor 𝛿𝑘 by Eq. (4);
step 3: Determine the global objective 𝜉 by Eq. (5);
step 4: Estimate quality matrix 𝜷 by minimizing Eq. (5) with constraint of Eq. (6);
step 5: Correct the classification result of object 𝐲 by Eq. (9);
step 6: return the corrected result 𝝁˜.
end */

"""

import csv
import math
import operator
from decimal import Decimal

import pandas as pandas
import scipy.optimize as optimize
import numpy as np
import sns as sns
from scipy.spatial import distance
from sklearn import metrics, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RF
import seaborn as sns; sns.set()
from warnings import simplefilter

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 1: Select the K-NN of 𝐲𝑔 from 𝑋 (training data set)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# load data
def loadDataSet(file):
    with open(file) as csvfile:
        lines = list(csv.reader(csvfile, delimiter=','))
        return lines


# função que retorna a distância euclidiana
def euclidianDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance = distance + pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# retorna vizinhos de uma dada instância dado o k
def getNeighborsKnn(trainSet, instance, k):
    distances = []
    length = len(trainSet[0]) - 1
    for x in range(len(trainSet)):
        dist = euclidianDistance(instance, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 2: Calculate the distance penalizing factor 𝛿𝑘
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def getDistances(neighbors, y):
    distances = []
    for x in neighbors:
        distance = euclidianDistance(y, x, len(neighbors[0]))
        distances.append(distance)
    return distances


def getRelativeDistances(distances):
    relativeDistances = []
    for x in distances:
        relativeDistance = (x - min(distances)) / min(distances)
        relativeDistances.append(relativeDistance)
    return relativeDistances


def getDistancePenalizingFactors(relativeDistances, n):
    penalizingFactors = []
    for x in relativeDistances:
        penalizingFactor = math.exp((-n) * x)
        penalizingFactors.append(penalizingFactor)
    return penalizingFactors


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 3: Determine the global objective 𝜉
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def classifyNeighbors(neighbor):
    # escolher classificador e classificar todos os vizinhos de y
    result = []
    return result


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 4: Estimate quality matrix 𝜷
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def minimization(resultPredicted, trueResult, penalizingFactor):
    n_classes = 4

    def f(x):
        soma = 0
        for i in range(len(resultPredicted)):
            #print(np.reshape(x, (n_classes,n_classes)).transpose())
            soma = soma + (penalizingFactor[i]*(distance.euclidean(np.dot(resultPredicted[i], np.reshape(x, (n_classes,n_classes)).transpose()), trueResult[i])))
        #print(soma)
        return soma

    # cada linha da matriz deve somar 1
    def constraint(x):
        dimension = np.reshape(x, (n_classes,n_classes))
        soma = []
        for a in range(dimension.shape[0]):
            sump = 0.0
            for b in range(dimension.shape[0]):
                var = (np.reshape(x, (n_classes,n_classes)))
                sump = sump + var[a][b]
            soma.append(sump)
        print('soma das linhas')
        print(soma)
        return soma

    initial_guess = np.array(np.identity(n_classes))

    dimension = initial_guess.shape
    bounds = [(0, 1)]*(n_classes*n_classes)

    eq_cons = ({'type': 'eq',
               'fun': lambda x: np.array([x[0] + x[1] + x[2] + x[3] - 1,
                                          x[4] + x[5] + x[6] + x[7] - 1,
                                          x[8] + x[9] + x[10] + x[11] - 1,
                                          x[12] + x[13] + x[14] + x[15] - 1])})

    result = optimize.minimize(f, initial_guess, method='slsqp', constraints=eq_cons, options={'disp': True})

    print(result)

    return np.reshape(result.x, (n_classes,n_classes))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 5: Correct the classification result of object 𝐲
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def correctClassification(b, y, u):
    return (y*(np.dot((np.reshape(b, (len(u[0]),len(u[0]))).transpose()),(u.transpose())))) + ((1 - y)*(u.transpose()))

def main():

    simplefilter(action='ignore', category=FutureWarning)

    data = pandas.read_csv('kn.csv')
    X = data.drop('UNS', axis=1)
    y = data['UNS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # y_predVector = gnb.predict_proba(X_test)
    # y_pred = gnb.predict(X_test)
    # print("Accuracy:", metrics.accuracy_score(y_test, y_predVector))
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # mat = confusion_matrix(y_pred, y_test)
    # print(mat)

    k = 5
    counter = 0
    nParameter = 5
    for a in range(len(X_test)):
        #print(X_test.values[a])
        neighbors = getNeighborsKnn(X_train.values, X_test.values[a], k)
        distances = getDistances(neighbors, X_test.values[a])
        relativeDistances = getRelativeDistances(distances)
        penalizingFactors = getDistancePenalizingFactors(relativeDistances, nParameter)
        predNeighborsVector = gnb.predict_proba(neighbors)

        trueResult = []
        for v in range(len(neighbors)):
            for l in range(len(X_train.values)):
                if all(neighbors[v] == X_train.values[l]):
                    trueResult.append(y_train.values[l])

        trueResultVector = []
        for true in trueResult:
            aux = []
            if true == 'High':
                aux = np.zeros(4)
                aux[0] = 1
            if true == 'Low':
                aux = np.zeros(4)
                aux[1] = 1
            if true == 'Middle':
                aux = np.zeros(4)
                aux[2] = 1
            if true == 'very_low':
                aux = np.zeros(4)
                aux[3] = 1
            trueResultVector.append(aux)

        u = gnb.predict_proba([X_test.values[a]])
        uClass = gnb.predict([X_test.values[a]])
        print('antes')
        print(u)
        print(uClass)
        b = minimization(predNeighborsVector, trueResultVector, penalizingFactors)
        finalResultCorrected = correctClassification(b, 0.9, u)

        classUpdated = 0
        classFinal = 'very_low'
        for i in range(len(finalResultCorrected)):
            if finalResultCorrected[i] > classUpdated:
                classUpdated = finalResultCorrected[i]
                if i == 0:
                    classFinal = 'High'
                if i == 1:
                    classFinal = 'Low'
                if i == 2:
                    classFinal = 'Middle'
        print('depois')
        print(finalResultCorrected)
        print(classFinal)
        if uClass[0] != classFinal:
            counter = counter + 1

    print(counter)

    # Random Forest
    # clf = RF()
    # clf.fit(X_train, y_train)
    # pred_pro = clf.predict_log_proba(X_test)
    # print(pred_pro)


main()
