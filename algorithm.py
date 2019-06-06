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
import seaborn as sns;
from sklearn.svm import SVC
import matplotlib.pyplot as plt

sns.set()
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
        if min(distances) == 0:
            relativeDistances.append(0)
        else:
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

def minimization(resultPredicted, trueResult, penalizingFactor, n_classes):

    def f(x):
        soma = 0
        for i in range(len(resultPredicted)):
            bt = np.reshape(x, (n_classes,n_classes)).transpose()
            rp = np.reshape(np.array(resultPredicted[i]), (n_classes, 1))
            tr = np.reshape(np.array(trueResult[i]), (n_classes, 1))
            soma = soma + (penalizingFactor[i]*(distance.euclidean(np.dot(bt, rp), tr)))
        #print(soma)
        return soma

    initial_guess = np.array(np.identity(n_classes))
    dimension = initial_guess.shape
    bounds = [(0, 1)]*(n_classes*n_classes)

    eq_cons = ({'type': 'eq',
               'fun': lambda x: np.array([x[0] + x[1] + x[2] - 1,
                                          x[3] + x[4] + x[5] - 1,
                                          x[6] + x[7] + x[8] - 1])})

    result = optimize.minimize(f, initial_guess, method='slsqp', constraints=eq_cons, bounds=bounds)
    #print(np.reshape(np.array(result.x), (n_classes,n_classes)))
    return (np.reshape(np.array(result.x), (n_classes,n_classes)))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 5: Correct the classification result of object 𝐲
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def correctClassification(b, y, u):
    return (y*(np.dot((np.reshape(b, (len(u[0]),len(u[0]))).transpose()),(u.transpose())))) + ((1 - y)*(u.transpose()))



def main():

    graphResultBefore = []
    graphResultAfter = []

    simplefilter(action='ignore', category=FutureWarning)

    # dataset
    data = pandas.read_csv('datasets/tae.csv')
    X = data.drop('UNS', axis=1)
    y = data['UNS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classes = [1, 2, 3] # tae dataset
    # classes = ['bus', 'opel', 'saab', 'van'] # ve dataset
    # classes = ['High', 'Low', 'Middle', 'very_low'] # kn dataset

    yParameter = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in yParameter:
        print(i)

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Escolha do algoritmo
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # Random Forest, SVM ou Naive Bayes
        classifier = GaussianNB()
        #classifier = RF()
        #classifier = SVC(kernel='linear', probability=True)

        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)

        accuracyBefore = metrics.accuracy_score(y_test, pred)
        graphResultBefore.append(accuracyBefore)

        print("Accuracy before optimization:", accuracyBefore)
        matBefore = confusion_matrix(pred, y_test)

        k = 5
        counter = 0
        nParameter = 5
        resultCorrected = []
        for a in range(len(X_test)):
            neighbors = getNeighborsKnn(X_train.values, X_test.values[a], k)
            distances = getDistances(neighbors, X_test.values[a])
            relativeDistances = getRelativeDistances(distances)
            penalizingFactors = getDistancePenalizingFactors(relativeDistances, nParameter)
            predNeighborsVector = classifier.predict_proba(neighbors)

            trueResult = []
            for v in range(len(neighbors)):
                for l in range(len(X_train.values)):
                    if all(neighbors[v] == X_train.values[l]):
                        trueResult.append(y_train.values[l])

            # vetor que recebe a 'ground truth'
            trueResultVector = []
            for true in trueResult:
                aux = []
                if true == classes[0]: # bus e High
                    aux = np.zeros(len(classes))
                    aux[0] = 1.0
                if true == classes[1]: # opel e Low
                    aux = np.zeros(len(classes))
                    aux[1] = 1.0
                if true == classes[2]: # saab e Middle
                    aux = np.zeros(len(classes))
                    aux[2] = 1.0
                # if true == classes[3]: # van e very_low
                #     aux = np.zeros(len(classes))
                #     aux[3] = 1.0
                trueResultVector.append(aux)

            # predição do exemplo pelo classificador
            u = classifier.predict_proba([X_test.values[a]])
            uClass = classifier.predict([X_test.values[a]])

            # vetor que recebe as novas respostas corrigidas pelo algoritmo
            resultCorrected.append(uClass[0])

            b = minimization(predNeighborsVector, trueResultVector, penalizingFactors, len(classes))
            finalPredMatrix = correctClassification(b, i, u)

            classUpdated = 0
            classFinal = 1
            #classFinal = 'van'
            for i in range(len(finalPredMatrix)):
                if finalPredMatrix[i] > classUpdated:
                    classUpdated = finalPredMatrix[i]
                    if i == 0:
                        #classFinal = 'High'
                        #classFinal = 'bus'
                        classFinal = 1
                    if i == 1:
                        #classFinal = 'Low'
                        #classFinal = 'opel'
                        classFinal = 2
                    if i == 2:
                        #classFinal = 'Middle'
                        #classFinal = 'saab'
                        classFinal = 3

            # se a classe muda, atualiza o vetor
            if uClass[0] != classFinal:
                resultCorrected[a] = classFinal
                counter = counter + 1

        # recalculando acuracia depois da otimização do resultado do algoritmo puro
        accuracyAfter = metrics.accuracy_score(y_test, resultCorrected)
        print("Accuracy after optimization:", accuracyAfter)
        matAfter = confusion_matrix(resultCorrected, y_test)

        graphResultAfter.append(accuracyAfter)

    # grafico cuja linha laranja é a acuracia resultante da aplicacao do algoritmo e o azul representa o alforitmo sem otimizacao
    plt.plot(yParameter, graphResultBefore)
    plt.plot(yParameter, graphResultAfter)
    plt.show()

main()
