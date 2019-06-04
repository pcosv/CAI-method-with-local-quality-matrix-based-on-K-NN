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

import numpy as numpy
import scipy.optimize as optimize
import numpy as np
from scipy.spatial import distance

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
        distance = distance + pow((Decimal(instance1[x]) - Decimal(instance2[x])), 2)
    return math.sqrt(distance)


# retorna vizinhos de uma dada instância dado o k
def getNeighborsKnn(trainSet, instance, k):
    distances = []
    length = len(trainSet) - 1
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
        distance = euclidianDistance(y, x)
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


def minimization():
    resultPredicted = [[0.6, 0.2, 0.2], [0.4, 0.3, 0.3], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.1, 0.1, 0.7]]
    trueResult = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
    penalizingFactor = [0.6, 0.5, 0.2, 0.1, 0.1]

    def f(x):
        soma = 0
        for i in range(len(resultPredicted)):
            print(np.reshape(x, (3,3)).transpose())
            soma = soma + (penalizingFactor[i]*(distance.euclidean(np.dot(resultPredicted[i], np.reshape(x, (3,3)).transpose()), trueResult[i])))
        print(soma)
        return soma

    initial_guess = np.identity(3)
    result = optimize.minimize(f, initial_guess, method='SLSQP')
    if result.success:
        fitted_params = result.x
        print(np.reshape(fitted_params, (3,3)))
    else:
        raise ValueError(result.message)


def constraint(x):
    dimension = x.shape
    for x in range(dimension[0]):
        pass

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 4: Estimate quality matrix 𝜷
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# aqui sera chamada a função para minimizar a função objetivo respeitando as constraints
# def minimizeFunction(n_classes):
#     initialGuess = np.identity(n_classes)
#     result = optimize.minimize(objectiveFunction, initialGuess)
#     return result


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 5: Correct the classification result of object 𝐲
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def main():

    # print(np.identity(3))

    a = [1, 2, 3]
    b = [3, 4, 5]
    initialGuess = np.identity(3)
    #print(initialGuess.transpose())

    #dist = numpy.linalg.norm(a - b)
    #print(dist)

    #dst = distance.euclidean(a, b)
    #print(dst)

    #print(np.dot(a, initialGuess.transpose()))

    minimization()



main()
