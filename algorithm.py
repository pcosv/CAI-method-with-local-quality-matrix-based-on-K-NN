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
import scipy.optimize as optimize
import numpy as np
from scipy.optimize import Bounds, LinearConstraint
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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 4: Estimate quality matrix 𝜷
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def minimization():
    resultPredicted = [[0.6, 0.2, 0.2], [0.4, 0.3, 0.3], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.1, 0.1, 0.7]]
    trueResult = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
    penalizingFactor = [0.6, 0.5, 0.2, 0.1, 0.1]

    def f(x):
        soma = 0
        for i in range(len(resultPredicted)):
            print(np.reshape(x, (3,3)).transpose())
            soma = soma + (penalizingFactor[i]*(distance.euclidean(np.dot(resultPredicted[i], np.reshape(x, (3,3)).transpose()), trueResult[i])))
        #print(soma)
        return soma

    # cada linha da matriz deve somar 1
    def constraint(x):
        dimension = x.shape
        soma = []
        for a in range(dimension[0]):
            sum = 0
            for b in range(dimension[0]):
                #print(x[a][b])
                sum = sum + x[a][b]
            soma.append(sum)
        return soma


    initial_guess = np.identity(3)
    dimension = initial_guess.shape
    n = dimension[0]

    # o valor de cada elemendo da matriz deve estar entre 0 e 1
    bounds = [(0, 1)]*(n*n)

    eq_cons = {'type': 'eq',
               'fun': lambda x: np.array(constraint),
               'jac': lambda x: np.array(np.ones((3, 3)))}

    def rosen_der(x):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
        der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2] ** 2)

        return der

    #result = optimize.minimize(f, initial_guess, method='SLSQP', jac=rosen_der, bounds=bounds)
    result = optimize.minimize(f, initial_guess, method='BFGS', bounds=bounds)
    print(np.reshape(result.x, (3,3)))

    #print(constraint(np.reshape(result.x, (3,3))))


    return np.reshape(result.x, (3,3))



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Step 5: Correct the classification result of object 𝐲
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def correctClassification(b, y, u):
    return (y*(np.dot((np.reshape(b, (len(u),len(u))).transpose()),(u.transpose())))) + ((1 - y)*(u.transpose()))

def main():
    b = minimization()
    u = np.array([0.4, 0.35, 0.25])
    #b = np.array([[1, 0, 0], [0.0849, 0.9151, 0], [0.6148, 0, 0.3852]])
    print(b)

    finalResultCorrected = correctClassification(b, 0.9, u)
    print(finalResultCorrected)


main()
