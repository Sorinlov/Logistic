'''
2018.09.18
logistic function / sigmoid
gradDescent
'''

from math import exp
from numpy import mat, zeros

def sigmoid(x):
    return 1/(1 + exp(-x))

def gradDescent(dataset, labels, alpha = 0.001, maxcycle = 500):
    datamat = mat(dataset)
    labelsmat = labels
    m,n = shape(datamat)

    w = zeros((n,1))

    for k in range(maxcycle):
        fx = sigmoid(datamat * w)
        error = labelsmat - fx
        w = w + alpha * datamat.transpose() * error

    return w
