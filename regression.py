import numpy as np
import matplotlib.pyplot as plt

#Implementation of the Gradient Descent algoirithm, Batch
def batchGD(x, y, tol, alpha):
    q = len(y)
    x = np.c_[np.ones(x.shape[0]), x]
    w = np.array(np.ones(2))
    lossGrad = 1e12
    epochs = 1
    while lossGrad > tol:
        hypothesis = np.dot(x, w)
        error = y - hypothesis
        grad = np.dot(error.transpose(), x)/q
        lossGrad = np.sum(grad**2)
        w = w + alpha*grad
        epochs = epochs + 1
    return w, epochs

#Implementation of the Gradient Descent algoirithm, Stochastic (online)
def stochasticGD(x , y, w, alpha):
    x = [1, x]
    hypothesis = np.dot(x, w)
    error = y - hypothesis
    grad = np.dot(error.transpose(), x)
    w = w + alpha*grad
    return w