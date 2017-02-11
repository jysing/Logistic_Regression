import numpy as np
import matplotlib.pyplot as plt

def linRegression(x, y, tol, alpha, stoch):
    if stoch == True:
        return stochasticGD(x, y, tol, alpha)
    else:
        return batchGD(x, y, tol, alpha)

#Implementation of the Greatest Descent algoirithm, Batch
def batchGD(x, y, tol, alpha):
    q = len(y)
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

#Implementation of the Greatest Descent algoirithm, Stochastic
def stochasticGD(x , y, tol, alpha):
    q = len(y)
    w = np.array(np.ones(2))
    lossGrad = 1e12
    epochs = 1
    while lossGrad > tol:
        x, y = unison_shuffle(x, y)
        for i in range(q):
            w = w + alpha*(y[i] - np.dot(x[i],w))*x[i]
        hypothesis = np.dot(x,w)
        error = y - hypothesis
        grad = np.dot(error.transpose(), x)/q
        lossGrad = np.sum(grad**2)
        epochs = epochs + 1
    return w, epochs

#Shuffles x,y while preserving their relationship (x defined by [ones xData])
def unison_shuffle(x, y):
    p = np.random.permutation(len(y))
    return x[p,:], y[p]

#Very rudimentary check on whether the Gradient Descent works
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = np.c_[np.ones(x.shape[0]), x]
    y = np.array([5, 7, 11, 15, 18, 22, 29, 31, 35, 36])
    tol = 0.01
    alpha = 0.005
    useStoch = False
    w, epochs = linRegression(x, y, tol, alpha, useStoch)
    print(w, epochs)
    line = np.dot(x,w)
    print(x,y)
    plt.plot(x[:,1], line, 'g--', x[:,1], y, 'rs')
    plt.show()
