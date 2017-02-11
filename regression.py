import numpy as np
import matplotlib.pyplot as plt

def batchGradientDescent(x, y, tol, alpha):
    q = len(y)
    w = np.array(np.ones(2))
    lossGrad = 100
    iterations = 1
    while lossGrad > tol:
        hypothesis = np.dot(x, w)
        error = y - hypothesis
        grad = np.dot(error.transpose(), x)/q
        lossGrad = np.sum(grad**2)
        w = w + alpha*grad
        iterations = iterations + 1
    return w, iterations

#Very rudimentary check on whether the Gradient Descent works
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = np.c_[np.ones(x.shape[0]), x]
    y = [5, 7, 11, 15, 18, 22, 29, 31, 35, 36]
    tol = 0.01
    alpha = 0.005
    w, iters = batchGradientDescent(x, y, tol, alpha)

    line = np.dot(x,w)

    plt.plot(x[:,1], line, 'g--', x[:,1], y, 'rs')
    plt.show()
print(iters)
