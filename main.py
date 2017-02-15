import numpy as np
import matplotlib.pyplot as plt

import regression

def main():
	englishTot, englishA = np.loadtxt('english.txt', skiprows=0, unpack=True)
	frenchTot, frenchA = np.loadtxt('french.txt', skiprows=0, unpack=True)

	englishTot = englishTot/max(max(englishTot), max(frenchTot))
	englishA = englishA/max(max(englishA), max(frenchA))
	frenchTot = frenchTot/max(max(englishTot), max(frenchTot))
	frenchA = frenchA/max(max(englishA), max(frenchA))

	#Test Gradient Descent algoirithm, Batch
	tol = 10**-9
	alpha = 0.005

	w, epochs = regression.batchGD(englishA, englishTot, tol, alpha)
	line = w[1]*englishA+w[0]
	plt.plot(englishA, line, 'g--', englishA, englishTot, 'rs')

	print(w)

	w, epochs = regression.batchGD(frenchA, frenchTot, tol, alpha)
	line = w[1]*frenchA+w[0]
	plt.plot(frenchA, line, 'b--', frenchA, frenchTot, 'ys')

	plt.show()

	#Test Gradient Descent algoirithm, Stochastic (online)
	alpha = 0.5

	w = [0,1]
	for i in range(len(englishTot)):
		w = regression.stochasticGD(englishA[i], englishTot[i], w, alpha)
	line = w[1]*englishA+w[0]
	plt.plot(englishA, line, 'g--', englishA, englishTot, 'rs')

	print(w)

	w = [0,1]
	for i in range(len(frenchTot)):
		w = regression.stochasticGD(frenchA[i], frenchTot[i], w, alpha)
	line = w[1]*frenchA+w[0]
	plt.plot(frenchA, line, 'b--', frenchA, frenchTot, 'ys')

	plt.show()

main()