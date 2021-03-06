import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

	red_patch = mpatches.Patch(color='red', label='English')
	yellow_patch = mpatches.Patch(color='yellow', label='French')
	plt.legend(handles=[red_patch, yellow_patch], loc=4)

	plt.ylabel('Total number of letters')
	plt.xlabel('Number of A')
	plt.title('Gradient Descent, Batch')

	plt.show()

	#Test Gradient Descent algoirithm, Stochastic (online)
	alpha = 0.5

	w = [0,1]
	englishA, englishTot = shuffle(englishA, englishTot)
	for i in range(len(englishTot)):
		w = regression.stochasticGD(englishA[i], englishTot[i], w, alpha)
	line = w[1]*englishA+w[0]
	plt.plot(englishA, line, 'g--', englishA, englishTot, 'rs')

	print(w)

	w = [0,1]
	frenchA, frenchTot = shuffle(frenchA, frenchTot)
	for i in range(len(frenchTot)):
		w = regression.stochasticGD(frenchA[i], frenchTot[i], w, alpha)
	line = w[1]*frenchA+w[0]
	plt.plot(frenchA, line, 'b--', frenchA, frenchTot, 'ys')

	red_patch = mpatches.Patch(color='red', label='English')
	yellow_patch = mpatches.Patch(color='yellow', label='French')
	plt.legend(handles=[red_patch, yellow_patch], loc=4)

	plt.ylabel('Total number of letters')
	plt.xlabel('Number of A')
	plt.title('Stochastic Gradient Descent, online')

	plt.show()

def shuffle(x, y):
	p = np.random.permutation(len(y))
	return x[p], y[p]

main()
