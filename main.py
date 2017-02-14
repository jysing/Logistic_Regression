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

	tol = 10**-9
	alpha = 0.005
	stoch = False

	englishA = np.c_[np.ones(englishA.shape[0]), englishA]
	w, epochs = regression.linRegression(englishA, englishTot, tol, alpha, stoch)
	print(w, epochs)
	line = np.dot(englishA, w)
	print(englishA, englishTot)
	plt.plot(englishA[:,1], line, 'g--', englishA[:,1], englishTot, 'rs')

	frenchA = np.c_[np.ones(frenchA.shape[0]), frenchA]
	w, epochs = regression.linRegression(frenchA, frenchTot, tol, alpha, stoch)
	print(w, epochs)
	line = np.dot(frenchA, w)
	print(frenchA, frenchTot)
	plt.plot(frenchA[:,1], line, 'b--', frenchA[:,1], frenchTot, 'ys')
	plt.show()

main()