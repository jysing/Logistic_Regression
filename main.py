import numpy as np

def main():
	englishTot, englishA = np.loadtxt('english.txt', skiprows=0, unpack=True)
	englishTot = englishTot/max(englishTot)
	englishA = englishA/max(englishA)

	frenchTot, frenchA = np.loadtxt('french.txt', skiprows=0, unpack=True)
	frenchTot = frenchTot/max(frenchTot)
	frenchA = frenchA/max(frenchA)

	print(frenchTot)
	print(englishTot)

main()