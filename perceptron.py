import numpy as np
import matplotlib.pyplot as plt
import math

# Read LIBSVM-format data.
def read_svm_file(data_file_name):
	y = []
	x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		label, features = line
		xi = []
		for e in features.split():
			ind, val = e.split(":")
			xi += [float(val)]
		y += [float(label)]
		x += [xi]
	return (y, x)

# Perceptron learning algorithm.
def perc_alg(x, y, w, alpha):
	iters = 1;
	while True:
		x, y = unison_shuffle(x,y)
		misClassified = 0
		for i in range(len(y)):
			currX = list(x[i]) #Copies x[i]
			currX.insert(0, 1)
			currY = y[i]
			if np.dot(currX, w) >= 0: h = 1
			else : h = -1
			if h == -1 and currY == 1:
				misClassified = misClassified + 1
				for j in range(len(currX)):
					w[j] = w[j] + alpha*currX[j]
			elif h == 1 and currY == -1:
				misClassified = misClassified + 1
				for j in range(len(currX)):
					w[j] = w[j] - alpha*currX[j]
		if misClassified == 0 : break
		else: iters = iters + 1
	return w, iters

def perc_alg_reg(x, y, w, alpha):
	iters = 1
	while True:
		x, y = unison_shuffle(x,y)
		for i in range(len(y)):
			currX = list(x[i]) #Copies x[i]
			currX.insert(0, 1)
			currY = y[i]
			if currY == -1 : currY = 0
			wx = np.dot(currX, w)
			h = 1 / (1 + math.exp(-wx))
			for j in range(len(currX)):
				w[j] = w[j] + alpha*(currY-h)*h*(1-h)*currX[j]
		if iters == 1000 : break
		else : iters = iters + 1
	return w, iters

def unison_shuffle(x, y):
    p = np.random.permutation(len(y))
    x = np.asarray(x)[p].tolist()
    y = np.asarray(y)[p].tolist()
    return x, y

def classify(x, w):
	arrX = np.asarray(x)
	z = w[0] + w[1]*arrX[:,0] + w[2]*arrX[:,1]
	return (z > 0)*2 - 1

def scale(x):
	arrX = np.asarray(x)
	for i in range(len(arrX[0,:])):
		arrX[:,i] = (arrX[:,i] / np.amax(arrX[:,i])).tolist()
	return arrX.tolist()

if __name__ == "__main__":
	y, x = read_svm_file('data')
	x = scale(x)
	w = [0, 0, 0];
	alpha = 0.5
	#w, iters = perc_alg(x, y, w, alpha)
	w, iters = perc_alg_reg(x, y, w, alpha)

	#Exclusively used for debugging thus far.
	yHat = classify(x,w)
	print(yHat-y)

	x=np.asarray(x)
	# 0 = w0 + w1*x1 + w2*x2 => x2 = -w0 -(w1/w2)*x1
	line = -w[0]/w[2] - x[:,0]*w[1]/w[2]

	plt.plot(x[0:14,0], x[0:14,1], 'r.', x[15:29, 0], x[15:29, 1], 'y.', x[:,0], line, 'g-')
	plt.show()
