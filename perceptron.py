import numpy as np

# Read LIBSVM-format data.
def read_svm_file(data_file_name):
	y = []
	x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		y += [float(label)]
		x += [xi]
	return (y, x)

# Perceptron learning algorithm.
def perc_alg(x, y, w, alpha):
	if np.dot(x, w) >= 0: 
		hypothesis = 1
	else :
		hypothesis = -1

if __name__ == "__main__":
	y, x = read_svm_file('data')