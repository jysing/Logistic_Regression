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
	#while
	#for
	x = np.c_[np.ones(x.shape[0]), x]
	if np.dot(x, w) >= 0: h = 1
	else : h = -1

	if y == 1 and h == -1 :
		#todo
		w = w
	elif y == -1 and h == 1 :
		#todo
		w = w


if __name__ == "__main__":
	y, x = read_svm_file('data')
	print(y)
	print(x)