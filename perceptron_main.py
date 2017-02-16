import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import perceptron

def main():
	y, x = perceptron.read_svm_file('data')
	x = perceptron.scale(x)
	w = [0, 0, 0];
	alpha = 0.005
	w, iters = perceptron.perc_alg(x, y, w, alpha)

	#Exclusively used for debugging thus far.
	yHat = perceptron.classify(x,w)
	print(yHat-y)

	x=np.asarray(x)
	# 0 = w0 + w1*x1 + w2*x2 => x2 = -w0 -(w1/w2)*x1
	line = -w[0]/w[2] - x[:,0]*w[1]/w[2]

	plt.plot(x[0:14,0], x[0:14,1], 'y.', x[15:29, 0], x[15:29, 1], 'r.', x[:,0], line, 'g-')
	
	red_patch = mpatches.Patch(color='red', label='English')
	yellow_patch = mpatches.Patch(color='yellow', label='French')
	plt.legend(handles=[red_patch, yellow_patch], loc=4)

	plt.ylabel('Total number of letters')
	plt.xlabel('Number of A')
	plt.title('Perceptron')

	plt.show()

	y, x = perceptron.read_svm_file('data')
	x = perceptron.scale(x)
	w = [0, 0, 0];
	alpha = 0.5
	w, iters = perceptron.perc_alg_reg(x, y, w, alpha)

	#Exclusively used for debugging thus far.
	yHat = perceptron.classify(x,w)
	print(yHat-y)

	x=np.asarray(x)
	# 0 = w0 + w1*x1 + w2*x2 => x2 = -w0 -(w1/w2)*x1
	line = -w[0]/w[2] - x[:,0]*w[1]/w[2]

	plt.plot(x[0:14,0], x[0:14,1], 'y.', x[15:29, 0], x[15:29, 1], 'r.', x[:,0], line, 'g-')
	
	red_patch = mpatches.Patch(color='red', label='English')
	yellow_patch = mpatches.Patch(color='yellow', label='French')
	plt.legend(handles=[red_patch, yellow_patch], loc=4)

	plt.ylabel('Total number of letters')
	plt.xlabel('Number of A')
	plt.title('Perceptron using Logistic Regression')

	plt.show()

main()