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

if __name__ == "__main__":
	y, x = read_svm_file('data')
	print(y)
	print(x)