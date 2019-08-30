#python USC ID.py K D N PATH TO DATA DIR
#python 4916525888.py 5 12 20 ./mnist
import sys
import numpy as np
from sklearn.decomposition import PCA

K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])
path = sys.argv[4]

image_data = np.reshape(np.fromfile('train-images-idx3-ubyte', dtype=np.ubyte, count=784000, sep='', offset=16), (1000, 784))
label_data = np.fromfile('train-labels-idx1-ubyte', dtype=np.ubyte, count=1000, sep='', offset=8)
tr_data = image_data[N:, :]
va_data = image_data[:N, :]
tr_labels = label_data[N:]
va_labels = label_data[:N]

pca = PCA(n_components=D, svd_solver='full')
pca.fit(tr_data)
tr_data = pca.transform(tr_data)
va_data = pca.transform(va_data)

def calcDistance(data1, data2):
	distance = 0
	for x in range(D):
		distance += np.square(data1[x] - data2[x])
	return np.sqrt(distance)

def knn(data):
	#get all distances
	distances = []
	d = {} #distance, index
	sort = {}
	for x in range(len(tr_data)):
		distances.append(calcDistance(data, tr_data[x]))
		d[distances[x]] = x
	distances.sort()

	#find k nearest neighbors
	neighbors = []
	for x in range(K):
		neighbors.append(distances[x])

	#count votes
	digits = [0,0,0,0,0,0,0,0,0,0]
	for x in neighbors:
		index = d.get(x)
		prediction = tr_labels[index]
		digits[prediction] += 1

	return digits.index(max(digits))

for i in range(N):
	print(str(knn(va_data[i])) + " " + str(va_labels[i]))
