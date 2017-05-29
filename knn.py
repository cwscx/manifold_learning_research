import pickle, sys, os
import tensorflow as tf
import numpy as np

from scipy.spatial import distance
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

from manifold_learn import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def dist(x, y):
	x = x.reshape(1,-1)
	y = y.reshape(1,-1)
	return np.sqrt(np.sum((x - y) ** 2))

def getMajorLabels(labels):
	# Partition to put the largest element in the rightmost index
	return np.argpartition(sum(labels), 9)[-1]

if __name__ == "__main__":
	
	k = input("Please enter k: ")
	
	k = int(k)
	if k % 2 == 0:
		k = k + 1	# Make sure it is odd	
		print("WARNING: k should be odd. Justify your k from ", k - 1, " to ", k)

	test_num = 0.0
	test_err = 0.0

	# No need to train, go into test directly
	for _ in range(1):
		batch = mnist.test.next_batch(50)

		for i in range(len(batch[0])):
			dist_arr = np.array([dist(batch[0][i], img) for img in mnist.train.images])
			nearest_indices = np.argpartition(dist_arr, k)[:k]
			nearest_labels = np.array([mnist.train.labels[index] for index in nearest_indices])

			predicted_label = getMajorLabels(nearest_labels)
			actual_label = getMajorLabels([batch[1][i]])

			if predicted_label != actual_label:
				test_err += 1.0
			test_num += 1.0

	print(1.0 - test_err / test_num)