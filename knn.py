import pickle, sys, os
import tensorflow as tf
import numpy as np

from scipy.spatial import distance
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

from manifold_learn import *
from pollutionImage import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

protection = False
pollution = False

def dist(x, y):
	x = x.reshape(1,-1)
	y = y.reshape(1,-1)
	return np.sqrt(np.sum((x - y) ** 2))

def getMajorLabels(labels):
	# Partition to put the largest element in the rightmost index
	return np.argpartition(sum(labels), 9)[-1]

if __name__ == "__main__":
	if len(sys.argv) > 1:
		if "-i" in sys.argv or "--isomap" in sys.argv:
			print("Isomap Protection")
			protection = True

		if "-p" in sys.argv or "--pollution" in sys.argv:
			print("Polluted testing case")
			pollution = True

	# k = input("Please enter k: ")
	k = 1

	k = int(k)
	if k % 2 == 0:
		k = k + 1	# Make sure it is odd	
		print("WARNING: k should be odd. Justify your k from ", k - 1, " to ", k)

	if protection:
		im = manifold.Isomap(50, 100,n_jobs=-1)
		train_xs = im.fit_transform(mnist.train.images[:10000])
		print("finish training")
	else:
		train_xs = mnist.train.images

	test_num = 0.0
	test_err = 0.0

	# No need to train, go into test directly
	if not pollution:
		for _ in range(20):
			batch = mnist.test.next_batch(50)

			for i in range(len(batch[0])):
				if protection:
					test_x = im.transform(batch[0][i].reshape(1, -1))[0]
				else:
					test_x = batch[0][i]

				dist_arr = [dist(test_x, img) for img in train_xs]
				nearest_indices = np.argpartition(dist_arr, k)[:k]
				nearest_labels = np.array([mnist.train.labels[index] for index in nearest_indices])

				predicted_label = getMajorLabels(nearest_labels)
				actual_label = getMajorLabels([batch[1][i]])

				if predicted_label != actual_label:
					test_err += 1.0
				test_num += 1.0

	else:
		polluted_images = getPollutedImages()
		original_labels = getOrgLabel()

		for i in range(len(original_labels)):
			if protection:
				test_x = im.transform(polluted_images[i].reshape(1, -1))[0]
			else:
				test_x = polluted_images[i]

			dist_arr = [dist(test_x, img) for img in train_xs]
			nearest_indices = np.argpartition(dist_arr, k)[:k]
			nearest_labels = np.array([mnist.train.labels[index] for index in nearest_indices])

			predicted_label = getMajorLabels(nearest_labels)
			actual_label = getMajorLabels([original_labels[i]])

			if predicted_label != actual_label:
				test_err += 1.0
			test_num += 1.0

	print("test accuracy: ", 1.0 - test_err / test_num)
	