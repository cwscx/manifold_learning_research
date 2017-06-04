import pickle, sys, os
import tensorflow as tf
import numpy as np

from scipy.spatial import distance
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier

from manifold_learn import *
from pollutionImage import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

protection = False
pollution = False

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

	if protection:
		im = manifold.Isomap(10, 50, n_jobs=-1)
		train_xs = list()
		train_ls = list()

		for i in range(len(mnist.train.images)):
			if mnist.train.labels[i][1] == 1 or mnist.train.labels[i][7] == 1:
				train_xs.append(mnist.train.images[i])
				train_ls.append(mnist.train.labels[i])

			if len(train_xs) > 50000:
				break

		train_xs = im.fit_transform(train_xs)
		print("finish training")
	else:
		train_xs = mnist.train.images
		train_ls = mnist.train.labels

	test_num = 0.0
	test_err = 0.0

	knnt = KNeighborsClassifier(n_neighbors=1)
	knnt.fit(train_xs, train_ls)
	print("finish training knn model")

	# No need to train, go into test directly
	if not pollution:
		for _ in range(20):
			batch = mnist.test.next_batch(50)

			for i in range(len(batch[0])):
				if protection:
					test_x = im.transform(batch[0][i].reshape(1, -1))[0].reshape(1, -1)
				else:
					test_x = batch[0][i].reshape(1, -1)

				predicted_label = knnt.predict(test_x)[0]

				if any(predicted_label != batch[1][i]):
					test_err += 1.0
				test_num += 1.0
	else:
		polluted_images = getPollutedImages()
		original_labels = getOrgLabel()

		# for i in range(len(original_labels)):
		for i in range(len(polluted_images)):
			if getMajorLabels([original_labels[i]]) == 1 or getMajorLabels([original_labels[i]]) == 7:

				if protection:
					test_x = im.transform(polluted_images[i].reshape(1, -1))[0].reshape(1, -1)
				else:
					test_x = polluted_images[i].reshape(1, -1)

				predicted_label = knnt.predict(test_x)[0]

				if any(predicted_label != original_labels[i]):
					test_err += 1.0
				test_num += 1.0

	print("test accuracy: ", 1.0 - test_err / test_num)
