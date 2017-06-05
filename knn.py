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
save = False
distance = False

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

		if "-s" in sys.argv or "--savemodel" in sys.argv:
			print("Save the knn model")
			save = True
			model_name = "knn_model"

	if protection:
		im = manifold.Isomap(10, 50, n_jobs=-1)
		
		train_xs = im.fit_transform(train_xs)
		print("finish training isomap")
	else:
		train_xs = mnist.train.images
		train_ls = mnist.train.labels

	test_num = 0.0
	test_err = 0.0

	knnt = KNeighborsClassifier(n_neighbors=11)
	knnt.fit(train_xs, train_ls)
	print("finish training knn model")

	if save:
		pickle.dump(knnt, open(model_name, "wb"))

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
		polluted_images = getPollutedImages("knn_polluted_images")
		original_labels = getOrgLabel("knn_polluted_images")

		# for i in range(len(original_labels)):
		for i in range(len(polluted_images)):
			
			if protection:
				test_x = im.transform(polluted_images[i].reshape(1, -1))[0].reshape(1, -1)
			else:
				test_x = polluted_images[i].reshape(1, -1)

			predicted_label = knnt.predict(test_x)[0]

			if any(predicted_label != original_labels[i]):
				test_err += 1.0
			test_num += 1.0

	print("test accuracy: ", 1.0 - test_err / test_num)