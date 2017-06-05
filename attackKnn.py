import pickle
import tensorflow as tf
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
	if "-c" in sys.argv or "--clean" in sys.argv:
		target_knnt = KNeighborsClassifier(n_neighbors=1)
		target_knnt.train(mnist.train.images, mnist.train.labels)
		pickle.dump(target_knnt, open("knn_model", "rb"))
	else:
		target_knnt = pickle.load(open("knn_model", "rb"))
	
	print("Finish loading target knn...\n")

	model_knnt = KNeighborsClassifier(n_neighbors=5)

	test_xs = mnist.train.images[:1000]
	test_ls = target_knnt.predict(test_xs)

	print("Finish prediction...\n")

	model_knnt.fit(test_xs, test_ls)
	print("Finish mimicing knn model...\n")

	view_neighbor_num = int(len(test_xs) / len(test_ls[0])) + 1
	view_neighbor_num = len(test_xs)
	update_c = 0.01

	polluted_images = []
	org_labels = []
	files_to_write = []

	for i in range(len(test_xs)):
		image = test_xs[i]
		label = test_ls[i]

		ret = model_knnt.kneighbors([image], view_neighbor_num, True)
		indices = ret[1][0]
		
		target_label = None
		for j in indices:
			if np.argmax(test_ls[j]) != np.argmax(label):
				target_label = test_ls[j]
				target_image = test_xs[j]
				break

		if target_label is None:
			continue

		while np.argmax(target_label) != np.argmax(target_knnt.predict([image])[0]):
			diff = target_image - image

			image = image + diff * update_c
			
			for j in range(len(image)):
				if image[j] < 0:
					image[j] = 0.0
				elif image[j] > 1:
					image[j] = 1.0

		info = {}
		info["original_image"] = test_xs[i]
		info["original_label"] = test_ls[i]
		info["polluted_image"] = image.reshape([784])
		info["polluted_label"] = target_knnt.predict([image])[0]

		print(info["original_label"])

		files_to_write.append(info)
		polluted_images.append(image)
		org_labels.append(label)

		if (i * 100 / len(test_xs)) % 1 == 0:
			print((i * 100 / len(test_xs)), "%...")
		
	pickle.dump(files_to_write, open("knn_polluted_images", "wb"))

	# print(target_knnt.score(polluted_images, org_labels))