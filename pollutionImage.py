import pickle
import numpy as np
import matplotlib.pyplot as plt

def _checkNegative(arr):
	for e in arr:
		if e < 0:
			return True

	return False

def getPollutions(filename="polluted_images"):
	return pickle.load(open(filename, "rb"))
	
def getPollutedImages(filename="polluted_images"):
	pollution = pickle.load(open(filename, "rb"))

	polluted_images = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			polluted_images.append(p["polluted_image"])

	return np.asarray(polluted_images)

def getOrgImages(filename="polluted_images"):
	pollution = pickle.load(open(filename, "rb"))

	original_images = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			original_images.append(p["original_image"])

	return np.asarray(original_images)

def getOrgLabel(filename="polluted_images"):
	pollution = pickle.load(open(filename, "rb"))

	original_labels = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			original_labels.append(p["original_label"])

	return np.asarray(original_labels)

def showImage(filename="polluted_images", size=1):
	p_images = getPollutedImages(filename)
	o_images = getOrgImages(filename)
	o_labels = getOrgLabel(filename)
	
	sum = 0
	for i in range(size):
		p_image = p_images[i].reshape(28,28)		
		o_image = o_images[i].reshape(28,28)

		diff = p_images[i] - o_images[i]
		sum += np.linalg.norm(diff)

		"""
		plt.imshow(o_image, cmap="gray", vmin=0, vmax=1.0)
		plt.show()
		plt.imshow(p_image, cmap="gray", vmin=0, vmax=1.0)
		plt.show()
		"""

	print(sum / 1000)

showImage("knn_polluted_images", 1000)