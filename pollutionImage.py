import pickle
import numpy as np
import scipy.misc

def _checkNegative(arr):
	for e in arr:
		if e < 0:
			return True

	return False

def getPollutions():
	return pickle.load(open("polluted_images", "rb"))
	
def getPollutedImages():
	pollution = pickle.load(open("polluted_images", "rb"))

	polluted_images = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			polluted_images.append(p["polluted_image"])

	return np.asarray(polluted_images)

def getOrgImages():
	pollution = pickle.load(open("polluted_images", "rb"))

	original_images = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			original_images.append(p["original_image"])

	return np.asarray(original_images)

def getOrgLabel():
	pollution = pickle.load(open("polluted_images", "rb"))

	original_labels = list()
	for p in pollution:
		if not _checkNegative(p["polluted_image"]):
			l = [0] * 10
			l[p["original_label"]] = 1
			original_labels.append(l)

	return np.asarray(original_labels)

def showImage(size=10):
	p_images = getPollutedImages()
	o_images = getOrgImages()
	o_labels = getOrgLabel()

	for i in range(size):
		p_image = p_images[i].reshape(28,28)		
		o_image = o_images[i].reshape(28,28)

		scipy.misc.toimage(p_image).show()
		scipy.misc.toimage(o_image).show()
		print(o_labels[i])

