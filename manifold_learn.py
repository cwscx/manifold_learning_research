import pickle
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

class manifold_learn():

	def __init__(self, input_dimension, isomap_dimension, knn, batch_size, iter_time):
		if isomap_dimension > batch_size:
			print("Error: BATCH_SIZE should not be less than ISOMAP_DIMENSION. \
				Becuase the reducted dimension is min(ISOMAP_DIMENSINO, BATCH_SIZE)")
			exit(1)

		self.input_dimension = input_dimension
		self.isomap_dimension = isomap_dimension
		self.knn = knn
		self.batch_size = batch_size 
		self.iter_time = iter_time 

	def getInputDimension(self):
		return self.input_dimension

	def getIsomapDimension(self):
		return self.isomap_dimension

	def getKNN(self):
		return self.knn

	def getBatchSize(self):
		return self.batch_size

	def getIterTime(self):
		return self.iter_time

	"""
	Save the xs and ys data for training the isomap model. Save the xs and ys locally.
	"""
	def saveXY(self, filename1="xs", filename2="ys"):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		all_xs, all_ys = mnist.train.next_batch(self.iter_time * self.batch_size)
		
		pickle.dump(all_xs, open(filename1, "wb"))
		pickle.dump(all_ys, open(filename2, "wb"))

		return all_xs, all_ys

	"""
	Restore the saved xs and ys from the local file. If it's not there, call saveXY
	"""
	def getXY(self, filename1="xs", filename2="ys"):
		if os.path.isfile(filename1) and os.path.isfile(filename2):
			return pickle.load(open(filename1, "rb")), pickle.load(open(filename2, "rb"))
		else:
			return self.saveXY(filename1, filename2)

	"""
	Train the isomap model and save the model locally for future access (you don't know how long it takes to generate)
	"""
	def trainAndSaveIsomap(self, filename="isomap_model"):
		all_xs, all_ys = self.getXY("xs", "ys")

		im = manifold.Isomap(self.knn, self.isomap_dimension)
		im = im.fit(all_xs)
		pickle.dump(im, open(filename, "wb"))

		return im

	"""
	Restore the isomap model from the stored local file
	"""
	def readIsomap(self, filename="isomap_model"):
		if os.path.isfile(filename):
			return pickle.load(open(filename, "rb"))
		else:
			return self.trainAndSaveIsomap(filename)
