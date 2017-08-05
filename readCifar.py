import tarfile, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage


filename = "./Cifar data/cifar-10-python.tar.gz"
baseDir = "./Cifar data/"
dataDir = baseDir + "cifar-10-batches-py/"
train_batch = dataDir + "data_batch_"
test_batch = dataDir + "test_batch"

file = tarfile.open(filename)

# Extract if not extracted
if not os.path.exists(dataDir):
	file.extractall(baseDir)

file.close()

def _toByteKey(key):
	return bytes(key, "utf-8")

def _toListLabel(index):
	ret = [0.0] * 10
	ret[index] = 1.0
	return ret

def _getLabels(data_dict):
	return np.array([np.array(_toListLabel(label)) for label in data_dict.get(_toByteKey("labels"))])

def _flipImg(img):
	new_img = [0] * len(img)
	for i in range(int(len(img) / 32)):
		for j in range(32):
			new_img[i * 32 + j] = img[i * 32 + (31 - j)]

	return new_img

def _getData(data_dict):
	return data_dict.get(_toByteKey("data"))

def _getFlipData(data_dict):
	return list(map(lambda x: _flipImg(x), _getData(data_dict)))

def getTrainBatch(batch_id):
	# Check validity
	if batch_id < 1 or (batch_id != int(batch_id)):
		return dict()

	batch_id = batch_id % 5
	if batch_id == 0:
		batch_id = 5

	train_filename = train_batch + str(batch_id)
	with open(train_filename, "rb") as fo:
		data = pickle.load(fo, encoding="bytes")

	return data

def getAllPreProcessedTrainBatch():
	data, labels = getPreProcessedTrainBatch(1)

	for i in range(2, 6):
		batchData, batchLabels = getPreProcessedTrainBatch(i)
		data = np.append(data, batchData, axis=0)
		labels = np.append(labels, batchLabels, axis=0)

	return data, labels

# Only return preprocessed data and label
def getPreProcessedTrainBatch(batch_id):
	batch = getTrainBatch(batch_id)
	data = _getData(batch)
	flip_data = _getFlipData(batch)
	labels = _getLabels(batch)
	
	data = np.append(data, flip_data, axis=0)
	data = transformData(data)
	labels = np.append(labels, labels, axis=0)
	
	return data, labels

def getTestBatch():
	with open(test_batch, "rb") as fo:
		data = pickle.load(fo, encoding="bytes")

	return data

def getPreProcessedTestBatch():
	batch = getTestBatch()
	data = transformData(_getData(batch))
	labels = _getLabels(batch)

	return data, labels

def transformData(data, mean=0.0, stddev=1.0):
	means = [sum(d) / len(d) for d in data]
	stddevs = [(sum((d - (sum(d) / len(d))) ** 2) / (len(d) - 1)) ** 0.5 for d in data]
	return [(data[i] + mean - means[i]) * stddev / stddevs[i] for i in range(len(data))]

def displayImg(img):
	tmpimg = np.array(img).reshape(3,32,32).transpose(1,2,0)
	plt.imshow(tmpimg)
	plt.show()
