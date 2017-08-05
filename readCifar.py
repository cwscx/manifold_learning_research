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

def getAllTrainBatchData():
	data = list()
	for i in range(1, 6):
		data.append(getTrainBatchData(i))

	return data

def getTrainBatchData(batch_id):
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

def getTestBatchData():
	with open(test_batch, "rb") as fo:
		data = pickle.load(fo, encoding="bytes")

	return data

def toByteKey(key):
	return bytes(key, "utf-8")

def toListLabel(index):
	ret = [0.0] * 10
	ret[index] = 1.0
	return ret

def getLabels(data_dict):
	return np.array([np.array(toListLabel(label)) for label in data_dict.get(toByteKey("labels"))])

def transformData(data, mean=0.0, stddev=1.0):
	means = [sum(d) / len(d) for d in data]
	stddevs = [(sum((d - (sum(d) / len(d))) ** 2) / (len(d) - 1)) ** 0.5 for d in data]
	return [(data[i] + mean - means[i]) * stddev / stddevs[i] for i in range(len(data))]

def getData(data_dict):
	ret = data_dict.get(toByteKey("data"))
	return transformData(ret)

def getFlipData(data_dict):
	return list(map(lambda x: flipImg(x), getData(data_dict)))

def flipImg(img):
	new_img = [0] * len(img)
	for i in range(int(len(img) / 32)):
		for j in range(32):
			new_img[i * 32 + j] = img[i * 32 + (31 - j)]

	return new_img

def displayImg(img):
	tmpimg = np.array(img).reshape(3,32,32).transpose(1,2,0)
	plt.imshow(tmpimg)
	plt.show()
