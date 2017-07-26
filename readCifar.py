import tarfile, os, pickle

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
	if batch_id > 5 or batch_id < 1 or (batch_id != int(batch_id)):
		return dict()

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

def getLabels(data_dict):
	return data_dict.get(toByteKey("labels"))

def getData(data_dict):
	return data_dict.get(toByteKey("data"))
