filename = "./cifar/test_batch"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getByteKey(key):
	return bytes(key, "utf-8")

"""
print(unpickle(filename).keys())
a = unpickle(filename)[getByteKey("data")][0]
"""

print(len(unpickle("./cifar/test_batch")[getByteKey("data")]))