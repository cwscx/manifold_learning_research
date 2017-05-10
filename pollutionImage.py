import pickle

def getPolluteImage():
	return pickle.load(open("polluted_images", "rb"))
	
if __name__ == "__main__":
	b = getPolluteImage()
