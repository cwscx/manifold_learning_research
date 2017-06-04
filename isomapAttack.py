import pickle
import tensorflow as tf
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

attack_knn = False

def normalization(arr, num):
	return (sum(map(lambda x: x**num, arr))) ** (1 / num)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		if "-k" in sys.argv or "--knn" in sys.argv:
			print("Start attacking knn...")
			attack_knn = True

	if attack_knn:
		knn_model = pickle.load(open("knn_model", "rb"))

	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	optimizer = tf.train.GradientDescentOptimizer(0.1)
	class_gradient = optimizer.compute_gradients(y, tf.trainable_variables())
	get_gradient = optimizer.compute_gradients(cross_entropy, tf.trainable_variables())
	train_step = optimizer.minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)

		if attack_knn:
			batch_ys = knn_model.predict(batch_xs)
		
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		if (i * 100 / 1000) % 5 == 0:
			print(i * 100 / 1000, "%...") 
	
	print("Finish black box training...")

	prediction = tf.argmax(y, 1)
	correct = tf.argmax(y_, 1)

	polluted_images = []
	original_labels = []
	files_to_write = []

	# pollution_size = len(mnist.test.images)
	pollution_size = 100
	train_threshold = 5000
	
	for i in range(pollution_size):
		
		image = mnist.test.images[i].reshape([1, 784])
		label = mnist.test.labels[i].reshape([1, 10])

		# Get the gradient for all the 784 dimensions
		predicted_label = sess.run(prediction, feed_dict={x : image})[0]
		correct_label = sess.run(correct, feed_dict={y_: label})[0]

		prob_of_correct_label = sess.run(y, feed_dict={x:image})[0][correct_label]

		j = 0
		while predicted_label == correct_label and j < train_threshold:
			j += 1
			if j == train_threshold:
				print("reach train threshold")

			minimum = sys.maxsize
			minimum_confidence_diff = None
			minimum_gradient_diff = None

			for k in range(10):
				if k == predicted_label:
					continue
				else:
					a = sess.run(class_gradient, feed_dict={x: image})

					correct_gradient = np.array([ ws[correct_label] for ws in sess.run(class_gradient, feed_dict={x: image})[0][1]])
					target_gradient = np.array([ ws[k] for ws in sess.run(class_gradient, feed_dict={x: image})[0][1]])
						
					delta_gradient = target_gradient - correct_gradient

					prob_of_k_label = sess.run(y, feed_dict={x: image})[0][k]
					confidence_diff = prob_of_k_label - prob_of_correct_label

					n = np.linalg.norm(delta_gradient)

					if n == 0:
						l = sys.maxsize
					else:
						l = abs(confidence_diff) / n
					
					if l < minimum:
						minimum_confidence_diff = confidence_diff
						minimum_gradient_diff = delta_gradient
				
			if minimum_gradient_diff is None or minimum_gradient_diff is None:
				continue
			else:
				r = abs(minimum_confidence_diff) / (normalization(minimum_gradient_diff, 2) ** 2) * minimum_gradient_diff
				image = image + r

				for j in range(len(image[0])):
					if image[0][j] < 0:
						image[0][j] = 0.0

			predicted_label = sess.run(prediction, feed_dict={x: image})[0]


		if (i * 100 / pollution_size) % 1 == 0:
			print((i * 100 / pollution_size), "%...") 

		info = {}
		info["original_image"] = mnist.test.images[i].reshape([1, 784])
		info["original_label"] = correct_label
		info["polluted_image"] = image.reshape([784])
		info["polluted_label"] = predicted_label

		polluted_images.append(image.reshape([784]))
		original_labels.append(mnist.test.labels[i])
		files_to_write.append(info)

	# end deep fool
	# out_file = input("outfile name: ")
	if attack_knn:
		out_file = "knn_polluted_images"
	else:
		out_file = "polluted_images"

	pickle.dump(files_to_write, open(out_file, "wb"))

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("test accuracy: ", sess.run(accuracy, feed_dict={x: polluted_images, y_: original_labels}))

	sess.close()
