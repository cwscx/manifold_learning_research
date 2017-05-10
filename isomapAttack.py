import pickle
import tensorflow as tf
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def normalization(arr, num):
	return (sum(map(lambda x: x**num, arr))) ** (1 / num)

if __name__ == "__main__":
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	optimizer = tf.train.GradientDescentOptimizer(0.1)
	get_gradient = optimizer.compute_gradients(cross_entropy, tf.trainable_variables())
	train_step = optimizer.minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		
		"""
		gradients_and_vars = sess.run(get_gradient, feed_dict={x: batch_xs, y_: batch_ys})

		for g, v in gradients_and_vars:
			if g is not None:
				print("****************this is variable*************")
				print("variable's shape:", v.shape)
				print(v)
				print("****************this is gradient*************")
				print("gradient's shape:", g.shape)
				print(g)
				for i in range(10):
					correct_gradient = np.array([ ws[i] for ws in g])
					print(normalization(correct_gradient, 2))
		"""

	prediction = tf.argmax(y, 1)
	correct = tf.argmax(y_, 1)

	polluted_images = []
	original_labels = []
	files_to_write = []

	for i in range(500):
	# for i in range(len(mnist.test._images)):
		image = mnist.test._images[i].reshape([1, 784])
		label = mnist.test._labels[i].reshape([1, 10])

		predicted_label = sess.run(prediction, feed_dict={x : image})[0]
		correct_label = sess.run(correct, feed_dict={y_: label})[0]

		prob_of_correct_label = sess.run(y, feed_dict={x:image})[0][correct_label]

		j = 0
		while predicted_label == correct_label and j < 2000:
			j += 1
			minimum = sys.maxsize
			minimum_confidence_diff = None
			minimum_gradient_diff = None

			for k in range(10):
				if k == predicted_label:
					continue
				else:
					target_label = [l for l in label]
					target_label[0][predicted_label] = 0
					target_label[0][k] = 1

					correct_gradient = np.array([ ws[correct_label] for ws in sess.run(get_gradient, feed_dict={x: image, y_: label})[0][0]])
					target_gradient = np.array([ ws[k] for ws in sess.run(get_gradient, feed_dict={x: image, y_: target_label})[0][0]])
					
					delta_gradient = target_gradient - correct_gradient

					prob_of_k_label = sess.run(y, feed_dict={x: image})[0][k]
					confidence_diff = prob_of_k_label - prob_of_correct_label

					l = abs(confidence_diff) / normalization(delta_gradient, 2)
					if l < minimum:
						minimum_confidence_diff = confidence_diff
						minimum_gradient_diff = delta_gradient
			
			if minimum_gradient_diff is None or minimum_gradient_diff is None:
				continue
			else:
				r = abs(minimum_confidence_diff) / (normalization(minimum_gradient_diff, 2) ** 2) * minimum_gradient_diff
				image = image + r

			predicted_label = sess.run(prediction, feed_dict={x: image})[0]

		if (i / 500 * 100) % 1 == 0:
			print((i / 500 * 100), "%...") 

		info = {}
		info["original_image"] = mnist.test._images[i].reshape([1, 784])
		info["original_label"] = correct_label
		info["polluted_image"] = image.reshape([784])
		info["polluted_label"] = predicted_label

		polluted_images.append(image.reshape([784]))
		original_labels.append(mnist.test._labels[i])
		files_to_write.append(info)

	# end deep fool
	pickle.dump(files_to_write, open("polluted_images", "wb"))

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: polluted_images, y_: original_labels}))

	sess.close()

	