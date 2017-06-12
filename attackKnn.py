import pickle
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def showImage(image):
	size = int(len(image) ** 0.5)
	reshaped_image = image.reshape(size, size)
	print(image)
	if(any(image)):
		plt.imshow(reshaped_image, cmap="gray")
		# plt.imshow(reshaped_image, cmap="gray", vmin=0.0, vmax=1.0)
		plt.show()

if __name__ == "__main__":
	if "-c" in sys.argv or "--clean" in sys.argv:
		target_knnt = KNeighborsClassifier(n_neighbors=1)
		target_knnt.train(mnist.train.images, mnist.train.labels)
		pickle.dump(target_knnt, open("knn_model", "rb"))
	else:
		target_knnt = pickle.load(open("knn_model", "rb"))

	print("Finish loading target knn...\n")

	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])

	x = tf.placeholder(tf.float32, shape=[None, 784])
	x_image = tf.reshape(x, [-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
 
	for i in range(20000):
		batch = mnist.train.next_batch(50)

		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


	print ("finish training cnn...")

	pollution_size = 100
	train_threshold = 2000
	
	optimizer = tf.train.GradientDescentOptimizer(0.1)
	class_gradient = optimizer.compute_gradients(y_conv, tf.trainable_variables())

	for i in range(pollution_size):
		image = mnist.test.images[i].reshape([1, 784])
		label = mnist.test.labels[i].reshape([1, 10])

		if np.argmax(label[0]) != 8:
			continue

		showImage(image[0])

		a = h_pool2.eval(feed_dict={x: image})[0]
		new_image = np.array([0] * (a.shape[0] * a.shape[1]))
		
		for k in range(a.shape[2]):
			f_map = np.array([0] * (a.shape[0] * a.shape[1]))
			
			for i in range(a.shape[0]):
				for j in range(a.shape[1]):
					f_map[i * a.shape[0] + j] = a[i][j][k]

			new_image = new_image + f_map

		showImage(new_image)

	# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	sess.close()
