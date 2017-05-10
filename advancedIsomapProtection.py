import pickle
import sys
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

from manifold_learn import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def helper(X):
	result = im.transform(X)
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

if __name__ == "__main__":
	if len(sys.argv) == 2:
		if sys.argv[1] == "-c" or sys.argv[1] == "--clean":
			os.system("del xs ys isomap_model")

	mfo = manifold_learn(784, 10, 6, 100, 100)
	im = mfo.readIsomap(FILENAME)
	
	x = tf.placeholder(tf.float32, [None, mfo.getInputDimension()])
	reduced_x = tf.py_func(helper, [x], [tf.float64])[0]
	reduced_x = tf.cast(reduced_x, tf.float32)

	# W = tf.Variable(tf.zeros([mfo.getInputDimension(), 10]))
	W = tf.Variable(tf.zeros([mfo.getIsomapDimension(), 10]))

	b = tf.Variable(tf.zeros([10]))

	# y = tf.nn.softmax(tf.matmul(x, W) + b)
	y = tf.matmul(reduced_x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	all_xs, all_ys = mfo.getXY()

	for i in range(mfo.getIterTime()):
		batch_xs = all_xs[i * mfo.getBatchSize() : (i + 1) * mfo.getBatchSize()]
		batch_ys = all_ys[i * mfo.getBatchSize() : (i + 1) * mfo.getBatchSize()]
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
