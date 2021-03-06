import pickle, sys, os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

from manifold_learn import *
from pollutionImage import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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

	mfo = manifold_learn(784, 100, 3, 100, 100)
	im = mfo.readIsomap(FILENAME)
	print("Training IsoMap done...")

	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])

	x = tf.placeholder(tf.float32, shape=[None, mfo.getInputDimension()])
	reduced_x = tf.py_func(im.transform, [x], [tf.float64])[0]
	reduced_x = tf.cast(reduced_x, tf.float32)

	reduced_d = tf.cast(mfo.getIsomapDimension()**0.5, tf.int32)
	x_image = tf.reshape(reduced_x, [-1, reduced_d, reduced_d, 1])
	# x_image = tf.reshape(x, [-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	w_d = int((mfo.getIsomapDimension()**0.5 - 1) / 4) + 1
	W_fc1 = weight_variable([w_d * w_d * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, w_d * w_d * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	all_xs, all_ys = mfo.getXY()

	for i in range(mfo.getIterTime()):	
		batch_xs = all_xs[i * mfo.getBatchSize() : (i + 1) * mfo.getBatchSize()]
		batch_ys = all_ys[i * mfo.getBatchSize() : (i + 1) * mfo.getBatchSize()]

		# if i%100 == 0:
		# 	train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		# 	print("step %d, training accuracy %g"%(i, train_accuracy))
		
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	
	print("test accuracy %g"%accuracy.eval(feed_dict={x: getPollutedImages(), y_: getOrgLabel(), keep_prob: 1.0}))
	print(sess.run(y_conv, feed_dict={x: getPollutedImages()[:2], keep_prob: 1.0}))
	sess.close()

