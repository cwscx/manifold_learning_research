import pickle, sys, os
import tensorflow as tf
import numpy as np

from readCifar import *

cifar_train_batch_1 = getTrainBatchData(1)
cifar_test_batch = getTestBatchData()

test_image = getData(cifar_test_batch)
test_label = getLabels(cifar_test_batch)

def weight_variable(shape, stddev):
	initial = tf.truncated_normal(shape, stddev=stddev)
	return tf.Variable(initial)

def bias_variable(shape, constant):
	initial = tf.constant(constant, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

if __name__ == "__main__":

	W_conv1 = weight_variable([5,5,3,64], 5e-2)
	b_conv1 = bias_variable([64], 0.0)

	x = tf.placeholder(tf.float32, shape=[100, 3072])
	x_image = tf.reshape(x, [-1,32,32,3])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

	W_conv2 = weight_variable([5,5,64,64], 5e-2)
	b_conv2 = bias_variable([64], 0.1)

	h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)
	h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
	h_pool2 = max_pool_2x2(h_norm2)
	reshape = tf.reshape(h_pool2, [100, -1])

	weight_dim = reshape.get_shape()[1].value

	W_fc1 = weight_variable([weight_dim, 384], 4e-2)
	b_fc1 = bias_variable([384], 0.1)
	
	h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

	W_fc2 = weight_variable([384, 192], 4e-2)
	b_fc2 = bias_variable([192], 0.1)

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

	W_fc3 = weight_variable([192, 10], 1/192)
	b_fc3 = bias_variable([10], 0.0)

	y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
	y_ = tf.placeholder(tf.float32, [None, 10])

	print(y_conv.shape)

	learning_rate = 1e-4

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	class_gradient = optimizer.compute_gradients(y_conv, tf.trainable_variables())

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for j in range(1, 201):
		cifar_train_batch = getTrainBatchData(j)
		
		if (j / 5) % 2 == 0:	
			train_images = getData(cifar_train_batch)
		else:
			train_images = getFlipData(cifar_train_batch)

		train_labels = getLabels(cifar_train_batch)

		for i in range(100):
			batch_images = train_images[i * 100 : (i + 1) * 100]
			batch_labels = train_labels[i * 100 : (i + 1) * 100]

			train_accuracy = accuracy.eval(feed_dict={x:batch_images, y_: batch_labels})
			print("step %d, training accuracy %g" %((j - 1) * 100 + i, train_accuracy))

			train_step.run(feed_dict={x: batch_images, y_: batch_labels})

		if j % 10 == 0:
			acc = 0.0

			for i in range(100):
				acc += accuracy.eval(feed_dict={x: test_image[i * 100 : (i + 1) * 100], y_: test_label[i * 100 : (i + 1) * 100]})

			acc /= 100.0
			print("test accuracy %g"% acc)

	"""
	saver = tf.train.Saver()
	model_dir = "./tmp/cnnModel.ckpt"

	if os.path.exists(model_dir + ".index"):
		saver.restore(sess, model_dir)
		print("Model restored")
	else:
		
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if i%100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
				print("step %d, training accuracy %g"%(i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
		save_path = saver.save(sess, "./tmp/cnnModel.ckpt")
		print("Model saved in file: %s" % save_path)
	"""

	# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


	sess.close()
	