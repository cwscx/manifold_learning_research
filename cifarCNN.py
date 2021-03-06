import pickle, sys, os, random
import tensorflow as tf
import numpy as np

from readCifar import *

processed_train_data = "cifar_processed_train_batch"

LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

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
	x_image = tf.reshape(x, [-1,3,32,32])
	t_x_image = tf.transpose(x_image, (0,2,3,1))

	h_conv1 = conv2d(t_x_image, W_conv1)
	pre_activation1 = tf.nn.bias_add(h_conv1, b_conv1)
	conv1 = tf.nn.relu(pre_activation1)
	h_pool1 = max_pool_2x2(conv1)
	h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	W_conv2 = weight_variable([5,5,64,64], 5e-2)
	b_conv2 = bias_variable([64], 0.1)

	h_conv2 = conv2d(h_norm1, W_conv2)
	pre_activation2 = tf.nn.bias_add(h_conv2, b_conv2)
	conv2 = tf.nn.relu(pre_activation2)
	h_norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
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
	b_fc3 = bias_variable([10], 0.1)

	y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
	y_ = tf.placeholder(tf.float32, [None, 10])

	global_step = tf.contrib.framework.get_or_create_global_step()
	decay_steps = 100000
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
			                                  global_step,
			                                  decay_steps,
			                                  LEARNING_RATE_DECAY_FACTOR,
			                                  staircase=True)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	opt = tf.train.GradientDescentOptimizer(learning_rate)
	train_step = opt.minimize(cross_entropy_mean)

	class_gradient = opt.compute_gradients(y_conv, tf.trainable_variables())

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	if not os.path.exists(processed_train_data):
		train_images, train_labels = getAllPreProcessedTrainBatch()
		pickle.dump((train_images, train_labels), open(processed_train_data, "wb"))
	else:
		train_images, train_labels = pickle.load(open(processed_train_data, "rb"))

	test_image, test_label = getPreProcessedTestBatch()

	print("Finish pre process all data")
	
	indices = list(range(len(train_labels)))

	for j in range(1, 10001):
		random.seed(j)
		random_indices = random.sample(indices, 100)

		batch_images = np.array([train_images[random_indices[0]]])
		batch_labels = np.array([train_labels[random_indices[0]]])

		for i in range(1, 100):
			batch_images = np.append(batch_images, np.array([train_images[random_indices[i]]]), axis=0)
			batch_labels = np.append(batch_labels, np.array([train_labels[random_indices[i]]]), axis=0)

		train_accuracy = accuracy.eval(feed_dict={x:batch_images, y_: batch_labels})
		print("step %d, training accuracy %g" %(j, train_accuracy))

		train_step.run(feed_dict={x: batch_images, y_: batch_labels})

		if j % 100 == 0:
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
	