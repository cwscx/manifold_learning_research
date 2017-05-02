import pickle
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import manifold

from manifold_learn import *

FILENAME = "isomap_model"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def helper(X):
	return im.transform(X)

if __name__ == "__main__":
	mfo = manifold_learn(784, 20, 1, 100, 400)
	im = mfo.readIsomap(FILENAME)
		
	x = tf.placeholder(tf.float32, [None, mfo.getInputDimension()])
	reduced_x = tf.py_func(im.transform, [x], [tf.float64])[0]
	reduced_x = tf.cast(reduced_x, tf.float32)
	W = tf.Variable(tf.zeros([mfo.getIsomapDimension(), 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(reduced_x, W) + b)

	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

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
