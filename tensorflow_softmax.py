import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classification import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

allTrainX, allTrainY, X_test, testIDs = load_data()
allTrainX = np.delete(allTrainX, 519, axis=1) # deleting column 519
X_test = np.delete(X_test, 519, axis=1)# deleted column 519 

cutoff = int(allTrainX.shape[0] * 0.8)

X_train = allTrainX[0:cutoff, :]
Y_train = allTrainY[0:cutoff, :]
X_val = allTrainX[cutoff+1:, :]
Y_val = allTrainY[cutoff+1:, :]

num_features = X_train.shape[1]
num_labels = Y_train.shape[1]
learning_rate = 0.05
batch_size = 128
num_steps = 5001

graph = tf.Graph() # initialize a tensorflow graph

with graph.as_default():
	# Inputs
	tf_train_dataset = tf1.placeholder(tf.float32,
									shape=(batch_size, num_features))
	tf_train_labels = tf1.placeholder(tf.float32,
									shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(X_val)

	# Variables.
	weights = tf.Variable(
		tf.random.truncated_normal([num_features, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	# Training computation.
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=tf_train_labels, logits=logits))

	# Optimizer.
	optimizer = tf1.train.GradientDescentOptimizer(
		learning_rate).minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	tf_valid_dataset = tf.cast(tf_valid_dataset, tf.float32)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)

# utility function to calculate accuracy
def accuracy(predictions, labels):
	correctly_predicted = np.sum(
		np.argmax(predictions, 1) == np.argmax(labels, 1))
	acc = (100.0 * correctly_predicted) / predictions.shape[0]
	return acc

with tf1.Session(graph=graph) as session:
	# initialize weights and biases
	tf1.global_variables_initializer().run()
	print("Initialized")

	for step in range(num_steps):
		# pick a randomized offset
		offset = np.random.randint(0, Y_train.shape[0] - batch_size - 1)

		# Generate a minibatch.
		batch_data = X_train[offset:(offset + batch_size), :]
		batch_labels = Y_train[offset:(offset + batch_size), :]

		# Prepare the feed dict
		feed_dict = {tf_train_dataset: batch_data,
					tf_train_labels: batch_labels}

		# run one step of computation
		_, l, predictions = session.run([optimizer, loss, train_prediction],
										feed_dict=feed_dict)

		if (step % 500 == 0):
			print("Minibatch loss at step {0}: {1}".format(step, l))
			print("Minibatch accuracy: {:.1f}%".format(
				accuracy(predictions, batch_labels)))
			print("Validation accuracy: {:.1f}%".format(
				accuracy(valid_prediction.eval(), Y_val)))
			
feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

