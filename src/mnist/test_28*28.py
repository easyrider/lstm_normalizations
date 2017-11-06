from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from normal_cells.lstm_bn_sep import BNLSTMCell
from normal_cells.lstm_cn_scale_input import CNSCALELSTMCell
from normal_cells.lstm_cn_sep import CNLSTMCell
from normal_cells.lstm_ln_sep import LNLSTMCell
from normal_cells.lstm_pcc_sep import PCCLSTMCell
from normal_cells.lstm_wn_sep import WNLSTMCell
from tensorflow.examples.tutorials.mnist import input_data

from src.normal_cells.lstm_basic import BASICLSTMCell

# Training Parameters
# learning_rate = 0.001
# training_steps = 100000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

FLAGS = None


# Import MNIST data


def run():
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	cell_dic = {
		'base': BASICLSTMCell,
		'bn_sep': BNLSTMCell,
		'cn_sep': CNLSTMCell,
		'cn_scale_sep': CNSCALELSTMCell,
		'ln_sep': LNLSTMCell,
		'wn_sep': WNLSTMCell,
		'pcc_sep': PCCLSTMCell
	}

	# tf Graph input
	X = tf.placeholder("float", [None, timesteps, num_input])
	Y = tf.placeholder("float", [None, num_classes])
	training = tf.placeholder(tf.bool)

	# Define weights
	with tf.name_scope('final_weights'):
		w = tf.get_variable(
			'W', [num_hidden, num_classes],
			initializer=tf.orthogonal_initializer())
	tf.summary.histogram("f_weight", w)

	with tf.name_scope('final_biases'):
		b = tf.get_variable('b', [num_classes])
	tf.summary.histogram("f_biases", b)

	def RNN(x, weights, biases):

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, timesteps, n_input)
		# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

		# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		x = tf.unstack(x, timesteps, 1)
		# x = tf.convert_to_tensor(x)
		if FLAGS.cell != 'bn_sep':
			# Define a lstm cell with tensorflow
			init_state = tf.contrib.rnn.LSTMStateTuple(
				tf.truncated_normal([batch_size, num_hidden], stddev=0.1),
				tf.truncated_normal([batch_size, num_hidden], stddev=0.1))
			lstm_cell = cell_dic[FLAGS.cell](num_hidden, forget_bias=1.0)

			# Get lstm cell output
			outputs, states = tf.nn.static_rnn(
				lstm_cell, x, initial_state=init_state, dtype=tf.float32)

			# Linear activation, using rnn inner loop last output
			return tf.matmul(outputs[-1], weights) + biases

		else:
			init_state = (tf.truncated_normal(
				[batch_size, num_hidden], stddev=0.1), tf.truncated_normal(
				[batch_size, num_hidden], stddev=0.1), tf.constant(
				0.0, shape=[1]))
			lstm_cell = BNLSTMCell(
				num_hidden,
				forget_bias=1.0,
				max_bn_steps=FLAGS.max_steps,
				is_training_tensor=training)
			outputs, states = tf.nn.static_rnn(
				lstm_cell, x, initial_state=init_state, dtype=tf.float32)
			_, final_hidden, _ = states

			return tf.matmul(final_hidden, weights) + biases

	logits = RNN(X, w, b)

	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
	gvs = optimizer.compute_gradients(loss_op)
	capped_gvs = [(None
	               if grad is None else tf.clip_by_value(grad, -1., 1.), var)
	              for grad, var in gvs]

	train_op = optimizer.apply_gradients(capped_gvs)

	# Evaluate model (with test logits, for dropout to be disabled)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	for (grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
		if grad is not None:
			tf.summary.histogram('grad/{}'.format(var.name), capped_grad)
			tf.summary.histogram('capped_fraction/{}'.format(var.name),
			                     tf.nn.zero_fraction(grad - capped_grad))
			tf.summary.histogram('variable/{}'.format(var.name), var)

	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar("xe_loss", loss_op)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

		# Run the initializer
		sess.run(init)

		for step in range(1, FLAGS.max_steps + 1):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Reshape data to get 28 seq of 28 elements
			batch_x = batch_x.reshape((batch_size, timesteps, num_input))
			# Run optimization op (backprop)
			sess.run(
				[train_op], feed_dict={X: batch_x,
				                       Y: batch_y,
				                       training: True})
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				summary, loss, acc, _ = sess.run(
					[merged, loss_op, accuracy, train_op],
					feed_dict={X: batch_x,
					           Y: batch_y,
					           training: True})
				print("Step " + str(step) + ", Minibatch Loss= " +
				      "{:.4f}".format(loss) + ", Training Accuracy= " +
				      "{:.3f}".format(acc))
				train_writer.add_summary(summary, step)

				valid_x, valid_y = mnist.validation.next_batch(batch_size)
				valid_x = valid_x.reshape((batch_size, timesteps, num_input))
				# Calculate batch loss and accuracy
				summary, loss, acc = sess.run(
					[merged, loss_op, accuracy],
					feed_dict={X: valid_x,
					           Y: valid_y,
					           training: False})
				print("Step " + str(step) + ", Minibatch Loss= " +
				      "{:.4f}".format(loss) + ", Valid Accuracy= " +
				      "{:.3f}".format(acc))
				valid_writer.add_summary(summary, step)

		print("Optimization Finished!")

		print("Validation Finished!")

		test_loss, test_acc = 0, 0
		for step in range(1, 1000):
			test_data, test_label = mnist.test.next_batch(batch_size)
			# Reshape data to get 28 seq of 28 elements
			test_data = test_data.reshape((batch_size, timesteps, num_input))
			# Calculate batch loss and accuracy
			summary, loss, acc = sess.run(
				[merged, loss_op, accuracy],
				feed_dict={X: test_data,
				           Y: test_label,
				           training: False})
			test_loss += loss
			test_acc += acc
			print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
				loss) + ", Testing Accuracy= " + "{:.3f}".format(acc))
			test_writer.add_summary(summary, step)

		print("Test Finished!")
		print("Test Ave_loss: " + str(test_loss / (1000)))
		print("Test Ave_acc: " + str(test_acc / (1000)))

		train_writer.close()
		test_writer.close()


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--max_steps',
		type=int,
		default=10000,
		help='Number of steps to run trainer.')
	parser.add_argument(
		'--lr', type=float, default=0.003, help='Learning rate')
	parser.add_argument(
		'--data_dir',
		type=str,
		default='../data/MNIST',
		help='Directory for storing input data')
	parser.add_argument(
		'--log_dir',
		type=str,
		default='/tmp/logs/mnist/base',
		help='Summaries log directory')
	parser.add_argument('--cell', type=str, default='bn_sep', help='RNN Cell')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
