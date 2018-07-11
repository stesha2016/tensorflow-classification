import src.utils as utils
import tensorflow as tf

def conv2d(x, f, kernel_size, strides=1, padding='same', use_batchnorm=True, name=None):
	x = tf.layers.conv2d(x, f, kernel_size=kernel_size, strides=strides, padding=padding, name=name, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
	x = tf.contrib.layers.batch_norm(x, decay=0.9997, epsilon=0.001)
	x = tf.nn.relu(x)
	return x

class Inception-v4():
	# 299x299x3 -> 35x35x384
	def stem(x, scope=None):
		with tf.variable_scope(scope, [x]):
			# 299x299x3 -> 149x149x32
			x = conv2d(x, 32, kernel_size=[3, 3], strides=2, padding='valid', name='Conv2d_1a_3x3')
			# 149x149x32 -> 147x147x32
			x = conv2d(x, 32, kernel_size=[3, 3], padding='valid', name='Conv2d_2a_3x3')
			# 147x147x32 -> 147x147x64
			x = cong2d(x, 64, kernel_size=[3, 3], name='Conv2d_2b_3x3')
			with tf.variable_scope('Mixed_3a'):
				with tf.variable_scope('Branch_0'):
					# 147x147x64 -> 73x73x64
					branch_0 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_0a_3x3')
				with tf.variable_scope('Branch_1'):
					# 147x147x64 -> 73x73x96
					branch_1 = conv2d(x, 96, [3, 3], strides=2, padding='valid', name='Conv2d_0a_3x3')
				# 73x73x64 + 73x73x96 -> 73x73x160
				x = tf.concat([branch_0, branch_1], axis=-1)
			with tf.variable_scope('Mixed_4a'):
				with tf.variable_scope('Branch_0'):
					# 73x73x160 -> 73x73x64
					branch_0 = conv2d(x, 64, [1, 1], name='Conv2d_0a_1x1')
					# 73x73x60 -> 71x71x96
					branch_0 = conv2d(branch_0, 96, [3, 3], padding='valid', name='Conv2d_1a_3x3')
				with tf.variable_scope('Branch_1'):
					# 73x73x160 -> 73x73x64
					branch_1 = conv2d(x, 64, [1, 1], name='Conv2d_0a_1x1')
					# 73x73x64 -> 73x73x64
					branch_1 = conv2d(branch_1, 64, [7, 1], name='Conv2d_0b_7x1')
					# 73x73x64 -> 73x73x64
					branch_1 = conv2d(branch_1, 64, [1, 7], name='Conv2d_0c_1x7')
					# 73x73x64 -> 71x71x96
					branch_1 = conv2d(branch_1, 96, [3, 3], padding='valid', name='Conv2d_1a_3x3')
				# 71x71x96 + 71x71x96 -> 71x71x192
				x = tf.concat([branch_0, branch_1], axis=-1)
			with tf.variable_scope('Mixed_5a'):
				with tf.variable_scope('Branch_0'):
					# 71x71x192 -> 35x35x192
					branch_0 = conv2d(x, 192, [3, 3], strides=2, padding='valid', name='Conv2d_1a_3x3')
				with tf.variable_scope('Branch_1'):
					# 71x71x192 -> 35x35x192
					branch_1 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_1a_3x3')
				# 35x35x192 + 35x35x192 -> 35x35x384
				x = tf.concat([branch_0, branch_1], axis=-1)
			return x

	def block_inception_a(x, scope=None, reuse=None):
		with tf.variable_scope(scope, 'BlockInceptionA', [x], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				branch_0 = tf.layers.average_pooling2d(x, [3, 3], name='AvgPool_0a_3x3')
				branch_0 = conv2d(branch_0, 96, [1, 1], name='Conv2d_0b_1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = conv2d(x, 96, [1, 1], name='Conv2d_0a_1x1')
			with tf.variable_scope('Branch_2'):
				branch_2 = conv2d(x, 64, [1, 1], name='Conv2d_0a_1x1')
				branch_2 = conv2d(branch_2, 96, [3, 3], name='Conv2d_0b_3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = conv2d(x, 64, [1, 1], name='Conv2d_0a_1x1')
				branch_3 = conv2d(branch_3, 96, [3, 3], name='Conv2d_0b_3x3')
				branch_3 = conv2d(branch_3, 96, [3, 3], name='Conv2d_0c_3x3')
			x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
			return x

	def block_reduction_a(x, scope=None, reuse=None):
		with tf.variable_scope(scope, 'BlockReductionA', [x], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				branch_0 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_1a_3x3')
			with tf.variable_scope('Branch_1'):
				branch_1 = conv2d(x, 384, [3, 3], strides=2, padding='valid', name='Conv2d_1a_3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = conv2d(x, 192, [1, 1], name='Conv2d_0a_1x1')
				branch_2 = conv2d(branch_2, 224, [3, 3], name='Conv2d_0b_3x3')
				branch_2 = conv2d(branch_2, 256, [3, 3], strides=2, padding='valid', name='Conv2d_1c_3x3')
			x = tf.concat([branch_0, branch_1, branch_2], axis=-1)
			return x

	def block_inception_b(x, scope=None, reuse=None):
		with tf.variable_scope(scope, 'BlockInceptionB', [x], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				branch_0 = tf.layers.average_pooling2d(x, [3, 3], name='MaxPool_0a_3x3')
				branch_0 = conv2d(branch_0, 128, [1, 1], name='Conv2d_0b, 1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = conv2d(x, 384, [1, 1], name='Conv2d_0a_1x1')
			with tf.variable_scope('Branch_2'):
				branch_2 = conv2d(x, 192, [1, 1], name='Conv2d_0a_1x1')
				branch_2 = conv2d(branch_2, 224, [1, 7], name='Conv2d_0b_1x7')
				branch_2 = conv2d(branch_2, 256, [7, 1], name='Conv2d_0c_7x1')
			with tf.variable_scope('Branch_3'):
				branch_3 = conv2d(x, 192, [1, 1], name='Conv2d_0a_1x1')
				branch_3 = conv2d(branch_3, 192, [1, 7], name='Conv2d_0b_1x7')
				branch_3 = conv2d(branch_3, 224, [7, 1], name='Conv2d_0c_7x1')
				branch_3 = conv2d(branch_3, 224, [1, 7], name='Conv2d_0d_1x7')
				branch_3 = conv2d(branch_3, 256, [7, 1], name='Conv2d_0e_7x1')
			x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
			return x

	def block_reduction_b(x, scope=None, reuse=None):
		with tf.variable_scope(scope, 'BlockReductionB', [x], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				branch_0 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_1a_3x3')
			with tf.variable_scope('Branch_1'):
				branch_1 = conv2d(x, 192, [1, 1], name='Conv2d_0a_1x1')
				branch_1 = conv2d(branch_1, 192, [3, 3], strides=2, padding='valid', name='Conv2d_1b_3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = conv2d(x, 256, [1, 1], name='Conv2d_0a_1x1')
				branch_2 = conv2d(branch_2, 256, [1, 7], name='Conv2d_0b_1x7')
				branch_2 = conv2d(branch_2, 320, [7, 1], name='Conv2d_0c_7x1')
				branch_2 = conv2d(branch_2, 320, [3, 3], strides=2, padding='valid', name='Conv2d_1d_3x3')
			x = tf.concat([branch_0, branch_1, branch_2], axis=-1)
			return x

	def block_inception_c(x, scope=None, reuse=None):
		with tf.variable_scope(scope, 'BlockInceptionC', [x], reuse=reuse):
			with tf.variable_scope('Branch_0'):
				branch_0 = tf.layers.average_pooling2d(x, [3, 3], name='MaxPool_0a_3x3')
				branch_0 = conv2d(branch_0, 256, [1, 1], name='Conv2d_0b_1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = conv2d(x, 256, [1, 1], name='Conv2d_0a_1x1')
			with tf.variable_scope('Branch_2'):
				branch_2 = conv2d(x, 384, [1, 1], name='Conv2d_0a_1x1')
				branch_2 = tf.concat([conv2d(branch_2, 256, [1, 3], name='Conv2d_0b_1x3'),
									  conv2d(branch_2, 256, [3, 1], name='Conv2d_0c_3x1')], axis=-1)
			with tf.variable_scope('Branch_3'):
				branch_3 = conv2d(x, 384, [1, 1], name='Conv2d_0a_1x1')
				branch_3 = conv2d(branch_3, 448, [1, 3], name='Conv2d_0b_1x3')
				branch_3 = conv2d(branch_3, 512, [3, 1], name='Conv2d_0c_3x1')
				branch_3 = tf.concat([conv2d(branch_3, 256, [3, 1], name='Conv2d_0d_3x1'),
					                  conv2d(branch_3, 256, [1, 3], name='Conv2d_0e_1x3')], axis=-1)
			x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)


	def inception_v4(x, num_classes=1001, isTraining=True, dropout_keep_prob=0.8, scope='InceptionV4'):
		with tf.variable_scope(scope, 'InceptionV4', regularizer=tf.contrib.layers.l2_regularizer(0.00004)) as scope:
			# 299x299x3 -> 35x35x384
			x = stem(x, scope)

			# 4 x Inception-A Blocks
			for i in range(4):
				block_scope = 'Mixed_5' + chr(ord('b') + i)
				x = block_inception_a(x, block_scope)

			# Reduction-A Block
			x = block_reduction_a(x, 'Mixed_6a')

			# 7 x Inception-B Blocks
			for i in range(7):
				block_scope = 'Mixed_6' + chr(ord('b') + i)
				x = block_inception_b(x, block_scope)

			# Reduction-B Block
			x = block_reduction_b(x, 'Mixed_7a')

			# 3 x Inception-C Blocks
			for i in range(3):
				block_scope = 'Mixed_7' + chr(ord('b') + i)
				x = block_inception_c(x, block_scope)

			with tf.variable_scope('Auxlogits'):
				x = tf.layers.average_pooling2d(x, [5, 5], strides=3, padding='valid', name='AvgPool_1a_5x5')
				x = conv2d(x, 128, [1, 1], name='Conv2d_1b_1x1')
				x = conv2d(x, 768, x.get_shape()[1:3], padding='valid', name='Conv2d_2a')
				x = tf.layers.flatten(x)
				x = tf.contrib.layers.fully_connected(x, num_classes, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm, scope='Aux_logits',
					normalizer_params={'decay': 0.9997, 'epsilon': 0.001}, weights_initializer=tf.contrib.layers.variance_scaling_initializer())

		with tf.variable_scope('Logits'):
			kernel_size = x.get_shape()[1:3]
			x = tf.layers.average_pooling2d(x, kernel_size, padding='valid', name='AvgPool_1a')
			x = tf.reduce_mean(x, [1, 2], keep_dims=True, name='global_pool')
			x = tf.layers.dropout(x, dropout_keep_prob, name='Dropout_1b')
			x = tf.layers.flatten(x)

			logits = tf.contrib.layers.fully_connected(x, num_classes, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm, scope='Logits',
					normalizer_params={'decay': 0.9997, 'epsilon': 0.001}, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
			predictions = tf.nn.softmax(logits, name='Predictions')
    	return logits, predictions
