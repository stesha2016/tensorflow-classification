import tensorflow as tf

def conv2d(x, f, kernel_size, strides=1, padding='SAME', name=None, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu):
	return tf.contrib.layers.conv2d(x, f, kernel_size=kernel_size, stride=strides, padding=padding.upper(), activation_fn=activation_fn,
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004), biases_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
        normalizer_fn=normalizer_fn, normalizer_params={'decay':0.9997, 'epsilon':0.001}, scope=name)

def fully_connected(x, num_classes):
	return tf.contrib.slim.fully_connected(x, num_classes, weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
		biases_regularizer=tf.contrib.layers.l2_regularizer(0.00004))

# 299x299x3 -> 35x35x384
def stem(x, scope=None):
	with tf.variable_scope(scope, [x]):
		# 299x299x3 -> 149x149x32
		x = conv2d(x, 32, kernel_size=[3, 3], strides=2, padding='valid', name='Conv2d_1a_3x3')
		# 149x149x32 -> 147x147x32
		x = conv2d(x, 32, kernel_size=[3, 3], padding='valid', name='Conv2d_2a_3x3')
		# 147x147x32 -> 147x147x64
		x = conv2d(x, 64, kernel_size=[3, 3], name='Conv2d_2b_3x3')
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

def block35(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	# 35x35x384
	with tf.variable_scope(scope, 'Block35', [x], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			# 35x35x384 -> 35x35x32
			branch_0 = conv2d(x, 32, kernel_size=[1, 1], name='Conv2d_0a_1x1')
		with tf.variable_scope('Branch_1'):
			# 35x35x384 -> 35x35x32
			branch_1 = conv2d(x, 32, kernel_size=[1, 1], name='Conv2d_0a_1x1')
			# 35x35x32 -> 35x35x32
			branch_1 = conv2d(branch_1, 32, kernel_size=[3, 3], name='Conv2d_0b_3x3')
		with tf.variable_scope('Branch_2'):
			# 35x35x384 -> 35x35x32
			branch_2 = conv2d(x, 32, kernel_size=[1, 1], name='Conv2d_0a_1x1')
			# 35x35x32 -> 35x35x48
			branch_2 = conv2d(branch_2, 48, kernel_size=[3, 3], name='Conv2d_0b_3x3')
			# 35x35x48 -> 35x35x64
			branch_2 = conv2d(branch_2, 64, kernel_size=[3, 3], name='Conv2d_0c_3x3')
		# 35x35x32 + 35x35x32 + 35x35x64 -> 35x35x128
		mixed = tf.concat([branch_0, branch_1, branch_2], axis=-1)
		# 35x35x128 -> 35x35x384
		branch = conv2d(mixed, 384, kernel_size=[1, 1], normalizer_fn=None, activation_fn=None, name='Conv2d_1x1')
		scaled_branch = branch * scale
		x += scaled_branch
		if activation_fn:
			x = activation_fn(x)
		return x

def block_reduction_a(x, scope=None, reuse=None):
	# 35x35x384
	with tf.variable_scope(scope, 'BlockReductionA', [x], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			# 35x35x384 -> 17x17x384
			branch_0 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_1a_3x3')
		with tf.variable_scope('Branch_1'):
			# 35x35x384 -> 17x17x384
			branch_1 = conv2d(x, 384, [3, 3], strides=2, padding='valid', name='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_2'):
			# 35x35x384 -> 35x35x256
			branch_2 = conv2d(x, 256, [1, 1], name='Conv2d_0a_1x1')
			# 35x35x256 -> 35x35x256
			branch_2 = conv2d(branch_2, 256, [3, 3], name='Conv2d_0b_3x3')
			# 35x35x256 -> 17x17x384
			branch_2 = conv2d(branch_2, 384, [3, 3], strides=2, padding='valid', name='Conv2d_1c_3x3')
		# 17x17x384 + 17x17x384 + 17x17x384 -> 17x17x1152
		x = tf.concat([branch_0, branch_1, branch_2], axis=-1)
		return x

def block17(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	# 17x17x1152
	with tf.variable_scope(scope, 'Block17', [x], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			# 17x17x1152 -> 17x17x192
			branch_0 = conv2d(x, 192, kernel_size=[1, 1], name='Conv2d_0a_1x1')
		with tf.variable_scope('Branch_1'):
			# 17x17x1152 -> 17x17x128
			branch_1 = conv2d(x, 128, kernel_size=[1, 1], name='Conv2d_0a_1x1')
			# 17x17x128 -> 17x17x160
			branch_1 = conv2d(branch_1, 160, kernel_size=[1, 7], name='Conv2d_0b_1x7')
			# 17x17x160 -> 17x17x192
			branch_1 = conv2d(branch_1, 192, kernel_size=[7, 1], name='Conv2d_0c_7x1')
		# 17x17x192 + 17x17x192 -> 17x17x384 
		mixed = tf.concat([branch_0, branch_1], axis=-1)
		# 17x17x384 -> 17x17x1152
		branch = conv2d(mixed, 1152, kernel_size=[1, 1], normalizer_fn=None, activation_fn=None, name='Conv2d_1x1')
		scaled_branch = branch * scale
		x += scaled_branch
		if activation_fn:
			x = activation_fn(x)
		return x

def block_reduction_b(x, scope=None, reuse=None):
	# 17x17x1152
	with tf.variable_scope(scope, 'BlockReductionB', [x], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			# 17x17x1152 -> 8x8x1154
			branch_0 = tf.layers.max_pooling2d(x, [3, 3], strides=2, padding='valid', name='MaxPool_1a_3x3')
		with tf.variable_scope('Branch_1'):
			# 17x17x1152 -> 17x17x256
			branch_1 = conv2d(x, 256, [1, 1], name='Conv2d_0a_1x1')
			# 17x17x256 -> 8x8x384
			branch_1 = conv2d(branch_1, 384, [3, 3], strides=2, padding='valid', name='Conv2d_1b_3x3')
		with tf.variable_scope('Branch_2'):
			# 17x17x1152 -> 17x17x256
			branch_2 = conv2d(x, 256, [1, 1], name='Conv2d_0a_1x1')
			# 17x17x256 -> 8x8x288
			branch_2 = conv2d(branch_2, 288, kernel_size=[3, 3], strides=2, padding='valid', name='Conv2d_1b_3x3')
		with tf.variable_scope('Branch_3'):
			# 17x17x1152 -> 17x17x256
			branch_3 = conv2d(x, 256, kernel_size=[1, 1], name='Conv2d_0a_1x1')
			# 17x17x256 -> 17x17x288
			branch_3 = conv2d(x, 288, kernel_size=[3, 3], name='Conv2d_0b_3x3')
			# 17x17x288 -> 8x8x320
			branch_3 = conv2d(x, 320, kernel_size=[3, 3], strides=2, padding='valid', name='Conv2d_1c_3x3')
		# 8x8x1152 + 8x8x384 + 8x8x288 + 8x8x320 -> 8x8x2142
		x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
		return x

def block8(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	# 8x8x2144
	with tf.variable_scope(scope, 'Block8', [x], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			# 8x8x2144 -> 8x8x192
			branch_0 = conv2d(x, 192, kernel_size=[1, 1], name='Conv2d_0a_1x1')
		with tf.variable_scope('Branch_1'):
			# 8x8x2144 -> 8x8x192
			branch_1 = conv2d(x, 192, kernel_size=[1, 1], name='Conv2d_0a_1x1')
			# 8x8x192 -> 8x8x224
			branch_1 = conv2d(branch_1, 224, kernel_size=[1, 3], name='Conv2d_0b_1x3')
			# 8x8x224 -> 8x8x256
			branch_1 = conv2d(branch_1, 256, kernel_size=[3, 1], name='Conv2d_0c_3x1')
		# 8x8x192 + 8x8x256 -> 8x8x448
		mixed = tf.concat([branch_0, branch_1], axis=-1)
		# 8x8x448 -> 8x8x2144
		branch = conv2d(mixed, 2144, kernel_size=[1, 1], normalizer_fn=None, activation_fn=None, name='Conv2d_1x1')
		scaled_branch = branch * scale
		x += scaled_branch
		if activation_fn:
			x = activation_fn(x)
		return x

def inception_resnet_v2(x, num_classes=1001, is_training=True, dropout_keep_prob=0.8, reuse=None, scope='InceptionResnetV2', activation_fn=tf.nn.relu):
	with tf.variable_scope(scope, 'InceptionResnetV2', [x], reuse=reuse):
		# 299x299x3 -> 35x35x384
		x = stem(x, scope)

		# 5 x block35
		# 35x35x384 -> 35x35x384
		for i in range(5):
			block_scope = 'Mixed_5' + chr(ord('b') + i)
			x = block35(x, scale=0.17, activation_fn=activation_fn, scope=block_scope)

		# 35x35x384 -> 17x17x1152
		x = block_reduction_a(x, 'Mixed_6a')

		# 7 x block17
		# 17x17x1152 -> 17x17x1152
		for i in range(7):
			block_scope = 'Mixed_6' + chr(ord('b') + i)
			x = block17(x, scale=0.10, activation_fn=activation_fn, scope=block_scope)

		# 17x17x1152 -> 8x8x2144
		x = block_reduction_b(x, 'Mixed_7a')

		# 5 x block8
		# 8x8x2144 -> 8x8x2144
		for i in range(5):
			block_scope = 'Mixed_7' + chr(ord('b') + i)
			x = block8(x, scale=0.20, activation_fn=activation_fn, scope=block_scope)

		with tf.variable_scope('Logits'):
			kernel_size = x.get_shape()[1:3]
			x = tf.layers.average_pooling2d(x, kernel_size, strides=1, padding='valid', name='AvgPool_1a')
			x = tf.layers.dropout(x, dropout_keep_prob, name='Dropout_1b')
			x = tf.layers.flatten(x, name='PreLogitsFlatten')

			logits = fully_connected(x, num_classes)
			predictions = tf.nn.softmax(logits, name='Predictions')
			return predictions, logits

def losses(labels, logits):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
	return loss

def accurracy(labels, logits):
	prediction = tf.to_int64(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, tf.argmax(labels,1))
	accurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accurracy
