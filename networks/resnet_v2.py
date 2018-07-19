import tensorflow as tf
import collections

slim = tf.contrib.slim

def resnet_arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
	batch_norm_scale=True, activation_fn=tf.nn.relu, use_batch_norm=True):
	batch_norm_params = {
		'decay': batch_norm_decay,
		'epsilon': batch_norm_epsilon,
		'scale': batch_norm_scale
	}
	with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=activation_fn,
		normalizer_fn=slim.batch_norm if use_batch_norm else None,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
	if stride == 1:
		return slim.conv2d(inputs, num_outputs, kernel_size, scope=scope)
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)

def subsample(inputs, stride, scope=None):
	if stride == 1:
		return inputs
	else:
		return slim.max_pool2d(inputs, [1, 1], stride=stride, scope=scope)

def bottleneck(inputs, depth, depth_bottleneck, stride, scope=None):
	with tf.variable_scope(scope, 'bottleneck_v2', [inputs]):
		preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

		depth_in = inputs.get_shape()[3]
		if depth == depth_in:
			shortcut = subsample(inputs, stride, scope='shortcut')
		else:
			shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
		
		residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
		residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
		residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

		output = shortcut + residual
		return output

def stack_block_dense(inputs, blocks):
	x = inputs
	for block in blocks:
		with tf.variable_scope(block.scope, 'block', [inputs]):
			for i, unit in enumerate(block.args):
				with tf.variable_scope('unit_%d' % (i + 1), [inputs]):
					x = block.unit_fn(x, **unit)
	return x

def resnet_v2(inputs, blocks, num_classes=None, is_training=True, reuse=None, scope=None):
	with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse):
		with slim.arg_scope(resnet_arg_scope()):
			with slim.arg_scope([slim.batch_norm], is_training=is_training):
				x = inputs
				with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
					x = conv2d_same(x, 64, 7, stride=2, scope='conv1')
				x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')
				# resnet v2 blocks start
				x = stack_block_dense(x, blocks)
				# end
				x = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')
				kernel_size = x.get_shape()[1:3]
				logits = slim.conv2d(x, num_classes, kernel_size, padding='VALID', activation_fn=None, normalizer_fn=None, scope='logits')
				logits = tf.reshape(logits, [-1, num_classes])
				predictions = slim.softmax(logits, scope='predictions')
				return predictions, logits

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
	'''
	# store tuples
	'''

def resnet_v2_block(scope, base_depth, num_units, stride):
	return Block(scope, bottleneck, [{
			'depth': base_depth * 4,
			'depth_bottleneck': base_depth,
			'stride': 1
		}] * (num_units - 1) + [{
			'depth': base_depth * 4,
			'depth_bottleneck': base_depth,
			'stride': stride
		}])

def resnet_v2_50(inputs, num_classes=None, is_training=True, reuse=None, scope='resnet_v2_50'):
	blocks = [
		resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
		resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
		resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
		resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
	]
	return resnet_v2(inputs, blocks, num_classes, is_training=is_training, reuse=reuse, scope=scope)

def resnet_v2_101(inputs, num_classes=None, is_training=True, reuse=None, scope='resnet_v2_101'):
	blocks = [
		resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
		resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
		resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
		resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
	]
	return resnet_v2(inputs, blocks, num_classes, is_training=is_training, reuse=reuse, scope=scope)

def resnet_v2_152(inputs, num_classes=None, is_training=True, reuse=None, scope='resnet_v2_152'):
	blocks = [
		resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
		resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
		resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
		resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
	]
	return resnet_v2(inputs, blocks, num_classes, is_training=is_training, reuse=reuse, scope=scope)

def resnet_v2_200(inputs, num_classes=None, is_training=True, reuse=None, scope='resnet_v2_200'):
	blocks = [
		resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
		resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
		resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
		resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
	]
	return resnet_v2(inputs, blocks, num_classes, is_training=is_training, reuse=reuse, scope=scope)

def losses(labels, logits):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
	return loss

def accurracy(labels, logits):
	print(logits)
	prediction = tf.to_int64(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, tf.argmax(labels,1))
	accurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accurracy
