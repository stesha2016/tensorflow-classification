import tensorflow as tf
from collections import namedtuple

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
slim = tf.contrib.slim

_CONV_DEFS = [
	Conv(kernel=[3, 3], stride=2, depth=32),
	DepthSepConv(kernel=[3, 3], stride=1, depth=64),
	DepthSepConv(kernel=[3, 3], stride=2, depth=128),
	DepthSepConv(kernel=[3, 3], stride=1, depth=128),
	DepthSepConv(kernel=[3, 3], stride=2, depth=256),
	DepthSepConv(kernel=[3, 3], stride=1, depth=256),
	DepthSepConv(kernel=[3, 3], stride=2, depth=512),
	DepthSepConv(kernel=[3, 3], stride=1, depth=512),
	DepthSepConv(kernel=[3, 3], stride=1, depth=512),
	DepthSepConv(kernel=[3, 3], stride=1, depth=512),
	DepthSepConv(kernel=[3, 3], stride=1, depth=512),
	DepthSepConv(kernel=[3, 3], stride=1, depth=512),
	DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
	DepthSepConv(kernel=[3, 3], stride=2, depth=1024)
]

def mobilenet_v1(inputs, num_classes=1000, is_training=True, depth_multiplier=1.0, reuse=None, scope='Mobilenet_v1'):
	with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse):
		with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
			x = mobilenet_v1_base(inputs, depth_multiplier=depth_multiplier, scope=scope)
			shape = x.get_shape().as_list()
			kernel_size = [min(shape[1], 7), min(shape[2], 7)]
			x = slim.avg_pool2d(x, kernel_size, padding='VALID', scope='AvgPool_1a')
			x = slim.dropout(x, keep_prob=0.999, scope='Dropout_1b')
			logits = slim.conv2d(x, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
			logits = tf.reshape(logits, [-1, num_classes])
			predictions = tf.nn.softmax(logits, name='predictions')
			return predictions, logits

def mobilenet_v1_base(inputs, depth_multiplier=1.0, scope='Mobilenet_v1'):
	assert(depth_multiplier > 0)
	depth = lambda d: max(int(d * depth_multiplier), 8)
	conv_defs = _CONV_DEFS
	with tf.variable_scope(scope, 'Mobilenetv1', [inputs]):
		with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
			x = inputs
			for i, conv_def in enumerate(conv_defs):
				scope_string_base = 'Conv2d_%d' % i
				if isinstance(conv_def, Conv):
					scope_string = scope_string_base
					x = slim.conv2d(x, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride, normalizer_fn=slim.batch_norm, scope=scope_string)
				elif isinstance(conv_def, DepthSepConv):
					scope_string = scope_string_base + '_detphwise'
					x = slim.separable_conv2d(x, None, conv_def.kernel, depth_multiplier=1,
						stride=conv_def.stride, normalizer_fn=slim.batch_norm, scope=scope_string)
					scope_string = scope_string_base + '_pointwise'
					x = slim.conv2d(x, depth(conv_def.depth), [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope=scope_string)
			return x

def mobilenet_v1_arg_scope(is_training=True, weight_decay=0.00004, stddev=0.09,
	regularize_depthwise=False, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
	batch_norm_params = {
		'center': True,
		'scale': True,
		'decay': batch_norm_decay,
		'epsilon': batch_norm_epsilon
	}
	if is_training is not None:
		batch_norm_params['is_training'] = is_training

	weights_init = tf.truncated_normal_initializer(stddev=stddev)
	regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	if regularize_depthwise:
		depthwise_regularizer = regularizer
	else:
		depthwise_regularizer = None
	with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
						weights_initializer=weights_init,
						activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
				with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
					return sc

def losses(labels, logits):
	tf.losses.softmax_cross_entropy(labels, logits)
	total_loss = tf.losses.get_total_loss(name='total_loss')
	return total_loss

def accurracy(labels, logits):
	prediction = tf.to_int64(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, tf.argmax(labels,1))
	accurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accurracy
