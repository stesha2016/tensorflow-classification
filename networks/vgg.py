import tensorflow as tf
import numpy as np
# VGG16 output class number 1000 -> weights: 137,557,696

def max_pool(x, name):
	return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same', name=name)

def dropout(x, keepPro, name=None):
	return tf.nn.dropout(x, keepPro, name)

def conv_layer(x, f, size, name):
	with tf.variable_scope(name):
		# contain kernal and bias
		conv = tf.layers.conv2d(x, f, kernel_size=size, padding='same', name=name)
		relu = tf.nn.relu(conv)
		return relu

def fc_layer(x, inputD, outputD, name, relu=False):
	with tf.variable_scope(name):
		w = tf.get_variable('w', shape=[inputD, outputD], dtype='float32')
		b = tf.get_variable('b', shape=[outputD], dtype='float32')
		out = tf.nn.xw_plus_b(x, w, b, name=name)
		if relu:
			out = tf.nn.relu(out)
		return out

class Vgg():
	def __init__(self, im, class_num, isvgg19, modelpath='./models/vgg16.npy'):
		self.input_x = im
		self.class_num = class_num
		self.vgg19 = isvgg19
		self.modelpath = modelpath

	def build(self, isTraining=True):
		# 224x224x3 -> 224x224x64 -> 224x224x64 -> 112x112x64
		conv1_1 = conv_layer(self.input_x, 64, 3, name='conv1_1')
		conv1_2 = conv_layer(conv1_1, 64, 3, name='conv1_2')
		pool1 = max_pool(conv1_2, name='pool1')

		# 112x112x64 -> 112x112x128 -> 112x112x128 -> 56x56x128
		conv2_1 = conv_layer(pool1, 128, 3, name='conv2_1')
		conv2_2 = conv_layer(conv2_1, 128, 3, name='conv2_2')
		pool2 = max_pool(conv2_2, name='pool2')

		# 56x56x128 -> 56x56x256 -> 56x56x256 -> 56x56x256 -> 28x28x256
		conv3_1 = conv_layer(pool2, 256, 3, name='conv3_1')
		conv3_2 = conv_layer(conv3_1, 256, 3, name='conv3_2')
		conv3_3 = conv_layer(conv3_2, 256, 3, name='conv3_3')
		if self.vgg19:
			conv3_4 = conv_layer(conv3_3, 256, 3, name='conv3_4')
			pool3 = max_pool(conv3_4, name='pool3')
		else:
			pool3 = max_pool(conv3_3, name='pool3')

		# 28x28x256 -> 28x28x512 -> 28x28x512 -> 28x28x512 -> 14x14x512
		conv4_1 = conv_layer(pool3, 512, 3, name='conv4_1')
		conv4_2 = conv_layer(conv4_1, 512, 3, name='conv4_2')
		conv4_3 = conv_layer(conv4_2, 512, 3, name='conv4_3')
		if self.vgg19:
			conv4_4 = conv_layer(conv4_3, 512, 3, name='conv4_4')
			pool4 = max_pool(conv4_4, name='pool4')
		else:
			pool4 = max_pool(conv4_3, name='pool4')

		# 14x14x512 -> 14x14x512 -> 14x14x512 ->14x14x512 -> 7x7x512
		conv5_1 = conv_layer(pool4, 512, 3, name='conv5_1')
		conv5_2 = conv_layer(conv5_1, 512, 3, name='conv5_2')
		conv5_3 = conv_layer(conv5_2, 512, 3, name='conv5_3')
		if self.vgg19:
			conv5_4 = conv_layer(conv5_3, 512, 3, name='conv5_4')
			pool5 = max_pool(conv5_4, name='pool5')
		else:
			pool5 = max_pool(conv5_3, name='pool5')

		fc_in = tf.reshape(pool5, [-1, 7*7*512])
		# 7*7*512 -> 4096
		fc6 = fc_layer(fc_in, 7*7*512, 4096, name='fc6', relu=True)
		if isTraining:
			fc6 = dropout(fc6, 0.5)

		# 4096 -> 4096
		fc7 = fc_layer(fc6, 4096, 4096, name='fc7', relu=True)
		if isTraining:
			fc7 = dropout(fc7, 0.5)

		# 4096 -> class number
		fc8 = fc_layer(fc7, 4096, self.class_num, name='fc8', relu=False)

		prob = tf.nn.softmax(fc8, name='prob')
		return prob, fc8

	def loadModel(self, sess, isFineTuring=False):
		wData = np.load(self.modelpath, encoding='bytes').item()
		for name in wData:
			with tf.variable_scope(name, reuse=True):
				for p in wData[name]:
					if len(p.shape) == 1:
						# bias
						if name.startswith('conv'):
							sess.run(tf.get_variable(name+'/bias', trainable=False).assign(p))
						elif (not name.startswith('fc8')) or (not isFineTuring):
							sess.run(tf.get_variable('b', trainable=False).assign(p))
					else:
						# weights
						if name.startswith('conv'):
							sess.run(tf.get_variable(name+'/kernel', trainable=False).assign(p))
						elif (not name.startswith('fc8')) or (not isFineTuring):
							sess.run(tf.get_variable('w', trainable=False).assign(p))

	def losses(self, labels, logits):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
		#loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		return loss

	def accurracy(self, labels, logits):
		prediction = tf.to_int64(tf.argmax(logits, 1))
		correct_prediction = tf.equal(prediction, tf.argmax(labels,1))
		accurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accurracy
