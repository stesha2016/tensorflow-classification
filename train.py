import sys
import tensorflow as tf
import numpy as np
import networks.vgg as vgg
import networks.inception_v4 as inceptionV4
import networks.inception_resnet_v2 as inceptionResnetV2
import networks.resnet_v2 as resnetv2
import networks.mobilenet_v1 as mobilenetv1
import src.utils as utils
import random

classes_name = ['person', 'check-in', 'polo', 'goldfish', 'ray', 'electric ray', 'stingray', 'bird', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch']
classes_id = ['n00007846', 'n00141669', 'n00477639', 'n01443537', 'n01495701', 'n01496331', 'n01498041', 'n01503061', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829']
epoch_iter = 200
Mean = np.array([103.939, 116.779, 123.68]).reshape((1, 1, 3))
tensorboard = True
slim = tf.contrib.slim
def minibatch(file_list, batchsize, w, h):
	length = len(file_list)
	i = 0
	epoch = 0
	random.shuffle(file_list)
	while True:
		if i + batchsize >= length:
			random.shuffle(file_list)
			epoch += 1
			i = 0
		images = []
		labels = []
		for j in range(i, i+batchsize):
			content = file_list[j]
			npos = content.index(',')
			path = content[:npos]
			classid = content[npos+1:len(content)-1]

			image = utils.load_image(path, w, h)
			index = classes_id.index(classid)
			label = np.zeros(len(classes_id), dtype=np.int)
			label[index] = 1

			images.append(image)
			labels.append(label)
		i += batchsize
		images = np.array(images, dtype=np.float32)
		labels = np.array(labels, dtype=np.float32)
		yield epoch, images, labels

def train(cfg_path):
	cfg = utils.get_cfg(cfg_path)
	network = cfg['net']
	print('prepare network...')
	w = cfg['width']
	h = cfg['height']
	if network == 'mobilenetv1':
		w = int(w * cfg['resolution_multiplier'])
		h = int(h * cfg['resolution_multiplier'])
	x = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3])
	y = tf.placeholder(dtype=tf.float32, shape=[None, len(classes_id)])
	if network == 'vgg':
		if cfg['isvgg19'] == 'true':
			vgg_network = vgg.Vgg(x, len(classes_id), True, cfg['modelpath'])
		else:
			vgg_network = vgg.Vgg(x, len(classes_id), False, cfg['modelpath'])
		predictions, logits = vgg_network.build()
		loss = vgg_network.losses(y, logits)
		accurracy = vgg_network.accurracy(y, logits)
	elif network == 'inceptionv4':
		predictions, logits = inceptionV4.inception_v4(x, len(classes_id))
		loss = inceptionV4.losses(y, logits)
		accurracy = inceptionV4.accurracy(y, logits)
	elif network == 'inceptionResnetV2':
		predictions, logits = inceptionResnetV2.inception_resnet_v2(x, len(classes_id))
		loss = inceptionResnetV2.losses(y, logits)
		accurracy = inceptionResnetV2.accurracy(y, logits)
	elif network == 'resnetv2':
		predictions, logits = resnetv2.resnet_v2_50(x, len(classes_id))
		loss = resnetv2.losses(y, logits)
		accurracy = resnetv2.accurracy(y, logits)
	elif network == 'mobilenetv1':
		predictions, logits = mobilenetv1.mobilenet_v1(x, len(classes_id), depth_multiplier=cfg['depth_multiplier'])
		loss = mobilenetv1.losses(y, logits)
		accurracy = mobilenetv1.accurracy(y, logits)
	else:
		loss = 0
		accurracy = 0

	if tensorboard:
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('acc', accurracy)
		merge = tf.summary.merge_all()
		writer = tf.summary.FileWriter(logdir='./summary/')

	if network == 'vgg' and cfg['finetuning'] == 'true':
		T_list = tf.trainable_variables()
		V_list = [var for var in T_list if var.name.startswith('fc8')]
	else:
		V_list = tf.trainable_variables()

	if cfg['optimizer'] == 'RMSProp':
		print('Optimizer is RMSProp')
		optim = tf.train.RMSPropOptimizer(learning_rate=cfg['learningrate'], epsilon=1.0).minimize(loss, var_list=V_list)
	else:
		print('Optimizer is GradientDescent')
		optim = tf.train.GradientDescentOptimizer(learning_rate=cfg['learningrate']).minimize(loss, var_list=V_list)

	print('prepare data...')
	file_list = utils.load_data(cfg['trainpath'])
	batch = minibatch(file_list, cfg['batchsize'], cfg['width'], cfg['height'])
	val_list = utils.load_data(cfg['valpath'])
	val_batch = minibatch(val_list, cfg['batchsize'], cfg['width'], cfg['height'])

	print('start training...')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		if network == 'vgg' and cfg['finetuning'] == 'true':
			vgg_network.loadModel(sess, True)
		epoch = 0
		iteration = 0
		loss_mean = 0
		while epoch < epoch_iter:
			iteration += 1
			epoch, images, labels = next(batch)
			if network == 'vgg':
				images = images - np.array(cfg['mean']).reshape(1, 1, 1, 3)
			loss_curr, summary, _ = sess.run([loss, merge, optim], feed_dict={x: images, y: labels})
			loss_mean += loss_curr
			writer.add_summary(summary, iteration)
			if (iteration % 500 == 0):
				print('epoch/iter: [{}/{}], loss_mean: {}'.format(epoch, iteration, loss_mean/500))
				loss_mean = 0
				_, val_images, val_labels = next(val_batch)
				if network == 'vgg':
					val_images = val_images - np.array(cfg['mean']).reshape(1, 1, 1, 3)
				cc = sess.run(accurracy, feed_dict={x: val_images, y: val_labels})
				print('accurracy: {}'.format(cc))
		writer.close()

def main():
	assert(len(sys.argv) > 1)
	cfg_path = sys.argv[1]
	train(cfg_path)

if __name__ == '__main__':
	main()