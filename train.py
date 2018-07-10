import sys
import tensorflow as tf
import numpy as np
import networks.vgg as vgg
import src.utils as utils
import random

classes_name = ['person', 'check-in', 'polo', 'goldfish', 'ray', 'electric ray', 'stingray', 'bird', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch']
classes_id = ['n00007846', 'n00141669', 'n00477639', 'n01443537', 'n01495701', 'n01496331', 'n01498041', 'n01503061', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829']
bs = 16
w = 224
h = 224
epoch_iter = 200
lr = 0.001
VAL_PATH = './data/my_vgg_data/val.txt'
TRAIN_PATH = './data/my_vgg_data/train.txt'
Mean = np.array([103.939, 116.779, 123.68]).reshape((1, 1, 3))
tensorboard = True
def minibatch(file_list, batchsize):
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
			image = image - Mean
			index = classes_id.index(classid)
			label = np.zeros(len(classes_id), dtype=np.int)
			label[index] = 1

			images.append(image)
			labels.append(label)
		i += batchsize
		images = np.array(images, dtype=np.float32)
		labels = np.array(labels, dtype=np.float32)
		yield epoch, images, labels

def train_vgg16(data_path, isFineTurning):
	train_vgg(data_path, isFineTurning)

def train_vgg19(data_path, isFineTurning):
	train_vgg(data_path, isFineTurning, True)

def train_vgg(data_path, fineTurning=False, vgg19=False):
	print('prepare network...')
	x = tf.placeholder(dtype='float32', shape=[None, 224, 224, 3])
	y = tf.placeholder(dtype='float32', shape=[None, len(classes_id)])
	vgg16 = vgg.Vgg(x, len(classes_id), vgg19)
	prob, logit = vgg16.build()
	loss = vgg16.losses(y, logit)
	accurracy = vgg16.accurracy(y, logit)
	if tensorboard:
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('acc', accurracy)
		merge = tf.summary.merge_all()
		writer = tf.summary.FileWriter(logdir='./summary/')
	if fineTurning:
		T_list = tf.trainable_variables()
		fc8_list = [var for var in T_list if var.name.startswith('fc8')]
		print(fc8_list)
		optim = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, var_list=fc8_list)
	else:
		optim = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, var_list=tf.trainable_variables())

	print('prepare data...')
	file_list = utils.load_data(data_path)
	batch = minibatch(file_list, bs)
	val_list = utils.load_data(VAL_PATH)
	val_batch = minibatch(val_list, bs)

	print('start training...')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		if fineTurning:
			vgg16.loadModel(sess, fineTurning)
		epoch = 0
		iteration = 0
		while epoch < epoch_iter:
			iteration += 1
			epoch, images, labels = next(batch)
			loss_curr, summary, _ = sess.run([loss, merge, optim], feed_dict={x: images, y: labels})
			writer.add_summary(summary, iteration)
			if (iteration % 500 == 0):
				print('epoch/iter: [{}/{}], loss_curr: {}'.format(epoch, iteration, loss_curr))
				_, val_images, val_labels = next(val_batch)
				cc = sess.run(accurracy, feed_dict={x: val_images, y: val_labels})
				print('accurracy: {}'.format(cc))
		writer.close()

def main():
	try:
		# input anything will trigger fine turning
		print('input ' + sys.argv[2] + ' will trigger fine turning')
		isFineTurning = True
	except:
		isFineTurning = False
	if sys.argv[1] == 'vgg16':
		train_vgg16(TRAIN_PATH, isFineTurning)
	elif sys.argv[1] == 'vgg19':
		train_vgg19(TRAIN_PATH, isFineTurning)

if __name__ == '__main__':
	main()