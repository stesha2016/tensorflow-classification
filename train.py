#import tensorflow as tf
import sys
import tensorflow as tf
import numpy as np
import networks.vgg as vgg
import src.utils as utils

classes_name = ['person', 'check-in', 'polo', 'goldfish', 'ray', 'electric ray', 'stingray', 'bird', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch']
classes_id = ['n00007846', 'n00141669', 'n00477639', 'n01443537', 'n01495701', 'n01496331', 'n01498041', 'n01503061', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829']
def train_vgg(data_path, vgg19=False):
	print('train_vgg', data_path)
	print('prepare network...')
	classes_num = 14
	x = tf.placeholder(dtype='float32', shape=[None, 224, 224, 3])
	y = tf.placeholder(dtype='float32', shape=[None, classes_num])
	vgg16 = vgg.Vgg(x, classes_num, vgg19, model)
	prob, logit = vgg16.build()
	loss = vgg16.losses(y, logit)
	optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=tf.trainable_variable())

	print('prepare data...')

def main():
	data_path = sys.argv[2]
	if sys.argv[1] == 'vgg16':
		train_vgg(data_path)
	elif sys.argv[1] == 'vgg19':
		train_vgg(data_path, True)

if __name__ == '__main__':
	main()