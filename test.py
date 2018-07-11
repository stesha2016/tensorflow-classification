import sys
import tensorflow as tf
import numpy as np
import networks.vgg as vgg
import src.utils as utils

modelPath16 = './models/vgg16.npy'
modelPath19 = './models/vgg19.npy'
def test_vgg(fn, vgg19=False):
	print('test_vgg', fn)
	x = tf.placeholder(dtype='float32', shape=[None, 224, 224, 3])
	if vgg19:
		model = modelPath19
	else:
		model = modelPath16
	vgg16 = vgg.Vgg(x, 1000, vgg19, model)
	prob = vgg16.build()
	img = utils.load_image(fn, 224, 224)
	Mean = np.array([103.939, 116.779, 123.68])
	img = img - Mean
	batch1 = img.reshape([1, 224, 224, 3])

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		vgg16.loadModel(sess)
		out = sess.run(prob, feed_dict={x: batch1})[0][0]
		classes_num = len(out)
		print(classes_num)
		import data.vgg_classes as classes
		pred = np.argsort(out)
		for i in range(5):
			index = classes_num - i - 1
			print(pred[index], out[pred[index]], classes.class_names[pred[index]])

def main():
	fn = sys.argv[2]
	if sys.argv[1] == 'vgg16':
		test_vgg(fn)
	elif sys.argv[1] == 'vgg19':
		test_vgg(fn, True)

if __name__ == '__main__':
	main()