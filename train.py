#import tensorflow as tf
import networks.vgg16 as vgg

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.session(config=config)

vgg16 = vgg.Vgg16()