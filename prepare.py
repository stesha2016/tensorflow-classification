import os

ROOT = './data/my_vgg_data/train/'
f = open('./data/my_vgg_data/train.txt', 'w')
for classes in os.listdir(ROOT):
	subpath = ROOT + classes + '/'
	print(subpath)
	for file in os.listdir(subpath):
		#print(file)
		f.writelines(subpath + file + ',' + classes + '\n')
f.close()