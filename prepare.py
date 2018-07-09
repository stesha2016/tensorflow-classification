import os

ROOT = './data/my_vgg_data/val/'
f = open('./data/my_vgg_data/val.txt', 'w')
for classes in os.listdir(ROOT):
	subpath = ROOT + classes + '/'
	print(subpath)
	for file in os.listdir(subpath):
		#print(file)
		f.writelines(subpath + file + ',' + classes + '\n')
f.close()