import cv2
import numpy as np

def load_image(fn, w, h):
	img = cv2.imread(fn)
	img = cv2.resize(img, (w, h))
	return np.array(img)

def load_data(data_path):
	f = open(data_path)
	path_class_list = f.readlines()
	f.close()
	path_list = []
	label_list = []
	for i in range(len(path_class_list)):
		content = path_class_list[i]
		npos = content.index(',')
		path = content[:npos-1]
		classid = content[npos+1:]
		path_list.append(path)
		label_list.append(classid)
	return path_list, label_list

def index_list(length):
	indexs = []
	for i in range(length):
		indexs.append(i)
	return indexs