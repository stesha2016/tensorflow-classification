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
	file_path = []
	for i in range(len(path_class_list)):
		file_path.append(path_class_list[i])
	return file_path
