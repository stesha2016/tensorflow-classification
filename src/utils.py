import cv2
import numpy as np

def load_image(fn, w, h):
	img = cv2.imread(fn)
	img = cv2.resize(img, (w, h))
	return np.array(img)