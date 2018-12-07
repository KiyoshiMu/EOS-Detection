import cv2
import os
import sys
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from EOS_tools import point_detect, inside, middle, path_matcher

def watershed_labels(img):
    """the lines is adapted from https://www.pyimagesearch.com/2015/11/02/watershed-opencv"""
    shifted = cv2.GaussianBlur(img, (7, 7), 0)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, np.ones((7, 7),np.uint8), iterations=0)
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=5, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(D * (-1), markers, mask=thresh)
    
    return labels

def mask_creator(img_p, label_p, dst, name):
    """"""
    print(img_p, label_p)
    img = cv2.resize(cv2.imread(img_p), (1920, 1440))
    label = cv2.resize(cv2.imread(label_p), (1920, 1440))
    label_hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    points = point_detect(label_hsv)
    
    mask_out = np.zeros(img.shape[:2], np.uint8)
    labels = watershed_labels(img)
                    
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500 and any([inside(point, c) for point in points]):
            cv2.drawContours(mask_out, [c], -1, 255, -1)
    
    cv2.imwrite(os.path.join(dst ,'{}.png'.format(name)), mask_out)

def point2mask(dir_img, dir_label, dst):
    os.makedirs(dst, exist_ok=True)
    for img_p, label_p, name in path_matcher(dir_img, dir_label):
        mask_creator(img_p, label_p, dst, name)

if __name__ == "__main__":
    dir_img = sys.argv[1]
    dir_label = sys.argv[2]
    dst = sys.argv[3]
    point2mask(dir_img, dir_label, dst)