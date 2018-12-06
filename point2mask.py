import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import random
from EOS_tools import path_matcher, point_detect, inside

def watershed_labels(image):
# the lines is adapted from https://www.pyimagesearch.com/2015/11/02/watershed-opencv

    shifted = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, np.ones((7, 7),np.uint8), iterations=1)
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=5, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(D * (-1), markers, mask=thresh)
    
    return labels, gray.shape

def segment(img_p, label_p, dst, name):
    """Segment possibel cells from a large image. Then, based on the labeled image, interested part 
    and non-interested part are divided into two groups. The small cell images are saved as .png files
    under the dst directory.
    file (str): the path of a primary image;
    lable (str): the path of a labeled image;
    dst (str): the path of the output directory."""
    print(img_p, label_p)
    image = cv2.resize(cv2.imread(img_p), (1920, 1440))
    label = cv2.resize(cv2.imread(label_p), (1920, 1440))
    label_hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    points = point_detect(label_hsv)
    
    mask_out = np.zeros(image.shape[:2], np.uint8)
    labels, gray_shape = watershed_labels(image)
                    
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray_shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500 and any([inside(point, c) for point in points]):
            cv2.drawContours(mask_out, [c], -1, 255, -1)
    
    cv2.imwrite(os.path.join(dst ,'{}.png'.format(name)), mask_out)
#     plt.figure(dpi=300)
#     plt.imshow(image)


def point2mask(dir_img, dir_label, dst):
    os.makedirs(dst, exist_ok=True)
    for img_p, label_p, name in path_matcher(dir_img, dir_label):
        segment(img_p, label_p, dst, name)