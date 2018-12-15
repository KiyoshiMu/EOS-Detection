import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from bisect import bisect
from skimage.filters import sobel
from scipy import ndimage as ndi

def segmentations_filters(shape, labels):
    canvas = np.zeros(shape, np.uint8)
    ranks_for_big = [800, 1000, 1200, 1400]
    rank_for_small = [100, 250, 400, 550]
    for label in np.unique(labels):
        if label == 0:
            continue
        temp = np.zeros(shape, np.uint8)
        temp[labels == label] = 255
        c = cv2.findContours(temp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2][0]
        area = cv2.contourArea(c)
        if area <= 10:
            continue
        if area > 600:
            temp = cv2.erode(temp, np.ones((5, 5), np.uint8), 
            iterations=bisect(ranks_for_big, area))
        else:
            temp = cv2.dilate(temp, np.ones((3, 3), np.uint8), 
            iterations=len(rank_for_small)-bisect(rank_for_small, area))
        canvas[temp == 255] = 255
    cnts = cv2.findContours(canvas, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    return [c for c in cnts if cv2.contourArea(c) > 100]

def mask_to_cnts_watershed(mask_img):
    # shifted = cv2.GaussianBlur(mask_img, (3, 3), 0)
    thresh = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=3, labels=thresh)
    markers = ndimage.label(localMax)[0]
    labels = watershed(D * (-1), markers, mask=thresh)
    return segmentations_filters(mask_img.shape, labels)

def mask_to_cnts_region(mask_img):
    elevation_map = sobel(mask_img)
    markers = np.zeros_like(mask_img)
    markers[mask_img < 25] = 1
    markers[mask_img > 127] = 2

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labels, _ = ndi.label(segmentation)
    return segmentations_filters(mask_img.shape, labels)
    # image_label_overlay = label2rgb(labeled_coins, image=raw)