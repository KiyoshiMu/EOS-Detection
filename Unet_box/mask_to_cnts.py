import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from bisect import bisect

def mask_to_cnts_watershed(mask_img):
    
    # shifted = cv2.GaussianBlur(raw, (3, 3), 0)
    thresh = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=15, labels=thresh)
    markers = ndimage.label(localMax)[0]
    labels = watershed(D * (-1), markers, mask=thresh)

    canvas = np.zeros(mask_img.shape, dtype="uint8")
    ranks_for_big = [750, 1000]
    rank_for_small = [100, 200, 300, 500]
    for label in np.unique(labels):
        if label == 0:
            continue
        temp = np.zeros(mask_img.shape, dtype="uint8")
        temp[labels == label] = 255
        c = cv2.findContours(temp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2][0]
        area = cv2.contourArea(c)
        if area <= 50:
            continue
        if area > 600:
            temp = cv2.erode(temp, np.ones((3, 3),np.uint8), 
            iterations=bisect(ranks_for_big, area))
        else:
            temp = cv2.dilate(temp, np.ones((5, 5),np.uint8), 
            iterations=len(rank_for_small)-bisect(rank_for_small, area))
        canvas[temp == 255] = 255

    cnts = cv2.findContours(canvas, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    return [c for c in cnts if cv2.contourArea(c) > 100]
# cells_info = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 50]