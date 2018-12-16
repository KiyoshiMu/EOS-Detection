import cv2
import os
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import pandas as pd
from bisect import bisect
from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage.color import label2rgb
from Unet_box.EOS_tools import inside, middle
from datetime import datetime

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

def mark_text(img, text:str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 11, 2)[0]
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 4 * 3)
    cv2.putText(img, text, (textX, textY), font, 11, 0, 2, cv2.LINE_AA)

def overlap(mask_img, pred_cnts, use_point=False):
    if use_point:
        label_cnts = cv2.findContours(mask_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        label_points = [middle(cnt) for cnt in label_cnts]
        cover_cnts = [c for c in pred_cnts 
        if any([inside(point, c) for point in label_points])]

    else:
        shape = mask_img.shape
        label_cnts = mask2contour(mask_img)
        img = cv2.drawContours(np.zeros(shape, dtype=np.uint8), label_cnts, -1, 100, -1) 
        mask = cv2.drawContours(np.zeros(shape, dtype=np.uint8), pred_cnts, -1, 100, -1)
        cover = img + mask
        cover_cnts = mask2contour(cover)
    return cover_cnts, label_cnts

def foolish_clean(shape, cnts):
    canvas = np.zeros(shape, np.uint8)
    for cnt in cnts:
        # area = cv2.contourArea(cnt)
        draw_circle(canvas, cnt, 255, 0, -1, target=8)
    return canvas

def pred_mask_to_cnts_old(pred_mask_img):
    raw_pred_cnts = mask2contour(pred_mask_img, iterations=2)
    canvas = foolish_clean(pred_mask_img.shape, raw_pred_cnts)
    pred_cnts = mask2contour(canvas)
    return pred_cnts

def pred_mask_to_cnts(pred_mask_img, region=True):
    if region:
        pred_cnts = mask_to_cnts_region(pred_mask_img)
    else:
        pred_cnts = mask_to_cnts_watershed(pred_mask_img)
    return pred_cnts

def mask2contour(mask_img, threshold=128, iterations=0):
    # thresh = 255 - cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = 255 - cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=iterations)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def draw_circle(img, c, use_color, expansion=0, thickness=2, target=None):
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = target or int(radius) + expansion
    cv2.circle(img, center, radius, use_color, thickness)

def draw_colorful(img, cnts):
    label = np.zeros((img.shape[:2]))
    for number, c in enumerate(cnts, start=1):
        draw_origin(label, [c], number, thickness=-1)
    colorful_img = np.array(label2rgb(label, image=img)*255, np.uint8)
    return colorful_img

def draw_origin(img, cnts, use_color, thickness=2):
    cv2.drawContours(img, cnts, -1, use_color, thickness)

def mask_visualization(img, cnts, method='circle', **kargs):
    if method == 'colorful':
        cur_img = draw_colorful(img, cnts)
    else:
        cur_img = img.copy()
        color=kargs.get('color', 'cyan')
        color_dict = {'cyan':(255, 255, 0), 'green':(0, 255, 0), 'black': (0, 0, 0)}
        use_color = color_dict.get(color, (255, 255, 0))
        if method == 'circle':
            expansion=kargs.get('expansion', 0)
            target=kargs.get('target', None)
            for c in cnts:
                draw_circle(cur_img, c, use_color, expansion, target=target)
        else:
            draw_origin(cur_img, cnts, use_color)
            
    return cur_img

def stats(label_cnts, pred_cnts, possible_right_cnts):
    true_objects = len(label_cnts)
    pred_objects = len(pred_cnts)
    true_positive = len(possible_right_cnts)
    false_positive = true_objects - true_positive
    false_negative = pred_objects - true_positive
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)

    values = [true_objects, pred_objects, true_positive, 
    false_positive, false_negative, precision, sensitivity]
    return values

def show_result(info:dict, columns:list, write:bool=True, 
    title:str=None, dst:str=None) -> None:
    if len(info) == 0:
        print('No record yet.')
        
    if write:
        df = pd.DataFrame.from_dict(info, orient='index', 
        columns=columns)
        if title:
            stamp = f'Result_{title}_{str(datetime.now())}.xlsx'
        else:
            stamp = f'Result_{str(datetime.now())}.xlsx'
        if not dst:
            dst = '.'
        df.to_excel(os.path.join(dst, stamp))