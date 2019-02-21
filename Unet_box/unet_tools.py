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
    ranks_for_big = [750, 900, 1050, 1200, 1350]
    rank_for_small = [300]
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

def segmentations_filters_special(shape, labels):
    cnts = []
    for label in np.unique(labels):
        if label == 0:
            continue
        temp = np.zeros(shape, np.uint8)
        temp[labels == label] = 255
        c = cv2.findContours(temp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2][0]
        cnts.append(c)
    return [c for c in cnts if cv2.contourArea(c) > 300]

def mask_to_cnts_watershed(mask_img, min_distance=3, for_real_mask=False):
    # shifted = cv2.GaussianBlur(mask_img, (3, 3), 0)
    thresh = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=min_distance, labels=thresh)
    markers = ndimage.label(localMax)[0]
    labels = watershed(D * (-1), markers, mask=thresh)

    if for_real_mask:
        return segmentations_filters_special(mask_img.shape, labels)

    return segmentations_filters(mask_img.shape, labels)

def mask_to_cnts_watershed_thresh(mask_img, threshold=0.5):

    img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    threshold = threshold * 255
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel)
    dilationKernel = np.ones((3,3), np.uint8)
    bg = cv2.dilate(closed, dilationKernel, iterations=3)
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist_transform,0.7*dist_transform.max(), 255, 0)
    fg = np.uint8(fg)
    unknown = cv2.subtract(bg, fg)
    _, marker = cv2.connectedComponents(fg)
    marker = marker+1
    marker[unknown==255] = 0

    cv2.watershed(img, marker)
    # maxMarker = np.max(marker)
    # minMarker = np.min(marker)
    return segmentations_filters_special(mask_img.shape, marker)

def mask_to_cnts_region(mask_img, for_real_mask=True, threshold=96):
    elevation_map = sobel(mask_img)
    markers = np.zeros_like(mask_img)
    markers[mask_img < 25] = 1
    markers[mask_img > threshold] = 2

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labels, _ = ndi.label(segmentation)
    if for_real_mask:
        return segmentations_filters_special(mask_img.shape, labels)

    return segmentations_filters(mask_img.shape, labels)

def mark_text(img, text:str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 11, 2)[0]
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 3 * 2.2)
    cv2.putText(img, text, (textX, textY), font, 11, 0, 4, cv2.LINE_AA)

def overlap(pred_cnts, mask_img=None, label_points:list=None) -> list and int:
    if label_points:
        cover_cnts = [c for c in pred_cnts 
        if any([inside(point, c, threshold=-7) for point in label_points])] # half of the minDistance 14 in point_loc_saver.py
        label_num = len(label_points)
    else:
        shape = mask_img.shape
        label_cnts = mask2contour(mask_img, iterations=1)
        img = cv2.drawContours(np.zeros(shape, dtype=np.uint8), label_cnts, -1, 100, -1) 
        mask = cv2.drawContours(np.zeros(shape, dtype=np.uint8), pred_cnts, -1, 100, -1)
        cover = img + mask
        cover_cnts = mask2contour(cover)
        label_num = len(label_cnts)
    return cover_cnts, label_num

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

def pred_mask_to_cnts(pred_mask_img, region=True, thresh=None):
    if thresh:
        pred_cnts = mask_to_cnts_watershed_thresh(pred_mask_img, threshold=thresh)
    elif region:
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

def draw_box(img, c, use_color, thickness=5, target=None):
    x, y, poss_w, poss_h = cv2.boundingRect(c)
    side = target or max((poss_w, poss_h))
    line = round(side/3)
    cv2.rectangle(img, (x-line, y-line), (x+2*line, y+2*line), use_color, thickness)

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
            expansion = kargs.get('expansion', 0)
            target = kargs.get('target', None)
            thickness = kargs.get('thickness', 2)
            for c in cnts:
                draw_circle(cur_img, c, use_color, expansion, thickness=thickness, target=target)
        elif method == 'box':
            target = kargs.get('target', None)
            thickness = kargs.get('thickness', 2)
            for c in cnts:
                draw_box(cur_img, c, use_color, thickness=thickness, target=target)
        else:
            draw_origin(cur_img, cnts, use_color)
            
    return cur_img

def stats(label_num, pred_cnts, possible_right_cnts) -> list:
    ''''retrun a list including "true_objects, pred_objects, true_positive, 
    false_positive, false_negative, precision, sensitivity" in order'''
    true_objects = label_num
    pred_objects = len(pred_cnts)
    true_positive = len(possible_right_cnts)
    false_positive = pred_objects - true_positive
    false_negative = true_objects - true_positive
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 0
    try:
        sensitivity = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        sensitivity = 0
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
        now = datetime.now()
        time = now.strftime('%m-%d_%H-%M')
        if title:
            stamp = f'Result_{title}_{time}.xlsx'
        else:
            stamp = f'Result_{time}.xlsx'
        if not dst:
            dst = '.'
        df.to_excel(os.path.join(dst, stamp))