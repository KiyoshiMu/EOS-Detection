import cv2
import os
import numpy as np

def middle(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def point_detect(hsv, yellow=False):
    """"""
    if yellow:
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    else:
        lower = np.array([50,100, 50])
        upper = np.array([70,255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    points = [middle(cnt) for cnt in cnts]
    return points

def get_name(path, more_ornaments=None) -> str:
    """more_ornaments is a list. Its default value is None"""
    ornaments = ['_label', '_pred', '_fit']
    if more_ornaments:
        ornaments.extend(more_ornaments)
    name = os.path.splitext(os.path.basename(path))[0]
    for clear in ornaments:
        name = name.replace(clear, '')
    return name

def path_matcher(dir_img, dir_label):
    img_path_list = path_list_creator(dir_img)
    label_path_list = path_list_creator(dir_label)
    label_path_dict = dict([(get_name(p), p) for p in label_path_list])
    for img_path in img_path_list:
        name = get_name(img_path)
        label_path = label_path_dict.get(name)
        if label_path:
            yield img_path, label_path, name

def inside(point, cnt):
    return cv2.pointPolygonTest(cnt, point, True) >= -3

def path_list_creator(dir_img, dir_label=None):
    img_p_list, label_p_list = [], []
    if dir_label:
        for img_p, label_p, _ in path_matcher(dir_img, dir_label):
            img_p_list.append(img_p)
            label_p_list.append(label_p)
        return img_p_list, label_p_list
    else:
        img_p_list = [os.path.join(item[0], f_p) for item in os.walk(dir_img) for f_p in item[2] if item[2]]
        return img_p_list

def read_from_path_list(path_list, islabel=False, w=256, h=256):
    dtype = np.bool if islabel else np.uint8
    channel = 1 if islabel else 3
    data = np.zeros((len(path_list), h, w, channel), dtype=dtype)
    for n, path in enumerate(path_list):
        if islabel:
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (w, h)).reshape((h, w, 1))
        else:
            img = cv2.imread(path)[:,:,:channel]
            img = cv2.resize(img, (w, h))
        data[n] = img
    return data

def data_creator(dir_img, dir_label):
    img_p_list, label_p_list = path_list_creator(dir_img, dir_label)
    img_data = read_from_path_list(img_p_list)
    label_data = read_from_path_list(label_p_list, islabel=True)
    return img_data, label_data