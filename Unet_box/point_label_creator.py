import cv2
import sys
import numpy as np
import os
from Unet_box.EOS_tools import point_detect, middle, path_matcher, inside

def box_detect(hsv):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    cnts = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = [c for c in cnts if cv2.contourArea(c) > 600]
    return cnts

def useful_marks(hsv):
    points = point_detect(hsv)
    boxes = box_detect(hsv)
    useful_boxes = [box for box in boxes if not any([inside(point, box) for point in points])]
    useful_points = [point for point in points if not any([inside(point, box) for box in boxes])]
    box_points = [middle(box) for box in useful_boxes]
    useful_points.extend(box_points)
    return useful_points

def mark_point(img, points, radius=5):
    green = (0, 255, 0)
    for point in points:
        cv2.circle(img, point, radius, green, -1)
    return img

def point_label_creator(dir_img, dir_label, dst):
    os.makedirs(dst, exist_ok=True)
    for img_p, label_p, name in path_matcher(dir_img, dir_label):
        hsv = cv2.cvtColor(cv2.imread(label_p), cv2.COLOR_BGR2HSV)
        points = useful_marks(hsv)
        canvas = cv2.imread(img_p)
        draw = mark_point(canvas, points)
        cv2.imwrite(os.path.join(dst ,'{}.png'.format(name)), draw)

if __name__ == "__main__":
    dir_img = sys.argv[1]
    dir_label = sys.argv[2]
    dst = sys.argv[3]
    point_label_creator(dir_img, dir_label, dst)