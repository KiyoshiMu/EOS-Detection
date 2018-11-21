import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
from copy import copy
import math
import matplotlib.pyplot as plt

"""The first step of our project. This stript is used to segment the possible cells from slides. 
Due to the large number of cells in each image, we use a shallow method to make the detection problem
become a classfication problem, that is we make small cell images by watershed algorithm first, then we
assign the label images into a group and others into another group. As a result, the problem which should
be solved is just a classification problem."""

def middle(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def from_label(label_img):
    """Read one labeled image, and save the interested location.
    label_img (str), the path of a labeled image."""
    frame = cv2.imread(label_img)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([50,100, 50])
    upper_red = np.array([70,255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    points = [middle(cnt) for cnt in cnts]
    return points

def islabel(point, cnt):
    return cv2.pointPolygonTest(cnt, point, True) >= -3

def watershed_labels(image):
# the lines is adapted from https://www.pyimagesearch.com/2015/11/02/watershed-opencv

    shifted = cv2.GaussianBlur(image, (15, 15), 0)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=14,labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(D * (-1), markers, mask=thresh)
    return labels, gray.shape

def segment(file, label, dst, name='name', count=0, nocount=0):
    """Segment possibel cells from a large image. Then, based on the labeled image, interested part 
    and non-interested part are divided into two groups. The small cell images are saved as .png files
    under the dst directory.
    file (str): the path of a primary image;
    lable (str): the path of a labeled image;
    dst (str): the path of the output directory."""
    points = from_label(label)
    
    image = cv2.imread(file)
    image = cv2.resize(image, (1920, 1440))
    image_c = image.copy()
    labels, gray_shape = watershed_labels(image)
    
    os.makedirs(os.path.join(dst, 'Yes'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'No'), exist_ok=True)
                    
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray_shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
            
        x,y,w,h = cv2.boundingRect(c)
        S1 = w * h
        if S1 < 1000 or S1 > 5000:
            continue
#         ell=cv2.fitEllipse(c) 
#         S2 =math.pi * ell[1][0] * ell[1][1] / 4 # S = pi * a * b
#         if (S1/S2) < 0.7:
#             continue
#         if w / h < 0.5 or w / h > 2:
#             continue
        one = copy(image_c[y:y+h, x:x+w])
        one_adjust = cv2.resize(one, (96, 96))
        
        if any([islabel(point, c) for point in points]):
            cv2.imwrite(os.path.join(dst, 'Yes', '{:05d}.png'.format(count)), one_adjust)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(image, "#{:04d}".format(count), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            count += 1
        else:
            cv2.imwrite(os.path.join(dst, 'No', '{:05d}.png'.format(nocount)), one_adjust)
            nocount +=1

    cv2.imwrite(os.path.join(dst, '{}{:05d}.jpg'.format(name, count)), image)
    print(count, nocount)
    return count, nocount
#     cv2.imshow("Output", image)
#     cv2.waitKey(0)

def rename(base): 
    for f in os.listdir(base): 
        if '标记' in f:
            sta = os.path.join(base, f)
            name = f[:-6] + 'label.tif'
            dst = os.path.join(base, name)
            os.rename(sta, dst)

def main(base, dst):
    path_pool = [os.path.join(base, f) for f in os.listdir(base)]
    count, nocount = 0, 0
    for i in range(0, len(path_pool), 2):
        pri = path_pool[i]
        label = path_pool[i+1]
        if 'label' in pri:
            pri, label = label, pri
        fn = os.path.basename(pri)
        count, nocount = segment(pri, label, dst, name=fn, count=count, nocount=nocount)
        print(i)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--func', required=False, help='if you need rename the label images, input "re"')
    ap.add_argument('-i', '--input', required=True, help='path to the input directory')
    ap.add_argument('-o', '--output', required=False, help='path to the output directory')
    args = vars(ap.parse_args())
    base = args['input']
    if args['func'] and args['func'] == 're':
        rename(base)
    else:
        if args['output']:
            dst = args['output']
            main(base, dst)
        else:
            print('segmentation needs output path')