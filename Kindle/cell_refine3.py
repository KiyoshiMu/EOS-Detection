import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import sys
from copy import copy
import math
from cell_segment0 import middle, from_label, islabel, watershed_labels
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator 

def match(one, another):
    for f1 in os.listdir(one):
        for f in os.listdir(another):
            if f1[:-4] == f[:-4]:
                yield (os.path.join(one, f1), os.path.join(another, f))

def refine(file, label, dst, count=0, nocount=0):
    os.makedirs(os.path.join(dst, 'Yes'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'No'), exist_ok=True)
    name = os.path.basename(file)
    points = from_label(label)

    image = cv2.imread(file)
    image = cv2.resize(image, (1920, 1440))
    image_c = image.copy()
    labels, gray_shape = watershed_labels(image)
                    
    for label in np.unique(labels):
        judge = False
        
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
        
        if any([islabel(point, c) for point in points]):
            judge = True

        one = copy(image_c[y:y+h, x:x+w])
        one_adjust = cv2.resize(one, (96, 96)) 

        classification = one_adjust.reshape((1, 96, 96, 3)) / 255
        classification = classification[...,::-1].astype(np.float32) # the RGB order is different
        pred = loaded_model.predict_classes(classification)[0][0]
        
        if pred != 1: # 1 is the interest
#             print(isis)
            if judge:
                cv2.imwrite(os.path.join(dst, 'Yes', '{:05d}.png'.format(count)), one_adjust)
                count += 1
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.imwrite(os.path.join(dst, 'No', '{:05d}.png'.format(nocount)), one_adjust)
                nocount +=1
            continue
            
        if judge: # in this case the pred is wrong, thus it is not eos
            cv2.imwrite(os.path.join(dst, 'No', '{:05d}.png'.format(nocount)), one_adjust)
            nocount +=1
            continue
            
        cv2.imwrite(os.path.join(dst, 'Yes', '{:05d}.png'.format(count)), one_adjust)
        count += 1       
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(dst, '{}{:05d}.jpg'.format(name, count)), image)
    print(count)
    return count, nocount

def main():
    raw = sys.argv[1]
    change = sys.argv[2]
    dst = sys.argv[3]
    count, nocount = 0, 0
    for img, lable in match(raw, change):
        count, nocount = refine(img, lable, dst, count=count, nocount=nocount)

if __name__ == "__main__":
    mp = sys.argv[4]
    loaded_model = models.load_model(mp)
    loaded_model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.RMSprop(lr=1e-4),
                        metrics=['acc'])
    main()