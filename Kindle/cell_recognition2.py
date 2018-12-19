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

def recognition(file, dst, name, count=0): 
    image = cv2.imread(file)
    image = cv2.resize(image, (1920, 1440))
    image_c = image.copy()
    labels, gray_shape = watershed_labels(image)
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

        one = copy(image_c[y:y+h, x:x+w])
        one_adjust = cv2.resize(one, (96, 96)) 

        classification = one_adjust.reshape((1, 96, 96, 3)) / 255
        classification = classification[...,::-1].astype(np.float32)

        isis = loaded_model.predict_classes(classification)[0][0]
        if isis != 1: # 1 is the interest
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        count += 1

    cv2.imwrite(os.path.join(dst, '{}.jpg'.format(name)), image)
    print(count)

def main():
    base = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(base):
        file = os.path.join(base, f)
        recognition(file, dst, name=f[:-4])

if __name__ == "__main__":
    mp = sys.argv[3]
    loaded_model = models.load_model(mp)
    loaded_model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.RMSprop(lr=1e-4),
                        metrics=['acc'])
    main()