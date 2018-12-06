from Unet import UNET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
from tiles import tiling, tile_simulation
from EOS_tools import path_list_creator

class Unet_predictor:

    def __init__(self, model_p):
        self.model = UNET()
        self.model.load_weights(model_p)
        self.size = (256, 256)

    def __repr__(self):
        return self.model.summary()

    def __call__(self, img):
        return self.predict(img)

    def predict(self, img):
        w, h = self.size
        w_stride, _, w_out, h_stride, _, h_out = tile_simulation(img, self.size)
        pred_mask = np.zeros((h_out, w_out, 1))
        for w_idx, h_idx, tile in tiling(img):
            tile = tile.reshape((1,) + tile.shape)
            tile_mask = self.model.predict(tile)
            x, y = (w_stride * w_idx, h_stride * h_idx)
            pred_mask[y:y+h, x:x+w] = np.maximum(pred_mask[y:y+h, x:x+w], tile_mask)
            
        raw_mask = pred_mask.reshape((h_out, w_out))
        mask = cv2.resize(raw_mask, img.shape[:2][::-1])
        return mask

    def predict_from_dir(self, dir_path, visualize_dst=None):
        img_p_list = path_list_creator(dir_path)
        with open('record.txt', 'w+') as record:
            record.write('{}    {}'.format('ID', "EOS_Count"))
            for img_p in img_p_list:
                img = cv2.imread(img_p)
                mask = self.predict(img)
                cnts = mask2contour(mask)
                ID = os.path.basename(img_p).split('.')[0]
                line = '{}    {}\n'.format(ID, len(cnts))
                record.write(line)
                if visualize_dst:
                    img_out = mask_visualization(img, cnts)
                    cv2.imwrite(os.path.join(visualize_dst, ID+'.jpg'), img_out) 

def mask2contour(mask):
    mask = (mask * 255).astype(np.uint8)
    thresh = 255 - cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5),np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def mask_visualization(img, cnts):
    for c in cnts:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius) + 3
        cv2.circle(img,center,radius,(255, 255, 0), 2)
    return img

if __name__ == "__main__":
    model_p = sys.argv[1]
    img_dir = sys.argv[2]
    dst = sys.argv[3]
    actor = Unet_predictor(model_p)
    os.makedirs(dst, exist_ok=True)
    actor.predict_from_dir(img_dir, dst)