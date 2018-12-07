from Unet import UNET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
from tiles import tiling, tile_simulation
from EOS_tools import path_list_creator, path_matcher, get_name

class Unet_predictor:

    def __init__(self, model_p):
        self.model = UNET()
        self.model.load_weights(model_p)
        self.size = (256, 256)
        self.path = model_p
        self.result = {}

    def __repr__(self):
        return self.model.summary()

    def __call__(self, img):
        return self.predict(img)

    def predict(self, img):
        w, h = self.size
        w_stride, _, w_out, h_stride, _, h_out = tile_simulation(img, self.size)
        raw_mask = np.zeros((h_out, w_out, 1))
        for w_idx, h_idx, tile in tiling(img):
            tile = np.expand_dims(tile, axis=0)
            tile_mask = self.model.predict(tile)
            x, y = (w_stride * w_idx, h_stride * h_idx)
            raw_mask[y:y+h, x:x+w] = np.maximum(raw_mask[y:y+h, x:x+w], tile_mask)
            
        raw_mask = raw_mask.reshape((h_out, w_out))
        pred_mask = cv2.resize(raw_mask, img.shape[:2][::-1])
        return pred_mask

    def show_result(self) -> None:
        if len(self.result) == 0:
            print('No record yet.')
            return
        with open('record.txt', 'w+') as record:
            line = '{}    {}'.format('ID', "EOS_Count")
            record.write(line)
            print(line)
            for ID, count in self.result.items():
                line = '{}    {}'.format(ID, count)
                record.write(line)
                print(line)

    def predict_from_dir(self, dir_path, visualize_dst=None):
        self.result.clear()
        img_p_list = path_list_creator(dir_path)
        for img_p in img_p_list:
            img = cv2.imread(img_p)
            pred_mask = self.predict(img)
            pred_mask_img = (pred_mask * 255).astype(np.uint8)
            pred_cnts = mask2contour(pred_mask_img, iterations=2)

            ID = get_name(img_p)
            self.result[ID] = len(pred_cnts)

            if visualize_dst:
                img_out = mask_visualization(img, pred_cnts)
                cv2.imwrite(os.path.join(visualize_dst, ID+'.jpg'), img_out)
        self.show_result()

    def _overlap(self, label_cnts, pred_cnts, shape):
        img = cv2.drawContours(np.zeros(shape, dtype=np.uint8), label_cnts, -1, 100, -1) 
        mask = cv2.drawContours(np.zeros(shape, dtype=np.uint8), pred_cnts, -1, 100, -1)
        cover = img + mask
        cover_cnts = mask2contour(cover)
        return cover_cnts

    def metric(self, img_p: str, label_p: str, name=None) -> None:
        """true_positives = Correct objects
        false_positives = Missed objects
        false_negatives = Extra objects
        (PPV), Precision = Σ True positive / Σ Predicted condition positive
        Sensitivity, probability of detection = Σ True positive / Σ Condition positive"""
        if name:
            print(name)
        img = cv2.imread(img_p)
        mask_img = cv2.imread(label_p, 0)
        pred_mask = self.predict(img)
        pred_mask_img = (pred_mask * 255).astype(np.uint8)
        pred_cnts = mask2contour(pred_mask_img, iterations=2)
        label_cnts = mask2contour(mask_img)
        possible_right_cnts = self._overlap(label_cnts, pred_cnts, mask_img.shape)
        if name:
            img_out = mask_visualization(img, possible_right_cnts)
            cv2.imwrite(os.path.join(dst, name+'.png'), img_out)

        true_objects = len(label_cnts)
        pred_objects = len(pred_cnts)
        print("Number of true objects:", true_objects)
        print("Number of predicted objects:", pred_objects)

        true_positive = len(possible_right_cnts)
        false_positive = true_objects - true_positive
        false_negative = pred_objects - true_positive
        precision = true_positive / (true_positive + false_positive)
        sensitivity = true_positive / (true_positive + false_negative)
        print("true_positive false_positive false_negative Precision Sensitivity")
        print(" ".join([str(item) for item in (true_positive, false_positive, 
        false_negative, precision, sensitivity)]))

    def metric_from_dir(self, img_dir_path, label_dir_path, dst):
        sys.stdout = open(os.path.join(dst, 'log.txt'), 'a')
        for img_p, label_p, name in path_matcher(img_dir_path, label_dir_path):
            self.metric(img_p, label_p, name)
        sys.stdout.close()

def mask2contour(mask_img, threshold=127, iterations=0):
    # thresh = 255 - cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = 255 - cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=iterations)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
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
    try:
        label_dir = sys.argv[4]
        actor.metric_from_dir(img_dir, label_dir, dst)
    except IndexError:
        actor.predict_from_dir(img_dir, dst)