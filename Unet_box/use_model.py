import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
import pandas as pd
from bisect import bisect_left
from scipy.stats import describe
from math import pi, sqrt
from Unet_box.Unet import UNET
from Unet_box.tile_creator import tiling, tile_simulation
from Unet_box.EOS_tools import path_list_creator, path_matcher, get_name

class Unet_predictor:

    def __init__(self, model_p):
        self.model = UNET()
        self.model.load_weights(model_p)
        self.size = (256, 256)
        self.path = model_p
        self.result = {}
        self.record = {}
        self.metric_keys = "Number_of_true_objects, Number_of_predicted_objects, true_positive, false_positive, false_negative, Precision, Sensitivity"
        self.cell_size = []

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

    def show_result(self, write=False) -> None:
        if len(self.result) == 0:
            print('No record yet.')
            return
        if write:
            sys.out = open('record.txt', 'w+')

        for ID, count in self.result.items():
            line = '{}    {}'.format(ID, count)
            print(line)

        if write:
            sys.out.close()
        
    def predict_from_dir(self, dir_path, visualize_dst=None, show_mask=True, target=15):
        self.result.clear()
        img_p_list = path_list_creator(dir_path)
        for img_p in img_p_list:
            img = cv2.imread(img_p)
            pred_cnts, pred_mask_img = self._mask_creator(img)
            ID = get_name(img_p)
            self.result[ID] = len(pred_cnts)

            if visualize_dst:
                pred_out = mask_visualization(img, pred_cnts, target=target)
                cv2.imwrite(os.path.join(visualize_dst, ID+'_pred.jpg'), pred_out)
                if show_mask:
                    cv2.imwrite(os.path.join(visualize_dst, ID+'_mask.jpg'), pred_mask_img)
        self.show_result()

    def _overlap(self, label_cnts, pred_cnts, shape):
        img = cv2.drawContours(np.zeros(shape, dtype=np.uint8), label_cnts, -1, 100, -1) 
        mask = cv2.drawContours(np.zeros(shape, dtype=np.uint8), pred_cnts, -1, 100, -1)
        cover = img + mask
        cover_cnts = mask2contour(cover)
        return cover_cnts

    def _foolish_clean(self, shape, cnts, count=True):
        canvas = np.zeros(shape, np.uint8)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # if area <= 100:
            #     continue
            if count:
                self.cell_size.append(area)
            draw_circle(canvas, cnt, 255, 0, -1, target=8)
        return canvas

    def _pred_mask_to_cnts(self, pred_mask_img):
        raw_pred_cnts = mask2contour(pred_mask_img, iterations=2)
        canvas = self._foolish_clean(pred_mask_img.shape, raw_pred_cnts)
        pred_cnts = mask2contour(canvas)

        return pred_cnts

    def _mask_creator(self, img):
        pred_mask = self.predict(img)
        pred_mask_img = (pred_mask * 255).astype(np.uint8)
        pred_cnts = self._pred_mask_to_cnts(pred_mask_img)
        return pred_cnts, pred_mask_img

    def metric(self, img_p: str, label_p: str, from_dir=False, show=True, name=None, show_mask=False, target=15) -> None:
        """true_positives = Correct objects
        false_positives = Missed objects
        false_negatives = Extra objects
        (PPV), Precision = Σ True positive / Σ Predicted condition positive
        Sensitivity, probability of detection = Σ True positive / Σ Condition positive
        If it is not form dir, we show the outcome directly"""
        if not name:
            name = get_name(img_p)
        if not from_dir:
            print(name)
        
        img = cv2.imread(img_p)
        mask_img = cv2.imread(label_p, 0)
        pred_cnts, pred_mask_img = self._mask_creator(img)
        label_cnts = mask2contour(mask_img)
        possible_right_cnts = self._overlap(label_cnts, pred_cnts, mask_img.shape)

        if show:
            fit_out = mask_visualization(img, possible_right_cnts, target=target)
            cv2.imwrite(os.path.join(dst, name+'_fit.jpg'), fit_out)
            pred_out = mask_visualization(img, pred_cnts, target=target)
            cv2.imwrite(os.path.join(dst, name+'_pred.jpg'), pred_out)
        if show_mask:
            cv2.imwrite(os.path.join(dst, name+'_mask.jpg'), pred_mask_img)

        true_objects = len(label_cnts)
        pred_objects = len(pred_cnts)
        true_positive = len(possible_right_cnts)
        false_positive = true_objects - true_positive
        false_negative = pred_objects - true_positive
        precision = true_positive / (true_positive + false_positive)
        sensitivity = true_positive / (true_positive + false_negative)

        values = [true_objects, pred_objects, true_positive, 
        false_positive, false_negative, precision, sensitivity]
        self.record[name] = values
        if not from_dir:
            print_dict = {}
            print_dict.update(dict(zip(self.metric_keys.split(', '), values)))
            print(print_dict)

    def _metric_summary(self):
        if len(self.record) == 0:
            print('No metric record')
            return
        df = pd.DataFrame.from_dict(self.record, orient='index', columns=self.metric_keys.split(', '))
        return df

    def metric_from_dir(self, img_dir_path, label_dir_path, dst):
        sys.stdout = open(os.path.join(dst, 'log.txt'), 'a')
        for img_p, label_p, name in path_matcher(img_dir_path, label_dir_path):
            self.metric(img_p, label_p, from_dir=True, name=name)
        df = self._metric_summary()
        print(df.describe())
        print(describe(np.array(self.cell_size)))
        sys.stdout.close()

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

def mask_visualization(img, cnts, color='cyan', expansion=0, target=None):
    color_dict = {'cyan':(255, 255, 0), 'green':(0, 255, 0), 'black': (0, 0, 0)}
    use_color = color_dict.get(color, (255, 255, 0))
    cur_img = img.copy()
    for c in cnts:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        draw_circle(cur_img, c, use_color, expansion, target=target)
    return cur_img

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

# 