import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from bisect import bisect_left
from scipy.stats import describe
from math import pi, sqrt
from datetime import datetime
from Unet_box.Unet import UNET
from Unet_box.tile_creator import tiling, tile_simulation
from Unet_box.EOS_tools import path_list_creator, path_matcher, get_name
from Unet_box.unet_tools import mark_text, overlap, pred_mask_to_cnts, mask_visualization, stats, show_result

class Unet_predictor:

    def __init__(self, model_p):
        self.model = UNET()
        self.model.load_weights(model_p)
        self.size = (256, 256)
        self.path = model_p
        self.result = {}
        self.metric_record = {}
        self.metric_keys = "Number_of_true_objects, Number_of_predicted_objects, true_positive, false_positive, false_negative, Precision, Sensitivity"

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

    def predict_from_img(self, img, ID, visualize_dst=None, show_mask=True, 
    method='circle', target=15, mark_num=False, assistance=False):
        pred_cnts, pred_mask_img = self._mask_creator(img)
        # self.result[ID] = [cv2.contourArea(c) for c in pred_cnts]
        self.result[ID] = len(pred_cnts)
        if visualize_dst:
            pred_out = mask_visualization(img, pred_cnts, method=method, target=target)
            if mark_num:
                mark_text(pred_out, f'{self.result[ID]}')
            cv2.imwrite(os.path.join(visualize_dst, ID+'_pred.jpg'), pred_out)
            if assistance:
                assist = mask_visualization(img, pred_cnts, method='circle', target=5, color='green', thickness=-1)
                cv2.imwrite(os.path.join(visualize_dst, ID+'_help.jpg'), assist)
            if show_mask:
                cv2.imwrite(os.path.join(visualize_dst, ID+'_mask.jpg'), pred_mask_img)

    def predict_from_imgs(self, img_p_list, result_dst, visualize_dst=None, show_mask=False, 
    target=15, assistance=False):
        # self.result.clear()
        for img_p in img_p_list:
            img = cv2.imread(img_p)
            ID = get_name(img_p)
            if ID in self.result:
                continue
            self.predict_from_img(img, ID, visualize_dst=visualize_dst,
            show_mask=show_mask, target=target, assistance=assistance)
        show_result(self.result, ['Counts'], title='Count', dst=result_dst)

    def _mask_creator(self, img):
        pred_mask = self.predict(img)
        pred_mask_img = (pred_mask * 255).astype(np.uint8)
        pred_cnts = pred_mask_to_cnts(pred_mask_img)
        return pred_cnts, pred_mask_img

    def metric(self, img_p: str, label_p:str=None, show_info=False, 
        show_img=True, dst='.', name=None, show_mask=False, target=15, label_points=None) -> None:
        """true_positives = Correct objects
        false_positives = Missed objects
        false_negatives = Extra objects
        (PPV), Precision = Σ True positive / Σ Predicted condition positive
        Sensitivity, probability of detection = Σ True positive / Σ Condition positive
        If it is not form dir, we show the outcome directly"""
        if not name:
            name = get_name(img_p)
        if show_info:
            print(name)
        
        img = cv2.imread(img_p)
        pred_cnts, pred_mask_img = self._mask_creator(img)
        mask_img = None
        if isinstance(label_points, list):
            label_points = label_points
        elif label_p:
            mask_img = cv2.imread(label_p, 0)
        possible_right_cnts, label_num = overlap(pred_cnts, mask_img=mask_img, label_points=label_points)

        if show_img:
            fit_out = mask_visualization(img, possible_right_cnts, target=target)
            cv2.imwrite(os.path.join(dst, name+'_fit.jpg'), fit_out)
            pred_out = mask_visualization(img, pred_cnts, target=target)
            cv2.imwrite(os.path.join(dst, name+'_pred.jpg'), pred_out)
        if show_mask:
            cv2.imwrite(os.path.join(dst, name+'_mask.jpg'), pred_mask_img)

        values = stats(label_num, pred_cnts, possible_right_cnts)
        self.metric_record[name] = values
        if show_info:
            print_dict = {}
            print_dict.update(dict(zip(self.metric_keys.split(', '), values)))
            print(print_dict)

    def metric_from_dir(self, img_dir_path, label_dir_path, dst):
        for img_p, label_p, name in path_matcher(img_dir_path, label_dir_path):
            self.metric(img_p, label_p, show_info=False, dst=dst, name=name)
        show_result(self.metric_record, self.metric_keys.split(', '), 
        title='Metric', dst=dst)

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
        img_p_list = path_list_creator(img_dir)
        actor.predict_from_imgs(img_p_list, dst)