from Unet import UNET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
from tiles import tiling, tile_simulation
from EOS_tools import path_list_creator, path_matcher

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
            pred_cnts = mask2contour(pred_mask_img)

            ID = os.path.basename(img_p).split('.')[0]
            self.result[ID] = len(pred_cnts)

            if visualize_dst:
                img_out = mask_visualization(img, pred_cnts)
                cv2.imwrite(os.path.join(visualize_dst, ID+'.jpg'), img_out)
        self.show_result()

    def _precision_at(self, threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    def metric(self, img_p, label_p, name=None):
        sys.stdout = open('log.txt', 'a')
        if name:
            print(name)

        img = cv2.imread(img_p)
        mask_img = cv2.imread(label_p, 0)
        pred_mask = self.predict(img)
        pred_mask_img = (pred_mask * 255).astype(np.uint8)
        pred_cnts = mask2contour(pred_mask_img)
        label_cnts = mask2contour(mask_img)

        true_objects = len(label_cnts)
        pred_objects = len(pred_cnts)
        print("Number of true objects:", true_objects)
        print("Number of predicted objects:", pred_objects)
        # Compute intersection between all objects
        intersection = np.histogram2d(mask_img.flatten(), pred_mask_img.flatten(),
        bins=(true_objects, pred_objects))[0]
        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(mask_img, bins = true_objects)[0]
        area_pred = np.histogram(pred_mask_img, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)
        # Compute union
        union = area_true + area_pred - intersection
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9
        # Compute the intersection over union
        iou = intersection / union

        # Loop over IoU thresholds
        prec = []
        print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = self._precision_at(t, iou)
            p = tp / (tp + fp + fn)
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

        sys.stdout.close()

    def metric_from_dir(self, img_dir_path, label_dir_path):
        for img_p, label_p, name in path_matcher(img_dir_path, label_dir_path):
            self.metric(img_p, label_p, name)
            
def mask2contour(mask_img, iterations=2):
    thresh = 255 - cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5),np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=iterations)
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