import os
import cv2
import sys
import numpy as np
from collections import defaultdict
from Unet_box.unet_tools import mask_to_cnts_region, overlap, stats, show_result
from metric.metric_tools import load_pickle

def mask_findor(mask_dir, suffix='_mask'):
    for f_n in filter(lambda x:suffix in x, os.listdir(mask_dir)):
        yield os.path.splitext(f_n)[0], os.path.join(mask_dir, f_n)

def manual_thresh(mask_img, label_points, start=26):
    record = {}
    for threshold in range(start, 256, 2):
        pred_cnts = mask_to_cnts_region(mask_img, threshold=threshold)
        possible_right_cnts, label_num = overlap(pred_cnts, label_points=label_points)
        values = stats(label_num, pred_cnts, possible_right_cnts)
        record[threshold] = np.array(values)
    return record

def batch_roc(mask_dir, refer_pkl:str):
    summary = defaultdict(int)
    answers = load_pickle(refer_pkl)
    for name, fp in mask_findor(mask_dir):
        mask_img = cv2.imread(fp, 0)
        label_points = answers[name.replace('_mask', '')]
        record = manual_thresh(mask_img, label_points)
        for k, v in record.items():
            summary[k] = summary[k] + v
    return summary

if __name__ == "__main__":
    summary = batch_roc(sys.argv[1], sys.argv[2])
    columns = "true_objects, pred_objects, true_positive, false_positive, false_negative, precision, sensitivity".split(', ')
    show_result(summary, columns)