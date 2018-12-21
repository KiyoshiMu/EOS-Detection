import sys
import os
sys.path.append(os.path.abspath('.'))
from Unet_box.unet_tools import mask_to_cnts_watershed
from Unet_box.EOS_tools import middle, get_name, path_list_creator, point_detect
from metric.metric_tools import save_dict, load_pickle
import cv2

def point_creator(refer_dir:str, name:str, dst:str, point_label=True, direct_mask=False, limit_pkl:str=None) -> None:
    """the refer_dir is the directory where all refer are;
    the name is the name of the created .pkl file;
    the dst is where the .pkl file will be saved;
    if the limit_pkl is provided, only the files in limit_pkl will be used, 
    this .pkl files should follow the format of metric_path_collector.py"""
    containor = {}
    refers = path_list_creator(refer_dir)
    if limit_pkl:
        target = load_pickle(limit_pkl)
        refers = [refer for refer in refers if get_name(refer) in target['labels']]

    if point_label:
        for label_p in refers:
            points = point_detect(cv2.cvtColor(cv2.imread(label_p), cv2.COLOR_BGR2HSV))
            containor[get_name(label_p)] = points
    elif direct_mask:
        for mask_p in refers:
            containor[get_name(mask_p)] = mask_p
    else:
        for mask_p in refers:
            mask = cv2.imread(mask_p, 0)
            cnts = mask_to_cnts_watershed(mask, min_distance=14, for_real_mask=True)
            points = [middle(c) for c in cnts]
            containor[get_name(mask_p)] = points
    save_dict(containor, name, dst)

if __name__ == "__main__":
    refer_dir = sys.argv[1]
    name = sys.argv[2]
    dst = sys.argv[3]
    try:
        limit = sys.argv[4]
    except IndexError:
        limit = None
    point_creator(refer_dir, name, dst, limit_pkl=limit)