from Unet_box.unet_tools import mask_to_cnts_watershed
from Unet_box.EOS_tools import middle, get_name, path_list_creator
from metric.metric_tools import save_dict, load_pickle
import cv2
import sys

def point_creator(mask_dir:str, name:str, dst:str, limit_pkl:str=None) -> None:
    """the mask_dir is the directory where all masks are;
    the name is the name of the created .pkl file;
    the dst is where the .pkl file will be saved;
    if the limit_pkl is provided, only the files in limit_pkl will be used, 
    this .pkl files should follow the format of metric_path_collector.py"""
    containor = {}
    masks = path_list_creator(mask_dir)
    if limit_pkl:
        target = load_pickle(limit_pkl)
        masks = [mask for mask in masks if get_name(mask) in target['labels']]

    for mask_p in masks:
        mask = cv2.imread(mask_p, 0)
        cnts = mask_to_cnts_watershed(mask, min_distance=14, for_real_mask=True)
        points = [middle(c) for c in cnts]
        containor[get_name(mask_p)] = points
    save_dict(containor, name, dst)

if __name__ == "__main__":
    mask_dir = sys.argv[1]
    name = sys.argv[2]
    dst = sys.argv[3]
    try:
        limit = sys.argv[4]
    except IndexError:
        limit = None
    point_creator(mask_dir, name, dst, limit_pkl=limit)