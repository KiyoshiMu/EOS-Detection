from Unet_box.unet_tools import mask_to_cnts_watershed
from Unet_box.EOS_tools import middle, get_name, path_list_creator
from learning_test import save_dict
import cv2
import sys

def point_creator(dir_p, name, dst):
    containor = {}
    for mask_p in path_list_creator(dir_p):
        mask = cv2.imread(mask_p, 0)
        cnts = mask_to_cnts_watershed(mask, min_distance=14, for_real_mask=True)
        points = [middle(c) for c in cnts]
        containor[get_name(mask_p)] = points
    save_dict(containor, name, dst)

if __name__ == "__main__":
    mask_dir = sys.argv[1]
    name = sys.argv[2]
    dst = sys.argv[3]
    point_creator(mask_dir, name, dst)