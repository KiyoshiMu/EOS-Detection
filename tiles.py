from EOS_tools import path_matcher
import cv2
import math
import os

def crop(img, start_point, size):
    w, h = size
    x, y = start_point
    return img[y:y+h, x:x+w]

def tiling(img_p, size=(256, 256)):
    w, h = size
    img = cv2.imread(img_p)
    
    real_h, real_w, _ = img.shape
    w_count = math.ceil(real_w / w)
    h_count = math.ceil(real_h / h)
    w_overlap = math.ceil((w_count * w - real_w) / (w_count - 1))
    h_overlap = math.ceil((h_count * h - real_h) / (h_count - 1))
    w_stride = w - w_overlap
    h_stride = h - h_overlap
    x = w_stride * (-1)

    for w_idx in range(w_count):
        y = h_stride * (-1)
        x += w_stride
        for h_idx in range(h_count):
            y += h_stride
            tile = crop(img, (x, y), size)
            yield w_idx, h_idx, tile

def creat_tiles(dir_img, dir_label, dst):
    img_dst = os.path.join(dst, 'raw_imgs')
    label_dst = os.path.join(dst, 'labels')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)
    for img_p, label_p, name in path_matcher(dir_img, dir_label):
        for w_idx, h_idx, tile in tiling(img_p):
            f_name = '{}_{}-{}.tif'.format(name, w_idx, h_idx)
            cv2.imwrite(os.path.join(img_dst, f_name), tile)
        for w_idx, h_idx, tile in tiling(label_p):
            f_name = '{}_{}-{}.png'.format(name, w_idx, h_idx)
            cv2.imwrite(os.path.join(label_dst, f_name), tile)