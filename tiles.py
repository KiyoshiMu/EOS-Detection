from EOS_tools import path_matcher
import cv2
import math
import os
import sys

def crop(img, start_point, size):
    w, h = size
    x, y = start_point
    return img[y:y+h, x:x+w]

def line_calculator(line, real_line):
    line_count = math.ceil(real_line / line)
    line_overlap = math.ceil((line_count * line - real_line) / (line_count - 1))
    line_stride = line - line_overlap
    line_out = line_stride * (line_count - 1) + line
    return line_stride, line_count, line_out

def tile_simulation(img, size):
    w, h = size
    real_h, real_w, _ = img.shape
    w_stride, w_count, w_out = line_calculator(w, real_w)
    h_stride, h_count, h_out = line_calculator(h, real_h)
    return w_stride, w_count, w_out, h_stride, h_count, h_out

def tiling(img, size=(256, 256)):
    w_stride, w_count, _, h_stride, h_count, _ = tile_simulation(img, size)
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
        img = cv2.imread(img_p)
        label = cv2.imread(label_p)
        for w_idx, h_idx, tile in tiling(img):
            f_name = '{}_{}-{}.tif'.format(name, w_idx, h_idx)
            cv2.imwrite(os.path.join(img_dst, f_name), tile)
        for w_idx, h_idx, tile in tiling(label):
            f_name = '{}_{}-{}.png'.format(name, w_idx, h_idx)
            cv2.imwrite(os.path.join(label_dst, f_name), tile)

if __name__ == "__main__":
    dir_img = sys.argv[1]
    dir_label = sys.argv[2]
    dst = sys.argv[3]
    creat_tiles(dir_img, dir_label, dst)