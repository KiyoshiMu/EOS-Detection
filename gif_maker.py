from PIL import Image, ImageDraw, ImageFont
import os
import sys
from collections import defaultdict
from EOS_tools import get_name, path_list_creator
from os.path import join

def make_gif_from_dir(metric_dir_path, dst):
    maid = defaultdict(dict)
    path_list = path_list_creator(metric_dir_path)
    for fp in path_list:
        name = get_name(fp)
        if '_pred' in fp:
            maid[name]['predicted'] = fp
        elif '_label' in fp:
            maid[name]['labeled'] = fp
        # elif '_fit' in fp:
        #     maid[name]['fit'] = fp
    for f_name, fp_dict in maid.items():
        if len(fp_dict) == 2:
            out_p = join(dst, f_name+'.gif')
            make_gif(fp_dict, out_p)

def img_makeup(img, text, f_size=100):
    fnt = ImageFont.truetype('Aaargh.ttf', f_size)
    draw = ImageDraw.Draw(img)
    text_w, text_h = draw.textsize(text, fnt)
    w, h = img.size
    x = int((w - text_w) / 2)
    y = int((h- text_h) / 2)
    draw.text((x, y), text, font=fnt)
    img = img.resize((int(w/2), int(h/2)))
    return img

def make_gif(fp_dict, out_p):
    images = []
    for element in ('predicted', 'labeled'):
        fp = fp_dict[element]
        img = Image.open(fp)
        img_up = img_makeup(img, element.upper())
        images.append(img_up)
        print(fp)

    images[0].save(out_p,
               save_all=True,
               append_images=images[1:],
               duration=666,
               loop=0)

if __name__ == "__main__":
    start = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    make_gif_from_dir(start, dst)