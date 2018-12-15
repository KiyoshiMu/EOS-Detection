# the show_image function and the framework are from
# Suraj Singh Admin S.S.B Group surajsinghbisht054@gmail.com http://bitforestinfo.blogspot.in/
# Thanks!

from tkinter import Canvas, Tk, Button, filedialog, Label
from PIL import Image, ImageTk
from copy import copy
from Unet_box.EOS_tools import path_list_creator, get_name
from os.path import join, dirname
import os
import cv2
import numpy as np
import compileall
from functools import partial
from Unet_box.use_model import Unet_predictor
from collections import deque

def ask_path():
    global dst
    path = filedialog.askdirectory()
    dst = join(dirname(path), 'temp')
    os.makedirs(dst, exist_ok=True)
    return path

def creat_path_list():
    path = ask_path()
    path_list = path_list_creator(path)
    return path_list

# Creating Canvas Widget
class PictureWindow(Canvas):
    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)

        self.type_to_idx = {'origin':0, 'detection':1, 'mask':2}
        self.type_to_suffix = {'detection':'_pred', 'mask':'_mask'}
        self.path_list = creat_path_list()
        self.cur = 0
        self.cur_type = deque(maxlen=2)

        self.cache = {}
        self.length = len(self.path_list)
        self.loc = self.path_list[self.cur]
        self.ID = get_name(self.loc)

        self.height = self.winfo_screenheight()
        self.width = self.winfo_screenwidth()
        self.all_function_trigger()
        self.img_switcher()

    def read_img(self, path, update_predicted=False):
        print(path)
        if path in self.cache and not update_predicted:
            return self.cache[path]
  
        self.cache[path] = [None] * 3
        idx = self.type_to_idx['origin']
        self.cache[path][idx] = cv2.imread(path)

        for img_type in self.type_to_suffix.keys():
            self._search(path, img_type)
            
        return self.cache[path]

    def _search(self, path, img_type:str):
        img_path = join(dst, f'{get_name(path)}{self.type_to_suffix[img_type]}.jpg')
        if os.path.isfile(img_path):
            idx = self.type_to_idx[img_type]
            self.cache[path][idx] = cv2.imread(img_path)
            print(f'find {img_type}')
        else:
            print(f'not find {img_type}')

    def show_image(self, raw_img):
        w, h = self.width, self.height
        resize_img = cv2.resize(raw_img, (w, h))[...,::-1]
        img = ImageTk.PhotoImage(Image.fromarray(resize_img))
        self.delete(self.find_withtag("img"))
        self.allready = self.create_image(w/2, h/2, image=img, anchor='center', tag="img")
        self.image = img
        self.master.title("EOS ({})".format(self.ID))
        
    def img_switcher(self, update_predicted=False):
        cache = self.read_img(self.loc, update_predicted=update_predicted)
        self._button_disable()
        if update_predicted == True:
            print('prediction completed')
            self._img_and_button(cache, 'detection', next_img_type='origin')
        else:
            if isinstance(cache[1], np.ndarray):
                self._img_and_button(cache, 'origin', next_img_type='detection')
            else:
                self._img_and_button(cache, 'origin')
                return
            if isinstance(cache[2], np.ndarray):
                self._button_update(cache, 'mask', 'mask')

    def _button_update(self, cache, img_type, button_type):
        self._button_disable(button_type=button_type)
        self._button_enable(f"show {img_type}", 
        partial(self._img_and_button, cache, img_type, next_img_type=self.cur_type[0]), 
        button_type=button_type)

    def _show_something(self, cache, img_type:str):
        """self.type_to_idx = {'origin':0, 'detection':1, 'mask':2}"""
        self.cur_type.append(img_type)
        idx = self.type_to_idx[img_type]
        print(f'show {img_type}')
        self.show_image(cache[idx])

    def _img_and_button(self, cache, img_type:str, next_img_type=None, dis_button=None):
        self._show_something(cache, img_type)
        self._button_handel(cache, next_img_type, dis_button)

    def _button_handel(self, cache, next_img_type=None, dis_button=None):
        if dis_button:
            self._button_disable(button_type=dis_button)
        if next_img_type:
            print(self.cur_type)
            button_type = 'mask' if self.cur_type[1] == 'mask' else 'origin'
            self._button_update(cache, next_img_type, button_type)

    def _button_disable(self, button_type=None):
        iters = self.type_to_button.keys() if not button_type else [button_type]
        for button_type in iters:
            try:
                self.type_to_button[button_type].destroy()
                print(f'disable {button_type} button')
            except:
                print(f'{button_type} button is disabled')
        
    def _button_enable(self, text, func, button_type):
        rely = 0.4 if button_type=='mask' else 0.5
        print(f'enable {button_type} button')
        self.type_to_button[button_type] = Button(self, text=text, command=func)
        self.type_to_button[button_type].place(relx=0.5, rely=rely, anchor='center')

    def _loc_updator(self):
        if self.cur >= self.length:
            self.cur = 0
        elif self.cur < 0:
            self.cur = self.length - 1
        self.loc = self.path_list[self.cur]
        self.ID = get_name(self.loc)

    def previous_image(self):
        self.cur -= 1
        self._loc_updator()
        self.img_switcher()

    def next_image(self):
        self.cur += 1
        self._loc_updator()
        self.img_switcher()
        
    def all_function_trigger(self):
        self.buttons_init()

    def buttons_init(self):
        Button(self, text=" > ", command=self.next_image).place(relx=0.8, rely=0.5, anchor='center')
        Button(self, text=" < ", command=self.previous_image).place(relx=0.2, rely=0.5, anchor='center')
        Button(self, text=" Detect", command=self.img_predict).place(relx=0.5, rely=0.8, anchor='center')
        self.center_button = None
        self.lower_center_button = None
        self.type_to_button = {'origin':self.center_button, 
        'mask':self.lower_center_button}
        
    def img_predict(self):
        raw_img = self.cache[self.loc][0]
        actor.predict_from_img(raw_img, self.ID, dst)
        self.img_switcher(update_predicted=True)


# def link():
#     os.makedirs(dst, exist_ok=True)
#     actor.predict_from_dir(path, dst)
# Main Function

def main():
    global actor
    actor = Unet_predictor('./Unet_box/not_fs_second.h5')
    # Creating Window
    root = Tk(className="EOS Predictor")
    PictureWindow(root).pack(expand="yes",fill="both")
    # root.resizable(width=0,height=0)
    root.mainloop()
    
if __name__ == '__main__':
    main()
