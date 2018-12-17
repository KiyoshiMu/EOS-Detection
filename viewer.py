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
from functools import partial
from Unet_box.use_model import Unet_predictor
from collections import deque

# Creating Canvas Widget
class PictureWindow(Canvas):
    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        self.actor = kwargs.get('actor')
        assert self.actor is not None, 'No Modle Is Loaded!'
        self.type_to_idx = {'origin':0, 'detection':1, 'mask':2}
        self.type_to_suffix = {'detection':'_pred', 'mask':'_mask'}
        # self.cur_type = deque(maxlen=2)
        self._temp_init()
        self.cache = {}
        
        self.height = self.winfo_screenheight()
        self.width = self.winfo_screenwidth()
        self.all_function_trigger()
        self.img_switcher()

    def _temp_init(self):
        img_dir = filedialog.askdirectory(title='Select the directory where your slides are in ...', 
        mustexist=True)
        self.dst = join(dirname(img_dir), 'temp')
        os.makedirs(self.dst, exist_ok=True)
        self.path_list = path_list_creator(img_dir)
        self.cur = 0
        self.length = len(self.path_list)
        self.loc = self.path_list[self.cur]
        self.ID = get_name(self.loc)

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
        img_path = join(self.dst, f'{get_name(path)}{self.type_to_suffix[img_type]}.jpg')
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
            self._show_something(cache, 'detection')
            self._button_update(cache, 'origin')
            self._button_update(cache, 'detection')
            self._button_update(cache, 'mask')
        else:
            self._show_something(cache, 'origin')
            if isinstance(cache[1], np.ndarray):
                self._button_update(cache, 'origin')
                self._button_update(cache, 'detection')
            if isinstance(cache[2], np.ndarray):
                self._button_update(cache, 'mask')

    def _button_update(self, cache, next_img_type):
        self._button_disable(button_type=next_img_type)
        self._button_enable(f"show {next_img_type}", 
        partial(self._show_something, cache, next_img_type), 
        button_type=next_img_type)

    def _button_enable(self, text, func, button_type):
        rely = self.type_to_button[button_type][1]
        print(f'enable {button_type} button')
        self.type_to_button[button_type][0] = Button(self, text=text, command=func)
        self.type_to_button[button_type][0].place(relx=0.5, rely=rely, anchor='center')

    def _show_something(self, cache, img_type:str):
        """self.type_to_idx = {'origin':0, 'detection':1, 'mask':2}"""
        # self.cur_type.append(img_type)
        idx = self.type_to_idx[img_type]
        print(f'show {img_type}')
        self.show_image(cache[idx])

    def _button_disable(self, button_type=None):
        iters = self.type_to_button.keys() if not button_type else [button_type]
        for button_type in iters:
            try:
                self.type_to_button[button_type][0].destroy()
                print(f'disable {button_type} button')
            except:
                print(f'{button_type} button is disabled')
        
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
        self.origin_button = None
        self.detection_button = None
        self.mask_button = None
        self.type_to_button = {'origin':[self.origin_button, 0.5], 
        'detection':[self.detection_button, 0.4], 
        'mask':[self.mask_button, 0.3]}
        
    def img_predict(self):
        raw_img = self.cache[self.loc][0]
        self.actor.predict_from_img(raw_img, self.ID, visualize_dst=self.dst, mark_num=True)
        self.img_switcher(update_predicted=True)

def main():
    actor = Unet_predictor('./Unet_box/not_fs_second.h5')
    # Creating Window
    root = Tk(className="EOS Predictor")
    PictureWindow(root, actor=actor).pack(expand="yes",fill="both")
    # root.resizable(width=0,height=0)
    root.mainloop()
    
if __name__ == '__main__':
    main()
