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
from functools import partial
from Unet_box.use_model import Unet_predictor

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

        self.path_list = creat_path_list()
        self.cur = 0
        self.pred_button = None
        self.cache = {}
        self.length = len(self.path_list)
        self.loc = self.path_list[self.cur]
        self.ID = get_name(self.loc)

        self.height = self.winfo_screenheight()
        self.width = self.winfo_screenwidth()
        self.all_function_trigger()

        self.img_switcher(self.loc)

    def read_img(self, path, update_predicted=False):
        if path in self.cache and not update_predicted:
            return self.cache[path]
  
        self.cache[path] = [cv2.imread(path), None]
        if update_predicted:
            pred_path = join(dst, get_name(path)+'_pred.jpg')
            self.cache[path][1] = cv2.imread(pred_path)
        # except:
        #     print('Not yet predicted')

        return self.cache[path]

    def show_image(self, raw_img):
        w, h = self.width, self.height
        resize_img = cv2.resize(raw_img, (w, h))
        img = ImageTk.PhotoImage(Image.fromarray(resize_img))
        self.delete(self.find_withtag("img"))
        self.allready = self.create_image(w/2, h/2, image=img, anchor='center', tag="img")
        self.image = img
        self.master.title("Image Viewer ({})".format(self.ID))
        
    def img_switcher(self, path, update_predicted=False):
        cache = self.read_img(path, update_predicted=update_predicted)
        if update_predicted:
            pred_img = cache[1]
            self.show_image(pred_img)        
        else:
            raw_img = cache[0]
            self.show_image(raw_img)
            if cache[1]:
                self._button_enable("show predicted", partial(self.show_image, cache[1]))
            elif self.pred_button:
                self._button_disable()

    def _button_disable(self):
        try:
            self.pred_button.destroy()
        except:
            print('button disable')
        
    def _button_enable(self, text, func):
        self.pred_button = Button(self, text=text, command=func)
        self.pred_button.place(relx=0.5, rely=0.5, anchor='center')

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
        self.img_switcher(self.loc)

    def next_image(self):
        self.cur += 1
        self._loc_updator()
        self.img_switcher(self.loc)
        
    def all_function_trigger(self):
        self.create_buttons()

    def create_buttons(self):
        Button(self, text=" > ", command=self.next_image).place(relx=0.8, rely=0.5, anchor='center')
        Button(self, text=" < ", command=self.previous_image).place(relx=0.2, rely=0.5, anchor='center')
        Button(self, text=" Predict", command=self.img_predict).place(relx=0.5, rely=0.8, anchor='center')
        self['bg']="white"
        
    def img_predict(self):
        raw_img = self.cache[self.loc][0]
        actor.predict_from_img(raw_img, self.ID, dst)
        self.img_switcher(self.loc, update_predicted=True)
# def link():
#     os.makedirs(dst, exist_ok=True)
#     actor.predict_from_dir(path, dst)
# Main Function

def main():
    global actor
    actor = Unet_predictor('./Unet_box/not_fs_second.h5')
    # Creating Window
    root = Tk(className=" Image Viewer")
    PictureWindow(root).pack(expand="yes",fill="both")
    # root.resizable(width=0,height=0)
    root.mainloop()
    
if __name__ == '__main__':
    main()
