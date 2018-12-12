# the show_image function and the framework are from
# Suraj Singh Admin S.S.B Group surajsinghbisht054@gmail.com http://bitforestinfo.blogspot.in/
# Thanks!

from tkinter import Canvas, Tk, Button, filedialog, Label
from PIL import Image, ImageTk
from copy import copy
from Unet_box.EOS_tools import path_list_creator
from os.path import join, dirname
import os
from Unet_box.use_model import Unet_predictor

def tk_image(path,w,h):
    img = Image.open(path)
    img = img.resize((w,h))
    storeobj = ImageTk.PhotoImage(img)
    return storeobj

def ask_path():
    global path
    path = filedialog.askdirectory()
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
        self.predicted = False
        self.length = len(self.path_list)
        self.all_function_trigger()
        self.w = self.winfo_screenwidth()
        self.h = self.winfo_screenheight()
        self.show_image(self.path_list[self.cur])

    def show_image(self, path):
        if not self.predicted:
            w, h = self.w, self.h
            img = tk_image(path, w, h)
            self.delete(self.find_withtag("img"))
            self.allready = self.create_image(w/2, h/2,image=img, anchor='center', tag="img")
            self.image = img
            self.master.title("Image Viewer ({})".format(path))
        
        
    def _path_selector(self):
        if self.cur >= self.length:
            self.cur = 0
        elif self.cur < 0:
            self.cur = self.length - 1
        return self.path_list[self.cur]

    def previous_image(self):
        self.cur -= 1
        path = self._path_selector()
        self.show_image(path)

    def next_image(self):
        self.cur += 1
        path = self._path_selector()
        self.show_image(path)
        
    def all_function_trigger(self):
        self.create_buttons()
        self.window_settings()
        
    def window_settings(self):
        self['width'] = self.winfo_screenwidth()
        self['height'] = self.winfo_screenheight()
        
    def create_buttons(self):
        Button(self, text=" > ", command=self.next_image).place(x=(self.winfo_screenwidth()/1.1),y=(self.winfo_screenheight()/2))
        Button(self, text=" < ", command=self.previous_image).place(x=20,y=(self.winfo_screenheight()/2))
        Button(self, text=" Predict ", command=link).place(relx=0.5, rely=0.8, anchor='center')
        self['bg']="white"
        
def link():
    dst = join(dirname(path), 'temp')
    actor = Unet_predictor('./Unet_box/not_fs_second.h5')
    os.makedirs(dst, exist_ok=True)
    actor.predict_from_dir(path, dst)
# Main Function

def main():
    # Creating Window
    root = Tk(className=" Image Viewer")
    # Creating Canvas Widget
    PictureWindow(root).pack(expand="yes",fill="both")
    # Not Resizable
    root.resizable(width=0,height=0)
    # Window Mainloop
    root.mainloop()
    

# Main Function Trigger
if __name__ == '__main__':
    main()