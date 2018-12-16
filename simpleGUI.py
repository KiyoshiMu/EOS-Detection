# edited from "Skeleton template for a Tkinter GUI" https://github.com/grassfedfarmboi/tkinter_template
from tkinter import ttk, Menu, Button, Toplevel, Tk, filedialog, Message
from PIL import Image, ImageTk
import abc
from tkinter import Canvas, Tk, Button, filedialog, Label
import os, cv2
import numpy as np
from Unet_box.EOS_tools import path_list_creator, get_name
from functools import partial
from os.path import join, dirname
from collections import deque
import datetime

def ask_path(title):
        return filedialog.askdirectory(title=title, mustexist=True)

class Menubar(ttk.Frame):
    """Builds a menu bar for the top of the main window"""
    def __init__(self, parent, *args, **kwargs):
        ''' Constructor'''
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_menubar()
        self.result_dir = None
        self.img_p_list = None

    def on_exit(self):
        '''Exits program'''
        quit()

    def _message_prompt(self, title, text, width=400, height=300, normal=True):
        top = Toplevel(self.root)
        top.geometry("%dx%d%+d%+d" % (width, height, 250, 125))
        top.title(title)
        help_message = text
        msg = Message(top, text=help_message, justify='left', width=width-100)
        msg.place(relx=0.5, rely=0.4, anchor='center')
        if normal:
            Button(top, text='Dismiss', command=top.destroy).place(relx=0.5, rely=0.9, anchor='center')
        else:
            def couple_func():
                self.result_dir = filedialog.askdirectory(title='Select a directory to save results ...', 
                mustexist=True)
                self.count_launch()
                top.destroy()
            Button(top, text='Continue', command=couple_func).place(relx=0.5, rely=0.8, anchor='center')
            Button(top, text='Cancel', command=top.destroy).place(relx=0.5, rely=0.9, anchor='center')
        
    def display_help(self):
        '''Displays help document'''
        title = 'Breif Help'
        text = '\n\
        To See Each Image, Press "Open Viewer".\n\
        To count cells from images in a directory,\n\
        Press "Count from a Directory" which is In FILE on the Menubar.'
        self._message_prompt(title, text)

    def display_about(self):
        '''Displays info about program'''
        title = 'About'
        text = '\n\
        FROM SYSU\nOn TEST PHASE\nTHANKS!\n\
        https://github.com/Moo-YewTsing/EOS-Detection.git'
        self._message_prompt(title, text)

    def count_from_dir(self):
        img_dir = filedialog.askdirectory(title='Select the directory where your slides are in ...',
        mustexist=True)
        self.img_p_list = path_list_creator(img_dir)
        amount = len(self.img_p_list)
        time_needed = str(datetime.timedelta(seconds=amount * 30))
        self._message_prompt(title='The below is what we know ...',
        text=f'\n\
        The directory you selected is\n\
        {img_dir}\n\
        {amount} possible slices are there\n\
        About {time_needed} is needed.\n\
        If it is OK, please PRESS "Continue" and then\n\
        select a directory to save results.',
                            normal=False)

    def count_launch(self):
        assert os.path.isdir(self.result_dir), 'the director where to save results does not exist'
        assert isinstance(self.img_p_list, list), 'the director where to read sildes is not ready'
        print('ok')

    def init_menubar(self):
        self.menubar = Menu(self.root)
        self.menu_file = Menu(self.menubar) # Creates a "File" menu
        self.menu_file.add_command(label='Exit', command=self.on_exit) # Adds an option to the menu
        self.menubar.add_cascade(menu=self.menu_file, label='File') # Adds File menu to the bar. Can also be used to create submenus.
        self.menu_file.add_command(label='Count from a Directory', command=self.count_from_dir)

        self.menu_help = Menu(self.menubar) #Creates a "Help" menu
        self.menu_help.add_command(label='Help', command=self.display_help)
        self.menu_help.add_command(label='About', command=self.display_about)
        self.menubar.add_cascade(menu=self.menu_help, label='Help')

        self.root.config(menu=self.menubar)


class GUI(ttk.Frame):
    """Main GUI class"""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def openwindow(self):
        self.new_win = Toplevel(self.root) # Set parent
        pic = PictureWindow(self.new_win)
        pic.pack(expand="yes",fill="both")

    def init_gui(self):
        self.root.title('EOS Detector Demo')
        self.root.geometry("600x400")
        self.grid(column=0, row=0, sticky='nsew')
        self.grid_columnconfigure(0, weight=1) # Allows column to stretch upon resizing
        self.grid_rowconfigure(0, weight=1) # Same with row
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.option_add('*tearOff', 'FALSE') # Disables ability to tear menu bar into own window
        
        self.menubar = Menubar(self.root)
        
        self.btn = ttk.Button(self, text='Open Viewer', command=self.openwindow)

        self.btn.grid(row=0, column=0, sticky='ew')

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

def _ask_path():
    global dst
    path = filedialog.askdirectory()
    dst = join(dirname(path), 'temp')
    os.makedirs(dst, exist_ok=True)
    return path

def creat_path_list():
    path = _ask_path()
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
        self.master.title(f"EOS ({self.ID})")
        
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
        self.cur_type.append(img_type)
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
        self.origin_button = None
        self.detection_button = None
        self.mask_button = None
        self.type_to_button = {'origin':[self.origin_button, 0.5], 
        'detection':[self.detection_button, 0.4], 
        'mask':[self.mask_button, 0.3]}
		
if __name__ == '__main__':
    root = Tk()
    GUI(root)
    root.mainloop()
