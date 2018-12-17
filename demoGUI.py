from tkinter import ttk, Menu, Button, Toplevel, Tk, filedialog, Message
import os
import datetime
from functools import partial
from viewer import PictureWindow
from Unet_box.use_model import Unet_predictor
from Unet_box.EOS_tools import path_list_creator

class Menubar(ttk.Frame):
    """Builds a menu bar for the top of the main window"""
    def __init__(self, parent, *args, **kwargs):
        ''' Constructor'''
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_menubar()
        self.result_dir = None
        self.img_p_list = None
        self.is_assist = False

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
                top.destroy()
                self.result_dir = filedialog.askdirectory(title='Select a directory to save results ...', 
                mustexist=True)
                self.count_launch()
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

    def init_menubar(self):
        self.menubar = Menu(self.root)
        self.menu_file = Menu(self.menubar) # Creates a "File" menu
        self.menu_file.add_command(label='Exit', command=self.on_exit) # Adds an option to the menu
        self.menubar.add_cascade(menu=self.menu_file, label='File') # Adds File menu to the bar. Can also be used to create submenus.
        self.menu_file.add_command(label='Count from a Directory', command=self.count_from_dir)
        self.menu_file.add_command(label='Assist from a Directory', command=partial(self.count_from_dir, True))

        self.menu_help = Menu(self.menubar) #Creates a "Help" menu
        self.menu_help.add_command(label='Help', command=self.display_help)
        self.menu_help.add_command(label='About', command=self.display_about)
        self.menubar.add_cascade(menu=self.menu_help, label='Help')

        self.root.config(menu=self.menubar)

    def count_from_dir(self, assist=False):
        if assist:
            self.is_assist = True
        else:
            self.is_assist = False
        img_dir = filedialog.askdirectory(title='Select the directory where your slides are ...',
        mustexist=True)
        self.img_p_list = path_list_creator(img_dir)
        amount = len(self.img_p_list)
        time_needed = str(datetime.timedelta(seconds=amount * 30))
        self._message_prompt(title='The below is what we know ...',
        text=f'\n\
        The directory you selected is\n\
        {img_dir}\n\
        {amount} possible slices are there\n\
        Depending on your computer, about {time_needed} is needed.\n\
        If it is OK, please PRESS "Continue" and then\n\
        select a directory to save results.',
                            normal=False)

    def count_launch(self, visulaize=True):
        assert os.path.isdir(self.result_dir), 'the director where to save results does not exist'
        assert isinstance(self.img_p_list, list), 'the director where to read sildes is not ready'
        if visulaize:
            visualize_dst = os.path.join(self.result_dir, 'visulization')
            os.makedirs(visualize_dst, exist_ok=True)
        else:
            visualize_dst = None
        assistance = self.is_assist
        actor.predict_from_imgs(self.img_p_list, self.result_dir, visualize_dst, assistance=assistance)
        self._message_prompt("Congratulation", f'Done! Please Cheak {self.result_dir}')

class GUI(ttk.Frame):
    """Main GUI class"""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def openwindow(self):
        self.new_win = Toplevel(self.root) # Set parent
        pic = PictureWindow(self.new_win, actor=actor)
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
        
        # Menu Bar
        self.menubar = Menubar(self.root)
        
        # Create Widgets
        self.btn = ttk.Button(self, text='Open Viewer', command=self.openwindow)

        # Layout using grid
        self.btn.grid(row=0, column=0, sticky='ew')

        # Padding
        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)


if __name__ == '__main__':
    global actor
    # actor = Unet_predictor('./Unet_box/not_fs_second.h5')
    actor = Unet_predictor('./Unet_box/Demo.h5')
    root = Tk()
    GUI(root)
    root.mainloop()