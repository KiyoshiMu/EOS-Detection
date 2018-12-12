import tkinter
from tkinter import ttk
from tkinter import filedialog
from img_handle import Img_server
from Unet_box.EOS_tools import path_list_creator
from itertools import cycle

class Menubar(ttk.Frame):
    """Builds a menu bar for the top of the main window"""
    def __init__(self, parent, *args, **kwargs): 
        ''' Constructor''' 
        ttk.Frame.__init__(self, parent, *args, **kwargs) 
        self.root = parent 
        self.init_menubar() 

    def init_menubar(self): 
        self.menubar = tkinter.Menu(self.root)

        self.menu_file = tkinter.Menu(self.menubar) # Creates a "File" menu 
        self.menu_file.add_command(label='Exit', command=self.on_exit) # Adds an option to the menu 
        self.menu_file.add_command(label='Open_dir', underline=0, command=self.open_file_dir) 
        self.menubar.add_cascade(menu=self.menu_file, label='File') # Adds File menu to the bar. Can also be used to create submenus.

        self.menu_help = tkinter.Menu(self.menubar) #Creates a "Help" menu 
        self.menu_help.add_command(label='Help', command=self.display_help) 
        self.menu_help.add_command(label='About', command=self.display_about) 
        self.menubar.add_cascade(menu=self.menu_help, label='Help') 
 
        self.root.config(menu=self.menubar)

    def on_exit(self): 
        '''Exits program''' 
        quit()

    def display_help(self): 
        '''Displays help document''' 
        pass 
 
    def display_about(self): 
        '''Displays info about program''' 
        pass

    def open_file_dir(self, event=None):
        """Open img_dir"""
        input_name = filedialog.askopenfilename(defaultextension=".jpg",
        filetypes=[("All Files", "*.*"), ("Images", "*.jpg")])
        if input_name:
            return input_name

class PictureWindow(tkinter.Canvas):

	def __init__(self, *args, **kwargs):
		tkinter.Canvas.__init__(self, *args, **kwargs)
		self.all_function_trigger()


	# def previous_image(self):
	# 	try:
	# 		pop = self.imagelist_p.pop()
	# 		self.show_image(pop)
	# 		self.imagelist.append(pop)
	# 	except:
	# 		pass
		

	# def next_image(self):
	# 	try:
	# 		pop = self.imagelist.pop()
		
	# 		self.show_image(pop)
	# 		self.imagelist_p.append(pop)
	# 	except EOFError as e:
	# 		pass
		

	def all_function_trigger(self):
		# self.create_buttons()
		self.window_settings()
		

	def window_settings(self):
		self['width']=self.winfo_screenwidth()
		self['height']=self.winfo_screenheight()
		

	# def create_buttons(self):
	# 	tkinter.Button(self, text=" > ", command=self.next_image).place(x=(self.winfo_screenwidth()/1.1),y=(self.winfo_screenheight()/2))
	# 	tkinter.Button(self, text=" < ", command=self.previous_image).place(x=20,y=(self.winfo_screenheight()/2))
	# 	self['bg']="white"


class GUI(ttk.Frame):
    """Main GUI class""" 
    def __init__(self, parent, *args, **kwargs): 
        ttk.Frame.__init__(self, parent, *args, **kwargs) 
        self.root = parent 
        self.init_gui() 
 
    def openwindow(self): 
        self.new_win = tkinter.Toplevel(self.root) # Set parent 
        PictureWindow(self.new_win)
 
    def init_gui(self): 
        self.root.title('Test') 
        self.grid_columnconfigure(0, weight=1) # Allows column to stretch upon resizing 
        self.grid_rowconfigure(0, weight=1) # Same with row 
        self.root.grid_columnconfigure(0, weight=1) 
        self.root.grid_rowconfigure(0, weight=1) 
        # self.root.option_add('*tearOffFALSE') # Disables ability to tear menu bar into own window 

        # Menu Bar
        self.menubar = Menubar(self.root)

        # Create Widgets 
        self.btn = ttk.Button(self, text='Open Window', command=self.openwindow)

        tkinter.Label(self.root, text='Select a image from disk').grid(row=0)
        button_in = tkinter.Button(self.root, text='Selecte a Image')
        button_in.grid(row=1, sticky='w',padx=2, pady=2)

        # Layout using grid 
        self.btn.grid(row=0, column=0, sticky='ew') 

        # Padding
        for child in self.winfo_children(): 
            child.grid_configure(padx=10, pady=5)
 
if __name__ == '__main__': 
    root = tkinter.Tk() 
    GUI(root)

    root.mainloop()
