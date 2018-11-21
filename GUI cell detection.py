from tkinter import Tk, Menu, Button, Label
from tkinter import filedialog
def open_file(event=None):
    input_name = filedialog.askopenfilename(defaultextension=".jpg",
filetypes=[("All Files", "*.*"), ("Images", "*.jpg")])
    if input_name:
        print(input_name)

parent = Tk(screenName='EOS Detection')
menu_bar = Menu(parent)

file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Open', underline=0, command=open_file)
about_menu = Menu(menu_bar, tearoff=0)

menu_bar.add_cascade(label='File', menu=file_menu)
menu_bar.add_cascade(label='About', menu=about_menu)
parent.config(menu=menu_bar)


Label(parent, text='Select a image from disk').grid(row=0)
button_in = Button(parent, text='Selecte a Image')
button_in.grid(row=1, sticky='w',padx=2, pady=2)
button_in.bind('<Button-1>', open_file)
parent.mainloop()