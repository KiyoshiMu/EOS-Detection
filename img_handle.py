from PIL import Image, ImageTk
# from Unet_box.use_model import Unet_predictor
class Img_server:
    def __init__(self, img_p):
        self.img = Image.open(img_p)
        self.size = self.img.shape[:2]

    def show(self, size=None):
        if size:
            img = self.img.resize(size)
        img_obj = ImageTk.PhotoImage(img)
        return img_obj

# class Img_operator:

#     def 
# def tk_image(path, w, h):
# 	img = Image.open(path)
# 	img = img.resize((w,h))
# 	storeobj = ImageTk.PhotoImage(img)
# 	return storeobj