from Unet_box.point_label_creator import point_label_creator
from Unet_box.point_to_maskor import point2mask
from Unet_box.tile_creator import creat_tiles
from os.path import join
from sys import argv

def from_point_to_tiles(imgs_p, labels_p, dst):
    point_labels_p = join(dst, 'point_labels')
    point_label_creator(imgs_p, labels_p, point_labels_p)

    masks_p = join(dst, 'created_masks')
    point2mask(imgs_p, point_labels_p, masks_p)

    tiles_p = join(dst, 'tiles')
    creat_tiles(imgs_p, masks_p, tiles_p)

if __name__ == "__main__":
    imgs_p, labels_p, dst = argv[1], argv[2], argv[3]
    from_point_to_tiles(imgs_p, labels_p, dst)