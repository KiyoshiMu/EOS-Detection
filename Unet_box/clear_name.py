from os.path import join, basename, split, splitext
from os import walk, listdir, rename, makedirs
from shutil import move
import sys

def creat_path_list(f_p):
    img_p_list = [join(item[0], fn) for item in walk(f_p) for fn in item[2] if item[2]]
    return img_p_list
    
def clear_name(img_p_list, dst, befores, afters):
    assert len(befores) == len(afters), "rename times doesn't match"
    
    for path in img_p_list:
        parent, f = split(path)
        fn = f
        for before, after in zip(befores, afters):
            if before in f:
                print(f'rename {f}')
            fn = fn.replace(before, after)
        try:
            rename(join(parent, f), join(parent, fn))
        except FileExistsError:
            print(f)
            move(join(parent, f), join(dst, f))

def clear(dir_path, dst):
    makedirs(dst, exist_ok=True)
    lib = ['_ch00', '',
    '_.', '.',
    ' .', '.',
    ' ', '_',
    '__', '_']
    befores = lib[::2]
    afters = lib[1::2]
    clear_name(creat_path_list(dir_path), dst, befores, afters)

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst = sys.argv[2]
    clear(dir_path, dst)