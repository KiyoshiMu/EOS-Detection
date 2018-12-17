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
        for before, after in zip(befores, afters):
            fn = f.replace(before, after)
            if before in fn:
                print(f'rename  {f}')
            try:
                rename(join(parent, f), join(parent, fn))
            except FileExistsError:
                print(f)
                move(join(parent, f), join(dst, f))

def clear(dir_path, dst):
    
    lib = ['_ch00', '',
    '_.', '.',
    ' .', '.',
    ' ', '_',
    '__', '_']
    befores = lib[::2]
    afters = lib[1::2]
    clear_name(creat_path_list(dir_path), dst, befores, afters)

if __name__ == "__main__":
    makedirs(sys.argv[2], exist_ok=True)
    clear(sys.argv[1], sys.argv[2])