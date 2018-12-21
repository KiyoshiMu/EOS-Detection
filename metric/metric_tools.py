import pickle
import os
from os.path import join

def load_pickle(pkl_path) -> dict:
    with open(pkl_path, 'rb') as temp:
        container = pickle.load(temp)
    return container

def save_dict(container:dict, name, dst) -> None:
    os.makedirs(dst, exist_ok=True)
    with open(join(dst, name+'.pkl'), 'wb+') as saver:
        pickle.dump(container, saver, pickle.HIGHEST_PROTOCOL)