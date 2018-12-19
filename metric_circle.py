import pickle
from Unet_trainor import model_train
from Unet_box.EOS_tools import read_from_path_list
import random
import os
import sys

def load_pickle(pkl_path) -> dict:
    with open(pkl_path, 'rb') as temp:
        container = pickle.load(temp)
        return container

def data_prepare(family:dict, kind):
    p_list = []
    for value in family[kind].values():
        p_list.extend(value)
    islabel = True if kind == 'labels' else False
    data = read_from_path_list(p_list, islabel=islabel)
    return data

def sub_dict(train_dict:dict, key_list) -> dict:
    return {'raw_imgs':{key:train_dict['raw_imgs'][key] for key in key_list}, 
            'labels':{key:train_dict['labels'][key] for key in key_list}}

def from_dict_to_data(family:dict):
    X = data_prepare(family, 'raw_imgs')
    Y = data_prepare(family, 'labels')
    return X, Y

def train_model(train_pkl_path, test_pkl_path, dst):
    train_dict = load_pickle(train_pkl_path)
    test_dict = load_pickle(test_pkl_path)
    X_test, Y_test = from_dict_to_data(test_dict)
    
    train_names = list(train_dict['labels'].keys())
    train_amount = len(train_names)
    # in this case keras will only take care of the shuffle of tiles, now it's for whole images
    random.shuffle(train_names)
    train_img_num = 0
    while train_img_num < train_amount:
        train_img_num += 10
        if train_img_num > train_amount:
            train_img_num = train_amount
        sub_train_name = train_names[:train_img_num]
        sub_train_dict = sub_dict(train_dict, sub_train_name)
        X_train, Y_train = from_dict_to_data(sub_train_dict)
        result_dst = os.path.join(dst, str(train_img_num))
        os.makedirs(result_dst, exist_ok=True)
        _ = model_train(X_train, Y_train, X_test, Y_test, model_name=str(train_img_num), dst=result_dst)

def batch_stat():
    pass
    # TODO the 
    # actor = Unet_predictor(model_p)


    # actor.metric( img_p: str, label_p:str=None, show_info=False, 
        # show_img=True, dst='.', name=None, show_mask=False, target=15, label_points=None)

if __name__ == "__main__":
    train_pkl_path = sys.argv[1]
    test_pkl_path = sys.argv[2]
    dst = sys.argv[3]
    train_model(train_pkl_path, test_pkl_path, dst)