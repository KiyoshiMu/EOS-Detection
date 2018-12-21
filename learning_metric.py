import pickle
import os
import sys
from Unet_box.EOS_tools import path_list_creator
from Unet_box.use_model import Unet_predictor
from metric_circle import load_pickle
from Unet_box.EOS_tools import get_name
from Unet_box.unet_tools import show_result

def find_models(path:str) -> list:
    files = [x for x in path_list_creator(path) if x[-3]=='.h5']
    return files

def test_dict(img_dir:str) -> list:
    return {get_name(fp):fp for fp in path_list_creator(img_dir)}

def tests(models_p:str, img_dir:str, answer_pkl:str, dst:str) -> None:
    actor = Unet_predictor()
    model_ps = find_models(models_p)
    all_imgs = test_dict(img_dir)
    answers = load_pickle(answer_pkl)
    imgs = {name:img_p for name, img_p in all_imgs.items() if name in answers}
    for model_p in model_ps:
        actor.load_weights(model_p)

        vis_dst = os.path.join(dst, os.path.basename(model_p))
        os.makedirs(vis_dst, exist_ok=True)
        for name, img_p in imgs.items():
            label_points = answers[name]
            actor.metric(img_p, name=name, label_points=label_points, dst=vis_dst)
        show_result(actor.metric_record, actor.metric_keys.split(', '), 
        title='Metric', dst=vis_dst)

        actor.clear()

if __name__ == "__main__":
    models_p = sys.argv[1]
    img_dir = sys.argv[2]
    answer_pkl = sys.argv[3]
    dst = sys.argv[4]
    tests(models_p, img_dir, answer_pkl, dst)