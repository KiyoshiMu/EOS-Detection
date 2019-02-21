import sys
import os
sys.path.append(os.path.abspath('.'))
import pickle
from Unet_box.EOS_tools import path_list_creator
from Unet_box.use_model import Unet_predictor
from metric.metric_tools import load_pickle
from Unet_box.EOS_tools import get_name
from Unet_box.unet_tools import show_result

def find_models(dir_p:str) -> list:
    files = [x for x in path_list_creator(dir_p) if x[-3:]=='.h5']
    return files

def test_dict(img_dir:str) -> list:
    return {get_name(fp):fp for fp in path_list_creator(img_dir)}

def evaluate(models_p:str, img_dir:str, refer_pkl:str, dst:str) -> None:
    actor = Unet_predictor()
    model_ps = find_models(models_p)
    # print(model_ps)
    all_imgs = test_dict(img_dir)
    answers = load_pickle(refer_pkl)
    imgs = {name:img_p for name, img_p in all_imgs.items() if name in answers}
    for model_p in model_ps:
        actor.load_weights(model_p)

        vis_dst = os.path.join(dst, os.path.basename(model_p))
        os.makedirs(vis_dst, exist_ok=True)
        for name, img_p in imgs.items():
            refer = answers[name]
            if isinstance(refer, list):
                label_points = refer
                label_p = None
            elif isinstance(refer, str):
                label_points = None
                label_p = refer
            actor.metric(img_p, name=name, label_p=label_p, label_points=label_points, dst=vis_dst, show_mask=True)
        show_result(actor.metric_record, actor.metric_keys.split(', '), 
        title='Metric', dst=vis_dst)

        actor.clear()

if __name__ == "__main__":
    models_p = sys.argv[1]
    img_dir = sys.argv[2]
    refer_pkl = sys.argv[3]
    dst = sys.argv[4]
    evaluate(models_p, img_dir, refer_pkl, dst)