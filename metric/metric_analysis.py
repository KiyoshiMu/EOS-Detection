import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import harmonic_mean
import re
from os.path import join, basename, split
from datetime import datetime
import sys

def gen_xlsx(dir_path):
    yield from [(int(re.findall(r'\d+',split(item[0])[-1])[0]),
        join(item[0], fn)) for item in os.walk(dir_path) for fn in item[2] if item[2] and fn[-4:] == 'xlsx']

def mark_time():
    now = datetime.now()
    time = now.strftime('%m-%d_%H-%M')
    return time

def compress(df):
    part_sum = df.loc[:,['true_positive', 'false_positive', 'false_negative']].sum()
    true_positive, false_positive, false_negative = part_sum
    part_sum['precision'] = true_positive / (true_positive + false_positive)
    part_sum['sensitivity']= true_positive / (true_positive + false_negative)
    part_sum['f1_score'] = harmonic_mean([part_sum['precision'], part_sum['sensitivity']])
    return part_sum

def metric_result(result_dir):
    x = []
    dfs = []
    for img_num, result in gen_xlsx(result_dir):
        part_sum = compress(pd.read_excel(result))
        x.append(img_num)
        dfs.append(part_sum)
    summary = pd.concat(dfs, axis=1).T
    summary['training image number'] = x
    summary = summary.astype({c:'int' for c in summary.columns[:3]})
    return summary

def reshape_df(result):
    targets = ['precision','sensitivity','f1_score']
    nums = result.loc[:,'training image number'].tolist() * len(targets)
    values = ((result.loc[:,targets]).unstack()).values
    types = [target for target in targets for _ in range(len(result))]

    reshaped = pd.DataFrame({'socre':values, 'training image number':nums, 'scorer':types})
    return reshaped

def plot_metric(result:pd.DataFrame, dst:str, dpi=300) -> None:
    os.makedirs(dst, exist_ok=True)
    sns.set()
    plt.figure()
    reshaped = reshape_df(result)
    sns.lmplot(x="training image number", y="socre", hue="scorer", data=reshaped,
        markers=["x", "o", 'v'], palette="Set1", ci=None)
    plt.savefig(join(dst, f'{mark_time()}_metric.jpg'), dpi=dpi)

def from_result_to_plot(result_dir:str, dst, dpi=300):
    result = metric_result(result_dir)
    plot_metric(result, dst, dpi=dpi)

if __name__ == "__main__":
    result_dir = sys.argv[1]
    dst = sys.argv[2]
    from_result_to_plot(result_dir, dst)