import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import harmonic_mean
import re
from os.path import join, basename, split
from datetime import datetime
import sys
from scipy.stats import pearsonr, linregress
from tabulate import tabulate

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

def reshape_df(summary):
    targets = ['precision','sensitivity','f1_score']
    nums = summary.loc[:,'training image number'].tolist() * len(targets)
    values = ((summary.loc[:,targets]).unstack()).values
    types = [target for target in targets for _ in range(len(summary))]

    reshaped = pd.DataFrame({'socre':values, 'training image number':nums, 'scorer':types})
    return reshaped

def plot_metric(summary:pd.DataFrame, dst:str, dpi=300) -> None:
    os.makedirs(dst, exist_ok=True)
    sns.set()
    plt.figure()
    reshaped = reshape_df(summary)
    sns.lmplot(x="training image number", y="socre", hue="scorer", data=reshaped,
        markers=["x", "o", 'v'], palette="Set1", ci=None)
    plt.savefig(join(dst, f'{mark_time()}_metric.jpg'), dpi=dpi)

def func_df(summary, func=linregress, columns=['slope', 'intercept', 'correlation coefficient', 'two-sided p-value', 'Standard error']):
    corrs = [list(func(summary['training image number'], summary[i])) for i in ['precision','sensitivity','f1_score']]
    corr_df = pd.DataFrame(corrs, index=['precision','sensitivity','f1_score'], columns=columns)
    return corr_df

def from_result_to_plot(result_dir:str, dst, dpi=300, markdown=False):
    summary = metric_result(result_dir)
    plot_metric(summary, dst, dpi=dpi)
    if markdown:
        with open(join(dst, 'mark_down.txt'),'w+') as md:
            md.write(tabulate(summary, tablefmt="pipe", headers="keys"))
            md.write('\n')
            md.write(tabulate(func_df(summary), tablefmt="pipe", headers="keys"))

if __name__ == "__main__":
    result_dir = sys.argv[1]
    dst = sys.argv[2]
    from_result_to_plot(result_dir, dst, markdown=True)