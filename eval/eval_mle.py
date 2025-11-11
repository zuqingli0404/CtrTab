import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
from mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='real')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')
parser.add_argument('--addnoise', type=bool, default = False, help='If adding noise to the dataset')
parser.add_argument('--noisescale', type=float, default = 0.0, help='the scale of the noise')
parser.add_argument('--savepath', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()

# def preprocess(train, test, info)

#     def norm_data(data, )

def add_gaussian_noise(data, scale, num_col_idx):
    std_dev = 0.1 * scale
    data = data.copy()
    noise = np.zeros_like(data, dtype=object)
    for idx in num_col_idx:
        try:
            numeric_column = data[:, idx].astype(float)
            noise_column = np.random.normal(loc=0.0, scale=std_dev, size=data.shape[0])
            data[:, idx] = numeric_column + noise_column
        except ValueError:
            print(f"ValueError")
    return data

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model
    
    if not args.path:
        train_path = f'synthetic/{dataname}/{model}.csv'
    else:
        train_path = args.path
    test_path = f'synthetic/{dataname}/test.csv'

    train = pd.read_csv(train_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()


    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    if args.addnoise:
        test = add_gaussian_noise(test, args.noisescale, num_col_idx)

    task_type = info['task_type']

    evaluator = get_evaluator(task_type)

    if task_type == 'regression':
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)
        
        overall_scores = {}
        for score_name in ['best_r2_scores', 'best_rmse_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    else:
        best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)

        overall_scores = {}
        for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    if not os.path.exists(f'eval/mle/{dataname}'):
        os.makedirs(f'eval/mle/{dataname}')
    
    if args.savepath != 'None':
        save_path = f'eval/mle/{dataname}/{args.savepath}.json'
    else:
        save_path = f'eval/mle/{dataname}/{model}.json'
    print('Saving scores to ', save_path)
    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file, indent=4, separators=(", ", ": "))
