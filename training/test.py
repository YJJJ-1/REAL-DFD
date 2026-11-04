# -*- coding: gbk -*-
"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument('--alpha', type=float, default=None, help='override alpha value for testing')
parser.add_argument('--no-save-results', action='store_true', help='print metrics without writing score or summary CSVs')
parser.add_argument('--config-loader', action='store_true', help='use test_batchSize and workers from detector YAML')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()  
        config['test_dataset'] = test_name  
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'] if args.config_loader else 128,          
                shuffle=False, 
                num_workers=int(config['workers']) if args.config_loader else 8,           
                pin_memory=True,         
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def get_respect_acc(prob, label):
    pred = np.where(prob > 0.5, 1, 0)
    judge = (pred == label)
    real_idx = np.where(label == 0)[0]
    fake_idx = np.where(label == 1)[0]
    acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx) if len(real_idx) > 0 else 0.0
    acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx) if len(fake_idx) > 0 else 0.0
    return acc_real, acc_fake


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        if 'label_spe' in data_dict:
            data_dict.pop('label_spe')  
        data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  
        
        for key in data_dict.keys():
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key].to(device)

        predictions = inference(model, data_dict)
        
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        if 'feat' in predictions:
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)
    
def test_epoch(model, test_data_loaders, output_dir):
    model.eval()
    
    summary_metrics_list = []
    keys = test_data_loaders.keys()
    
    for key in keys:
        tqdm.write(f"\n===> Testing on dataset: {key}")
        data_dict = test_data_loaders[key].dataset.data_dict
        img_names = data_dict['image']
        
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        predictions_nps = predictions_nps.flatten()
        label_nps = label_nps.flatten()
        
        # 提取真假准确率
        acc_real, acc_fake = get_respect_acc(predictions_nps, label_nps)
        
        # 核心修改：检测标签类别数量，防止单类别计算 AUC/EER 报错
        unique_labels = np.unique(label_nps)
        if len(unique_labels) < 2:
            tqdm.write(f"Warning: Dataset {key} only contains label {unique_labels}. Skipping AUC/EER calculation.")
            metric_one_dataset = {} # 初始化为空，跳过报错函数
        else:
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=img_names)
        
        # 打印当前数据集指标 
        for k, v in metric_one_dataset.items():
            if k not in ['dataset_dict', 'pred', 'label', 'acc_real', 'acc_fake']:
                tqdm.write(f"{k}: {v:.4f}")
        tqdm.write(f"acc_real: {acc_real:.4f}")
        tqdm.write(f"acc_fake: {acc_fake:.4f}")
        
        # 将当前数据集的指标压入汇总列表
        summary_metrics_list.append({
            'Dataset': key,
            'ACC': metric_one_dataset.get('acc', acc_real if unique_labels[0]==0 else acc_fake),
            'AUC': metric_one_dataset.get('auc', np.nan),
            'Video_AUC': metric_one_dataset.get('video_auc', np.nan),
            'ACC_Real': acc_real,
            'ACC_Fake': acc_fake,
            'AVG_ACC': (acc_real + acc_fake) / 2.0,  
            'EER': metric_one_dataset.get('eer', np.nan),
            'AP': metric_one_dataset.get('ap', np.nan)
        })

        # 2. 剥离图片路径并保存真假分数表
        flat_img_names = [paths[0] if isinstance(paths, list) else paths for paths in img_names]
        
        real_mask = (label_nps == 0)
        fake_mask = (label_nps == 1)
        
        if np.any(real_mask) and not args.no_save_results:
            df_real = pd.DataFrame({
                'image_path': np.array(flat_img_names)[real_mask],
                'deepfake_score': predictions_nps[real_mask]
            })
            real_csv_path = os.path.join(output_dir, f"{key}_real_scores.csv")
            df_real.to_csv(real_csv_path, index=False)
            tqdm.write(f"Saved real scores to {real_csv_path}")
            
        if np.any(fake_mask) and not args.no_save_results:
            df_fake = pd.DataFrame({
                'image_path': np.array(flat_img_names)[fake_mask],
                'deepfake_score': predictions_nps[fake_mask]
            })
            fake_csv_path = os.path.join(output_dir, f"{key}_fake_scores.csv")
            df_fake.to_csv(fake_csv_path, index=False)
            tqdm.write(f"Saved fake scores to {fake_csv_path}")

    # 3. 输出全局指标总表
    if summary_metrics_list:
        summary_df = pd.DataFrame(summary_metrics_list)
        
        mean_row = summary_df.mean(numeric_only=True).to_dict()
        mean_row['Dataset'] = 'Average'
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
        
        if not args.no_save_results:
            summary_csv_path = os.path.join(output_dir, "total_summary_metrics.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            tqdm.write(f"\n===> Global Summary Metrics saved to {summary_csv_path}")
        tqdm.write(summary_df.to_string(index=False))


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def main():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
        
    weights_path = None
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    if args.alpha is not None:
        config['alpha'] = args.alpha
        print(f"\n===========================================")
        print(f"===> Dynamically set Alpha to: {config['alpha']}")
        print(f"===========================================\n")
    init_seed(config)

    if config.get('cudnn', True):
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)
    
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        new_state_dict = {}
        for k, v in ckpt.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)

        # Force alpha after loading checkpoint, because ckpt may overwrite alpha_calibrated
        current_alpha = float(config.get('alpha', 0.0))
        if hasattr(model, "set_calibrated_alpha"):
            model.set_calibrated_alpha(current_alpha)
        
        print(f"[Alpha Sweep] force inference alpha = {current_alpha}")
        
        output_dir = os.path.join(os.path.dirname(weights_path), f"test_results_alpha_{current_alpha}")
    else:
        print('Fail to load the pre-trained weights')
        current_alpha = config.get('alpha', 'default')
        output_dir = f"/data2/Mining/asym-result_alpha_{current_alpha}"
        
    if not args.no_save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    test_epoch(model, test_data_loaders, output_dir)
    print('===> All Tasks Done!')

if __name__ == '__main__':
    main()