# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
import gc
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up() 

        self.timenow = time_now
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        local_rank = int(
            os.environ.get(
                'LOCAL_RANK',
                self.config.get('local_rank', 0)
            )
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            rank_device = torch.device(f'cuda:{local_rank}')
        else:
            rank_device = torch.device('cpu')

        self.model.to(rank_device)
        self.model.device = rank_device
        self.config['local_rank'] = local_rank

        if self.config.get('ddp', False):
            if not dist.is_initialized():
                raise RuntimeError(
                    "DDP requested but torch.distributed is not initialized. "
                    "Call dist.init_process_group() before constructing Trainer."
                )

            current_device = torch.cuda.current_device()
            if current_device != local_rank:
                raise RuntimeError(
                    "CUDA/local-rank mismatch before DDP construction: "
                    f"LOCAL_RANK={local_rank}, "
                    f"torch.cuda.current_device()={current_device}."
                )

            self.logger.info(
                "Constructing DDP: "
                f"rank={dist.get_rank()}, "
                f"local_rank={local_rank}, "
                f"world_size={dist.get_world_size()}, "
                f"device={rank_device}"
            )

            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError("=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()
            return losses,predictions

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.logger.info("===> Epoch[{}] start!".format(epoch))
        
        test_best_metric = None
        step_cnt = epoch * len(train_data_loader)

        data_dict = train_data_loader.dataset.data_dict
        if self.config.get('save_dataset_results', True):
            self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        continue
                    loss_str += f"training-loss, {k}: {v_avg:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        continue
                    metric_str += f"training-metric, {k}: {v_avg:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                for name, recorder in train_recorder_loss.items():  
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  
                    recorder.clear()


            step_cnt += 1


        if True:
            if test_data_loaders is not None and (not self.config['ddp'] ):
                self.logger.info("===> Test start!")
                test_best_metric = self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)
            elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                self.logger.info("===> Test start!")
                test_best_metric = self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)

        return test_best_metric

        


    @staticmethod
    def _normalize_binary_prob(prob_array):
        """Convert model probability output to one fake score per sample."""
        prob_array = np.asarray(prob_array)

        if prob_array.ndim == 0:
            prob_array = prob_array.reshape(1)
        elif prob_array.ndim == 2:
            if prob_array.shape[1] == 1:
                prob_array = prob_array[:, 0]
            elif prob_array.shape[1] == 2:
                prob_array = prob_array[:, 1]
            else:
                raise ValueError(
                    f"Unexpected binary prediction shape: {prob_array.shape}. "
                    "Expected [B], [B, 1], or [B, 2]."
                )
        elif prob_array.ndim > 2:
            raise ValueError(
                f"Unexpected prediction rank: {prob_array.ndim}, "
                f"shape={prob_array.shape}."
            )

        return prob_array.reshape(-1)

    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        label_lists = []
        image_name_lists = []

        dataset_images = getattr(
            getattr(data_loader, 'dataset', None),
            'data_dict',
            {}
        ).get('image', None)

        for i, data_dict in tqdm(
            enumerate(data_loader),
            total=len(data_loader)
        ):
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')

            batch_names = data_dict.pop('name', None)
            batch_indices = data_dict.pop('index', None)

            if batch_names is not None:
                if isinstance(batch_names, (list, tuple)):
                    image_name_lists.extend(list(batch_names))
                elif isinstance(batch_names, np.ndarray):
                    image_name_lists.extend(
                        batch_names.reshape(-1).tolist()
                    )
                else:
                    image_name_lists.append(batch_names)
            elif batch_indices is not None and dataset_images is not None:
                if isinstance(batch_indices, torch.Tensor):
                    batch_indices = (
                        batch_indices.detach().cpu().numpy()
                    )
                batch_indices = np.asarray(
                    batch_indices
                ).reshape(-1)
                image_name_lists.extend(
                    [
                        dataset_images[int(idx)]
                        for idx in batch_indices
                    ]
                )

            data_dict['label'] = torch.where(
                data_dict['label'] != 0, 1, 0
            )

            for key, value in list(data_dict.items()):
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.cuda(
                        non_blocking=True
                    )

            predictions = self.inference(data_dict)

            batch_labels = (
                data_dict['label']
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            batch_probs = self._normalize_binary_prob(
                predictions['prob'].detach().cpu().numpy()
            )

            if len(batch_probs) != len(batch_labels):
                raise ValueError(
                    "Batch prediction/label mismatch during "
                    "evaluation: "
                    f"predictions={len(batch_probs)}, "
                    f"labels={len(batch_labels)}, "
                    f"prediction_shape="
                    f"{tuple(predictions['prob'].shape)}."
                )

            label_lists.extend(batch_labels.tolist())
            prediction_lists.extend(batch_probs.tolist())

            if type(self.model) is not AveragedModel:
                try:
                    if type(self.model) is DDP:
                        losses = self.model.module.get_losses(
                            data_dict, predictions
                        )
                    else:
                        losses = self.model.get_losses(
                            data_dict, predictions
                        )
                    for name, value in losses.items():
                        val = (
                            value.item()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        test_recorder_loss[name].update(val)
                except KeyError:
                    pass

        predictions_nps = np.asarray(
            prediction_lists
        ).reshape(-1)
        label_nps = np.asarray(label_lists).reshape(-1)
        image_names_nps = (
            np.asarray(
                image_name_lists,
                dtype=object
            ).reshape(-1)
            if len(image_name_lists) > 0
            else None
        )

        return (
            test_recorder_loss,
            predictions_nps,
            label_nps,
            None,
            image_names_nps,
        )

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            if self.config['save_ckpt'] and key not in FFpp_pool and (
                key == 'avg' or self.config.get('save_dataset_results', True)
            ):
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            if self.config.get('save_dataset_results', True):
                self.save_metrics('test', metric_one_dataset, key)
            
        if losses_one_dataset_recorder is not None and len(losses_one_dataset_recorder) > 0:
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                v_avg = v.average()
                if v_avg == None:
                    continue
                if self.config.get('save_dataset_results', True):
                    writer = self.get_writer('test', key, k)
                    writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg:.4f}    "
            self.logger.info(loss_str)
            
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k in ['dataset_dict', 'pred', 'label']:
                continue
            metric_str += f"testing-metric, {k}: {v:.4f}    "
            if self.config.get('save_dataset_results', True):
                writer = self.get_writer('test', key, k)
                writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
            
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        keys = test_data_loaders.keys()
        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict
            if self.config.get('save_dataset_results', True):
                self.save_data_dict('test', data_dict, key)

            current_loader = test_data_loaders[key]

            if isinstance(
                current_loader.sampler,
                DistributedSampler
            ):
                raise RuntimeError(
                    f"Test loader for '{key}' uses "
                    "DistributedSampler, but test_epoch() "
                    "runs only on rank 0. Rank 0 therefore "
                    "sees only one shard while "
                    "dataset.data_dict['image'] contains "
                    "the full dataset. Build test loaders "
                    "with sampler=None, shuffle=False, and "
                    "drop_last=False."
                )

            (
                losses_one_dataset_recorder,
                predictions_nps,
                label_nps,
                feature_nps,
                sampled_image_names,
            ) = self.test_one_dataset(current_loader)

            losses_all_datasets[key] = (
                losses_one_dataset_recorder
            )

            if sampled_image_names is not None:
                image_names = sampled_image_names
            else:
                image_names = np.asarray(
                    data_dict['image'],
                    dtype=object
                ).reshape(-1)

            predictions_nps = np.asarray(
                predictions_nps
            ).reshape(-1)
            label_nps = np.asarray(
                label_nps
            ).reshape(-1)

            self.logger.info(
                f"[Metric input check] dataset={key}, "
                f"images={image_names.shape}, "
                f"predictions={predictions_nps.shape}, "
                f"labels={label_nps.shape}, "
                f"sampler="
                f"{type(current_loader.sampler).__name__}, "
                f"drop_last={current_loader.drop_last}"
            )

            if not (
                len(image_names)
                == len(predictions_nps)
                == len(label_nps)
            ):
                raise ValueError(
                    f"Metric input length mismatch for "
                    f"'{key}': images={len(image_names)}, "
                    f"predictions={len(predictions_nps)}, "
                    f"labels={len(label_nps)}. Likely "
                    "causes: DistributedSampler on the test "
                    "loader, drop_last=True, a filtered "
                    "dataset whose data_dict was not updated, "
                    "or predictions with an unexpected shape."
                )

            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps,
                y_true=label_nps,
                img_names=image_names
            )
            
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    if k not in ['dataset_dict']:
                        metric_str += f"testing-metric, {k}: {v:.4f}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
