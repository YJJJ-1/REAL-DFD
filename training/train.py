import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=None)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
env_local_rank = os.environ.get('LOCAL_RANK')
if env_local_rank is not None:
    args.local_rank = int(env_local_rank)
elif args.local_rank is None:
    args.local_rank = 0

if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    sampler = DistributedSampler(train_set) if config['ddp'] else None
    return torch.utils.data.DataLoader(
        dataset=train_set, batch_size=config['train_batchSize'],
        shuffle=sampler is None, sampler=sampler,
        num_workers=int(config['workers']), collate_fn=train_set.collate_fn,
    )


def prepare_testing_data(config):
    loaders = {}
    for test_name in config['test_dataset']:
        test_config = config.copy()
        test_config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=test_config, mode='test')
        loaders[test_name] = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=config['test_batchSize'],
            shuffle=False, num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn, drop_last=False,
        )
    return loaders


def choose_optimizer(model, config):
    opts = config['optimizer']['adam']
    return optim.Adam(
        model.parameters(), lr=opts['lr'], weight_decay=opts['weight_decay'],
        betas=(opts['beta1'], opts['beta2']), eps=opts['eps'],
        amsgrad=opts['amsgrad'],
    )


def choose_scheduler(config, optimizer):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'],
    )


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.task_target:
        config['task_target'] = args.task_target
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )

        logger.info(
            "DDP mapping: "
            f"RANK={dist.get_rank()}, "
            f"LOCAL_RANK={args.local_rank}, "
            f"WORLD_SIZE={dist.get_world_size()}, "
            f"CUDA_DEVICE={torch.cuda.current_device()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # start training
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        # Required by DistributedSampler so that each epoch uses a new shuffle.
        if config['ddp'] and isinstance(
            train_data_loader.sampler,
            DistributedSampler
        ):
            train_data_loader.sampler.set_epoch(epoch)

        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )

        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing "
                f"{metric_scoring}: "
                f"{parse_metric_for_print(best_metric)}!"
            )

        # Epoch-level updates must occur inside the epoch loop.
        if scheduler is not None:
            scheduler.step()

    logger.info(
        "Stop Training on best Testing metric {}".format(
            parse_metric_for_print(best_metric)
        )
    )

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()
