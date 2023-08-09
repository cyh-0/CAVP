#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/2 下午3:23
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : engine.py
import argparse
import os
import shutil
import time
import torch
import torch.distributed as dist
from utils.pyt_utils import link_file, make_dir
from collections import OrderedDict


class State(object):
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.dataloader = None
        self.model_v = None
        self.model_a = None
        self.optimizer_v = None
        self.optimizer_a = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['epoch', 'iteration', 'dataloader', 'model_v', 'model_a',
                         'optimizer_v', 'optimizer_a']
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_arg, logger):
        self.logger = logger
        self.state = State()
        self.devices = None
        self.distributed = custom_arg.ddp

        self.parser = argparse.ArgumentParser()
        self.inject_default_parser()
        
        self.args = custom_arg

        self.local_rank = 0 if self.args.local_rank < 0 else self.args.local_rank
        self.gpus = self.args.gpus
        self.world_size = self.args.world_size

        if self.distributed:
            os.environ['MASTER_ADDR'] = '127.0.0.4'
            os.environ['MASTER_PORT'] = '9904'
            dist.init_process_group(backend="nccl", init_method='env://', rank=self.local_rank,
                                    world_size=self.world_size)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-p', '--port', type=str, default='16001', dest="port",
                       help='port for init_process_group')
        p.add_argument('--save_path', default=None)

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration
    
    def get_global_iteration(self,):
        return len(self.state.dataloader) * self.state.epoch + self.state.iteration

    def save_checkpoint(self, path):
        # self.logger.info("\nSaving checkpoint to file {}".format(path))
        state_dict = {}
        # new_state_dict_a = OrderedDict()
        # for k, v in self.state.model_a.state_dict().items():
        #     key = k
        #     if k.split('.')[0] == 'module':
        #         key = k[7:]
        #     new_state_dict_a[key] = v

        # new_state_dict_v = OrderedDict()
        # for k, v in self.state.model_v.state_dict().items():
        #     key = k
        #     if k.split('.')[0] == 'module':
        #         key = k[7:]
        #     new_state_dict_v[key] = v

        # state_dict['audio_model'] = new_state_dict_a
        # state_dict['visual_model'] = new_state_dict_v
        state_dict['model'] = self.state.model_v.state_dict()

        state_dict['audio_optimizer'] = self.state.optimizer_a.state_dict()
        state_dict['visual_optimizer'] = self.state.optimizer_v.state_dict()

        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration

        torch.save(state_dict, path)
        del state_dict

    def link_tb(self, source, target):
        make_dir(source)
        make_dir(target)
        link_file(source, target)

    def save_and_link_checkpoint(self, snapshot_dir, m_ap=None):
        current_epoch_checkpoint = os.path.join(snapshot_dir, 'epoch-{}-mAP-{}.pth'.format(self.state.epoch, m_ap))

        self.save_checkpoint(current_epoch_checkpoint)
        last_epoch_checkpoint = os.path.join(snapshot_dir, 'epoch-last.pth')
        shutil.copy(current_epoch_checkpoint, last_epoch_checkpoint)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            self.logger.warning("A exception occurred during Engine initialization, give up running process")
            return False
