from OSLActionSpotting.core.utils.eval import evaluate_e2e
from OSLActionSpotting.core.utils.io import  store_json, clear_files

from OSLActionSpotting.core.utils.lightning import CustomProgressBar, MyCallback
import pytorch_lightning as pl

from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import os

import torch

from OSLActionSpotting.datasets.builder import build_dataset
from abc import ABC, abstractmethod


def build_trainer(cfg, model = None, default_args=None):
    if cfg.type == "trainer_e2e":
        optimizer, scaler = model.get_optimizer({'lr': cfg.learning_rate})

        # Warmup schedule
        num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            cfg, optimizer, num_steps_per_epoch)
        trainer = Trainer_e2e(cfg,model,optimizer,scaler,lr_scheduler, default_args['work_dir'], default_args['dali'], default_args['repartitions'], default_args['cfg_test'], default_args['cfg_challenge'], default_args['cfg_val_data_frames'])
    else:
        call=MyCallback()
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,devices=[cfg.GPU],callbacks=[call,CustomProgressBar(refresh_rate=1)],num_sanity_val_steps=0)
    return trainer

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])

class Trainer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

class Trainer_e2e(Trainer):
    def __init__(self, 
                 args, 
                 model, 
                 optimizer,
                 scaler,
                 lr_scheduler, 
                 work_dir, 
                 dali, 
                 repartitions = None, 
                 cfg_test = None, 
                 cfg_challenge = None, 
                 cfg_val_data_frames = None):
        self.losses = []
        self.best_epoch = 0
        self.best_criterion = 0 if args.criterion == 'map' else float('inf')

        self.num_epochs = args.num_epochs
        self.epoch = 0
        self.model = model

        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        self.acc_grad_iter = args.acc_grad_iter

        self.start_val_epoch = args.start_val_epoch
        self.criterion = args.criterion
        self.save_dir = work_dir
        self.dali = dali
        self.inference_batch_size = args.inference_batch_size
        # self.base_num_workers = args.base_num_workers

        self.repartitions = repartitions 
        self.cfg_test = cfg_test
        self.cfg_challenge = cfg_challenge
        self.cfg_val_data_frames = cfg_val_data_frames

    def train(self,train_loader,val_loader,val_data_frames,classes):
        for epoch in range(self.epoch, self.num_epochs):
            train_loss = self.model.epoch(
                train_loader, self.dali, self.optimizer, self.scaler,
                lr_scheduler=self.lr_scheduler, acc_grad_iter=self.acc_grad_iter)
            
            val_loss = self.model.epoch(val_loader, self.dali, acc_grad_iter=self.acc_grad_iter)
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
                        
            val_mAP = 0
            if self.criterion == 'loss':
                if val_loss < self.best_criterion:
                    self.best_criterion = val_loss
                    self.best_epoch = epoch
                    print('New best epoch!')
            elif self.criterion == 'map':
                if epoch >= self.start_val_epoch:
                    pred_file = None
                    if self.save_dir is not None:
                        pred_file = os.path.join(
                            self.save_dir, 'pred-val.{}'.format(epoch))
                        os.makedirs(self.save_dir, exist_ok=True)
                    val_mAP = evaluate_e2e(self.model, self.dali,  val_data_frames, 'VAL', classes, 
                                           pred_file, save_scores=False , dataloader_params= self.cfg_val_data_frames.dataloader)
                    if self.criterion == 'map' and val_mAP > self.best_criterion:
                        self.best_criterion = val_mAP
                        self.best_epoch = epoch
                        print('New best epoch!')
            else:
                print('Unknown criterion:', self.criterion)

            self.losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss,
            'val_mAP': val_mAP})
            if self.save_dir is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                store_json(os.path.join(self.save_dir, 'loss.json'), self.losses,
                            pretty=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir,
                        'checkpoint_{:03d}.pt'.format(epoch)))
                clear_files(self.save_dir, r'optim_\d+\.pt')
                torch.save(
                    {'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'lr_state_dict': self.lr_scheduler.state_dict()},
                    os.path.join(self.save_dir,
                                    'optim_{:03d}.pt'.format(epoch)))
            
        print('Best epoch: {}\n'.format(self.best_epoch))

        if self.dali:
            train_loader.delete()
            val_loader.delete()
            if self.criterion == 'map':
                val_data_frames.delete()

        if self.save_dir is not None:
            self.model.load(torch.load(os.path.join(
                self.save_dir, 'checkpoint_{:03d}.pt'.format(self.best_epoch))))

            # Evaluate on VAL if not already done
            eval_splits = ['val'] if self.criterion != 'map' else []

            # Evaluate on hold out splits
            eval_splits += ['test', 'challenge']
            for split in eval_splits:
                if split == 'val':
                    cfg_tmp = self.cfg_val_data_frames
                if split == 'test':
                    cfg_tmp = self.cfg_test
                elif split == 'challenge':
                    cfg_tmp = self.cfg_challenge
                split_path = os.path.join(cfg_tmp.path)
                # split_path = os.path.join(self.labels_dir,
                #     '{}.json'.format(split))
                if os.path.exists(split_path):
                    cfg_tmp.overlap_len = cfg_tmp.clip_len // 2
                    split_data = build_dataset(cfg_tmp,None,{'repartitions' : self.repartitions, 'classes' : classes})
                        # split_data = DaliDataSetVideo(
                        #     cfg_tmp.dataloader.batch_size,cfg_tmp.output_map,self.repartitions[0],
                        #     classes, cfg_tmp.label_file,
                        #     cfg_tmp.modality, cfg_tmp.clip_len, cfg_tmp.stride, cfg_tmp.data_root, 
                        #     crop_dim=cfg_tmp.crop_dim, overlap_len=cfg_tmp.clip_len // 2)
                    # else:
                    #     split_data = ActionSpotVideoDataset(
                    #     classes, split_path, args.frame_dir, args.modality,
                    #     args.clip_len, overlap_len=args.clip_len // 2,
                    #     crop_dim=args.crop_dim)
                    split_data.print_info()

                    pred_file = None
                    if self.save_dir is not None:
                        pred_file = os.path.join(
                            self.save_dir, 'pred-{}.{}'.format(split, self.best_epoch))

                    evaluate_e2e(self.model, self.dali, split_data, split.upper(), classes, pred_file,
                            calc_stats=split != 'challenge', dataloader_params= cfg_tmp.dataloader)
                    
                    if self.dali:
                        split_data.delete()