from tabulate import tabulate
from snspotting.core.utils.eval import process_frame_predictions
from snspotting.core.utils.io import  store_json, store_gz_json, clear_files
from snspotting.core.utils.score import compute_mAPs

from snspotting.core.utils.lightning import CustomProgressBar, MyCallback
import pytorch_lightning as pl

from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import os

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from snspotting.datasets.frame import DaliDataSetVideo

def build_trainer(cfg, model = None, default_args=None):
    if cfg.type == "trainer_e2e":
        optimizer, scaler = model.get_optimizer({'lr': cfg.learning_rate})

        # Warmup schedule
        num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            cfg, optimizer, num_steps_per_epoch)
        print(optimizer)
        print(scaler)
        print(num_steps_per_epoch)
        print(num_epochs, lr_scheduler)
        trainer = Trainer(cfg,model,optimizer,scaler,lr_scheduler, default_args['work_dir'], default_args['dali'], default_args['modality'], default_args['clip_len'], default_args['crop_dim'], default_args['label_dir'])
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

class Trainer():
    def __init__(self,args,model,optimizer,scaler,lr_scheduler, work_dir, dali, modality, clip_len, crop_dim, labels_dir):
        self.losses = []
        self.best_epoch = None
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

        self.losses = []
        self.best_epoch = None
        self.best_criterion = 0 if args.criterion == 'map' else float('inf')

        self.modality = modality 
        self.clip_len = clip_len
        self.crop_dim = crop_dim

        self.labels_dir = labels_dir

    def train(self,train_loader,val_loader,val_data_frames,classes):
        for epoch in range(self.epoch, self.num_epochs):
            train_loss = self.model.epoch(
                train_loader, True, self.optimizer, self.scaler,
                lr_scheduler=self.lr_scheduler, acc_grad_iter=self.acc_grad_iter)
            
            val_loss = self.model.epoch(val_loader, True, acc_grad_iter=self.acc_grad_iter)
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
                    val_mAP = self.evaluate(self.model, self.dali,  val_data_frames, 'VAL', classes,
                                        pred_file, save_scores=False)
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

        train_loader.delete()
        val_loader.delete()
        val_data_frames.delete()

        if self.save_dir is not None:
            self.model.load(torch.load(os.path.join(
                self.save_dir, 'checkpoint_{:03d}.pt'.format(self.best_epoch))))

            # Evaluate on VAL if not already done
            eval_splits = ['val'] if self.criterion != 'map' else []

            # Evaluate on hold out splits
            eval_splits += ['test', 'challenge']
            for split in eval_splits:
                split_path = os.path.join(self.labels_dir,
                    '{}.json'.format(split))
                if os.path.exists(split_path):
                    if self.dali:
                        split_data = DaliDataSetVideo(
                            4,["data","label"],[0,1,2,3],
                            classes, split_path,
                            self.modality, self.clip_len,
                            crop_dim=self.crop_dim, overlap_len=self.clip_len // 2)
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

                    self.evaluate(self.model, self.dali, split_data, split.upper(), classes, pred_file,
                            calc_stats=split != 'challenge')
                    
                    if self.dali:
                        split_data.delete()
                
    def evaluate(self, model, dali, dataset, split, classes, save_pred, calc_stats=True,
             save_scores=True):
        pred_dict = {}
        for video, video_len, _ in dataset.videos:
            pred_dict[video] = (
                np.zeros((video_len, len(classes) + 1), np.float32),
                np.zeros(video_len, np.int32))

        # Do not up the batch size if the dataset augments
        batch_size = 1 if dataset.augment else self.inference_batch_size

        for clip in tqdm(dataset if dali else DataLoader(
                dataset, num_workers=8, pin_memory=True,
                batch_size=batch_size
        )):
            if batch_size > 1:
                # Batched by dataloader
                _, batch_pred_scores = model.predict(clip['frame'])
                for i in range(clip['frame'].shape[0]):
                    video = clip['video'][i]
                    scores, support = pred_dict[video]
                    pred_scores = batch_pred_scores[i]
                    start = clip['start'][i].item()
                    if start < 0:
                        pred_scores = pred_scores[-start:, :]
                        start = 0
                    end = start + pred_scores.shape[0]
                    if end >= scores.shape[0]:
                        end = scores.shape[0]
                        pred_scores = pred_scores[:end - start, :]
                    scores[start:end, :] += pred_scores
                    support[start:end] += 1

            else:
                # Batched by dataset
                scores, support = pred_dict[clip['video'][0]]

                start = clip['start'][0].item()
                start=start-1
                _, pred_scores = model.predict(clip['frame'][0])
                if start < 0:
                    pred_scores = pred_scores[:, -start:, :]
                    start = 0
                end = start + pred_scores.shape[1]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:,:end - start, :]

                print(pred_scores.shape)
                scores[start:end, :] += np.sum(pred_scores, axis=0)
                support[start:end] += pred_scores.shape[0]

        err, f1, pred_events, pred_events_high_recall, pred_scores = \
            process_frame_predictions(dataset, classes, pred_dict)

        avg_mAP = None
        if calc_stats:
            print('=== Results on {} (w/o NMS) ==='.format(split))
            print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

            def get_f1_tab_row(str_k):
                k = classes[str_k] if str_k != 'any' else None
                return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
            rows = [get_f1_tab_row('any')]
            for c in sorted(classes):
                rows.append(get_f1_tab_row(c))
            print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                        floatfmt='0.2f'))
            print()

            mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
            avg_mAP = np.mean(mAPs[1:])

        if save_pred is not None:
            store_json(save_pred + '.json', pred_events)
            store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
            if save_scores:
                store_gz_json(save_pred + '.score.json.gz', pred_scores)
        return avg_mAP