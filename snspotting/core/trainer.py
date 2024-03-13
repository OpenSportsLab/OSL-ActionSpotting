from snspotting.core.utils.lightning import CustomProgressBar, MyCallback
import pytorch_lightning as pl

from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)

def build_trainer(cfg, model = None, default_args=None):
    if cfg.training.type == "trainer_e2e":
        optimizer, scaler = model.get_optimizer({'lr': cfg.learning_rate})

        # Warmup schedule
        num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            cfg, optimizer, num_steps_per_epoch)
        print(optimizer)
        print(scaler)
        print(num_steps_per_epoch)
        print(num_epochs, lr_scheduler)
        trainer = Trainer(cfg,model,optimizer,scaler,lr_scheduler)
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
    def __init__(self,args,model,optimizer,scaler,lr_scheduler):
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
        
    def train(self,train_loader,val_loader):
        for epoch in range(self.epoch, self.num_epochs):
            train_loss = self.model.epoch(
                train_loader, True, self.optimizer, self.scaler,
                lr_scheduler=self.lr_scheduler, acc_grad_iter=self.acc_grad_iter)
            
            val_loss = self.model.epoch(val_loader, True, acc_grad_iter=self.acc_grad_iter)
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))