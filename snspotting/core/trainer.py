import torch
import os


# from .training import * 
# from .evaluation import *
import logging
# from snspotting.models.learnablepooling import CustomProgressBar, MyCallback
import pytorch_lightning as pl



# def build_trainer(cfg, model, default_args=None):
#     """Build a trainer from config dict.

#     Args:
#         cfg (dict): Config dict. It should at least contain the key "type".
#         default_args (dict | None, optional): Default initialization arguments.
#             Default: None.

#     Returns:
#         trainer: The constructed trainer.
#     """
#     criterion = build_criterion(cfg.criterion)
#     optimizer = build_optimizer(model.parameters(), cfg.optimizer)
#     scheduler = build_scheduler(optimizer, cfg.scheduler)


#     if cfg.type == "trainer_pooling" or cfg.type == "trainer_CALF":
#         trainer = Trainer(cfg=cfg,
#                         train_one_epoch=train_one_epoch if cfg.type == "trainer_pooling" else train_one_epoch_calf,
#                         valid_one_epoch=train_one_epoch if cfg.type == "trainer_pooling" else train_one_epoch_calf,
#                         model=model,
#                         criterion=criterion,
#                         optimizer=optimizer,
#                         scheduler=scheduler)
#     else:
#         trainer = None
#     return trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer,pl_module)
        items.pop("v_num", None)
        return items
    
class MyCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_validation_epoch_end(self, trainer, pl_module):
        loss_validation = pl_module.losses.avg
        state = {
                'epoch': trainer.current_epoch + 1,
                'state_dict': pl_module.model.state_dict(),
                'best_loss': pl_module.best_loss,
                'optimizer': pl_module.optimizer.state_dict(),
            }

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            pl_module.best_state = state
            # torch.save(state, best_model_path)

        # Reduce LR on Plateau after patience reached
        prevLR = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(loss_validation)
        currLR = self.optimizer.param_groups[0]['lr']

        if (currLR is not prevLR and self.scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")
        if (prevLR < 2 * self.scheduler.eps and
            self.scheduler.num_bad_epochs >= self.scheduler.patience):
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            self.should_stop=True

def build_trainer(cfg, default_args=None):
    call=MyCallback()
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,devices=[0],callbacks=[call,CustomProgressBar(refresh_rate=1)],num_sanity_val_steps=0)
    return trainer,call
# class Trainer():
#     def __init__(self, cfg, 
#                 train_one_epoch, 
#                 valid_one_epoch,
#                 model,
#                 criterion,
#                 optimizer,
#                 scheduler):
#         self.train_one_epoch = train_one_epoch
#         self.valid_one_epoch = valid_one_epoch

#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.criterion = criterion
#         self.model = model

#         self.cfg = cfg
#         self.max_epochs = cfg.max_epochs

#     def train(self,
#             train_loader,
#             val_loader):
#         best_loss = 9e99

#         # loop over epochs
#         for epoch in range(self.max_epochs):
#             # best_model_path = os.path.join(self.cfg.work_dir, "model.pth.tar")

#             # train for one epoch
#             loss_training = self.train_one_epoch(train_loader, self.model, self.criterion,
#                                 self.optimizer, self.cfg.GPU, epoch + 1, backprop=True)

#             # evaluate on validation set
#             loss_validation = self.train_one_epoch(
#                 val_loader, self.model, self.criterion, self.optimizer, self.cfg.GPU, epoch + 1, backprop=False)

#             state = {
#                 'epoch': epoch + 1,
#                 'state_dict': self.model.state_dict(),
#                 'best_loss': best_loss,
#                 'optimizer': self.optimizer.state_dict(),
#             }

#             # remember best prec@1 and save checkpoint
#             is_better = loss_validation < best_loss
#             best_loss = min(loss_validation, best_loss)

#             # Save the best model based on loss only if the evaluation frequency too long
#             if is_better:
#                 best_state = state
#                 # torch.save(state, best_model_path)

#             # Reduce LR on Plateau after patience reached
#             prevLR = self.optimizer.param_groups[0]['lr']
#             self.scheduler.step(loss_validation)
#             currLR = self.optimizer.param_groups[0]['lr']
#             if (currLR is not prevLR and self.scheduler.num_bad_epochs == 0):
#                 logging.info("Plateau Reached!")

#             if (prevLR < 2 * self.scheduler.eps and
#                     self.scheduler.num_bad_epochs >= self.scheduler.patience):
#                 logging.info(
#                     "Plateau Reached and no more reduction -> Exiting Loop")
#                 break

#         return best_state




# import time
# from tqdm import tqdm
# import torch

# from SoccerNet.Evaluation.utils import AverageMeter

# # def train_one_epoch(
# #         dataloader,
# #         model,
# #         criterion,
# #         optimizer,
# #         gpu,
# #         epoch,
# #         calf,
# #         backprop=False):

# #     batch_time = AverageMeter()
# #     data_time = AverageMeter()
# #     losses = AverageMeter()

# #     # switch to train mode
# #     if backprop:
# #         model.train()
# #     else:
# #         model.eval()

# #     end = time.time()
# #     with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
# #         for i, tuples in t:
# #             if calf:
# #                 feats, labels, targets = tuples
# #             else:
# #                 feats, labels = tuples
            
# #             # if cfg.GPU >= 0:
# #             if gpu >=0:
# #                 feats = feats.cuda()
# #                 labels = labels.cuda().float() if calf else labels.cuda()
# #                 if calf : targets = targets.cuda().float()
            
# #             if calf : feats=feats.unsqueeze(1)

# #             # compute output
# #             if calf : 
# #                 output_segmentation, output_spotting = model(feats)
# #             else :
# #                 output = model(feats)

# #             # hand written NLL criterion
# #             loss = criterion([labels, targets] if calf else labels, [output_segmentation, output_spotting] if calf else output)

# #             # measure accuracy and record loss
# #             losses.update(loss.item(), feats.size(0))

# #             if backprop:
# #                 # compute gradient and do SGD step
# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 optimizer.step()

# #             # measure elapsed time
# #             batch_time.update(time.time() - end)
# #             end = time.time()

# #             if backprop:
# #                 desc = f'Train {epoch}: '
# #             else:
# #                 desc = f'Evaluate {epoch}: '
# #             desc += f'Time {batch_time.avg:.3f}s '
# #             desc += f'(it:{batch_time.val:.3f}s) '
# #             desc += f'Data:{data_time.avg:.3f}s '
# #             desc += f'(it:{data_time.val:.3f}s) '
# #             desc += f'Loss {losses.avg:.4e} '
# #             t.set_description(desc)

# #     return losses.avg

# def pre_loop(model,backprop):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()

#     # switch to train mode
#     if backprop:
#         model.train()
#     else:
#         model.eval()

#     end = time.time()
#     return batch_time,data_time,losses,end

# def to_gpu(feats,labels):
#     feats = feats.cuda()
#     labels = labels.cuda()
#     return feats,labels

# def process(labels,targets,feats):
#     feats,labels=to_gpu(feats,labels)
#     labels=labels.float()
#     targets=targets.cuda().float()
#     feats=feats.unsqueeze(1)
#     return labels,targets,feats

# def post(backprop,optimizer,loss,batch_time,end,epoch,data_time,losses):
#     if backprop:
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # measure elapsed time
#     batch_time.update(time.time() - end)
#     end = time.time()

#     if backprop:
#         desc = f'Train {epoch}: '
#     else:
#         desc = f'Evaluate {epoch}: '
#     desc += f'Time {batch_time.avg:.3f}s '
#     desc += f'(it:{batch_time.val:.3f}s) '
#     desc += f'Data:{data_time.avg:.3f}s '
#     desc += f'(it:{data_time.val:.3f}s) '
#     desc += f'Loss {losses.avg:.4e} '
#     return desc

# def train_one_epoch(
#         dataloader,
#         model,
#         criterion,
#         optimizer,
#         gpu,
#         epoch,
#         backprop=False):
    
    
#     batch_time,data_time,losses,end = pre_loop(model,backprop)

#     with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
#         for i, (feats, labels)  in t:
            
#             # if cfg.GPU >= 0:
#             if gpu >=0:
#                 feats,labels=to_gpu(feats,labels)

#             output = model(feats)

#             # hand written NLL criterion
#             loss = criterion(labels,output)

#             # measure accuracy and record loss
#             losses.update(loss.item(), feats.size(0))

#             t.set_description(post(backprop,optimizer,loss,batch_time,end,epoch,data_time,losses))

#     return losses.avg

# def train_one_epoch_calf(
#         dataloader,
#         model,
#         criterion,
#         optimizer,
#         gpu,
#         epoch,
#         backprop=False):
    
#     batch_time,data_time,losses,end = pre_loop(model,backprop)

#     with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
#         for i, (feats, labels, targets)  in t:
            
#             # if cfg.GPU >= 0:
#             if gpu >=0:
#                 labels,targets,feats=process(labels,targets,feats)

#             output_segmentation, output_spotting = model(feats)

#             # hand written NLL criterion
#             loss = criterion([labels, targets], [output_segmentation, output_spotting])

#             # measure accuracy and record loss
#             losses.update(loss.item(), feats.size(0))

#             t.set_description(post(backprop,optimizer,loss,batch_time,end,epoch,data_time,losses))

#     return losses.avg