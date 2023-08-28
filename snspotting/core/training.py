import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


# from .evaluate import test

# def trainer(train_loader,
#             val_loader,
#             model,
#             optimizer,
#             scheduler,
#             criterion,
#             model_name,
#             max_epochs=1000,
#             evaluation_frequency=20):

#     # def train(self):
#     logging.info("start training")

#     best_loss = 9e99

#     for epoch in range(max_epochs):
#         best_model_path = os.path.join("outputs", model_name, "model.pth.tar")

#         # train for one epoch
#         loss_training = train_one_epoch(train_loader, model, criterion,
#                             optimizer, epoch + 1, backprop=True)

#         # evaluate on validation set
#         loss_validation = train_one_epoch(
#             val_loader, model, criterion, optimizer, epoch + 1, backprop=False)

#         state = {
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'best_loss': best_loss,
#             'optimizer': optimizer.state_dict(),
#         }
#         os.makedirs(os.path.join("outputs", model_name), exist_ok=True)

#         # remember best prec@1 and save checkpoint
#         is_better = loss_validation < best_loss
#         best_loss = min(loss_validation, best_loss)

#         # Save the best model based on loss only if the evaluation frequency too long
#         if is_better:
#             torch.save(state, best_model_path)

#         # Test the model on the validation set
#         if epoch % evaluation_frequency == 0 and epoch != 0:
#             performance_validation = test(
#                 val_loader,
#                 model,
#                 model_name)

#             logging.info("Validation performance at epoch " +
#                         str(epoch+1) + " -> " + str(performance_validation))

#         # Reduce LR on Plateau after patience reached
#         prevLR = optimizer.param_groups[0]['lr']
#         scheduler.step(loss_validation)
#         currLR = optimizer.param_groups[0]['lr']
#         if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
#             logging.info("Plateau Reached!")

#         if (prevLR < 2 * scheduler.eps and
#                 scheduler.num_bad_epochs >= scheduler.patience):
#             logging.info(
#                 "Plateau Reached and no more reduction -> Exiting Loop")
#             break

#     return


def train_one_epoch(dataloader,
        model,
        criterion,
        optimizer,
        epoch,
        backprop=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if backprop:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            # if cfg.GPU >= 0:
            feats = feats.cuda()
            labels = labels.cuda()

            # compute output
            output = model(feats)

            # hand written NLL criterion
            loss = criterion(labels, output)

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))

            if backprop:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if backprop:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg

