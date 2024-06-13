"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import logging
from oslactionspotting.apis.inference.utils import infer_and_process_predictions_e2e
from oslactionspotting.core.optimizer.builder import build_optimizer
from oslactionspotting.core.scheduler.builder import build_scheduler

from oslactionspotting.core.utils.io import store_json, clear_files

from oslactionspotting.core.utils.lightning import CustomProgressBar, MyCallback
import pytorch_lightning as pl

import os

import torch

from oslactionspotting.datasets.builder import build_dataset
from abc import ABC, abstractmethod


def build_trainer(cfg, model=None, default_args=None):
    """Build a trainer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        model : The model that is used to train. Needed only if E2E method because training do not rely on pytorch lightning.
            Default: None.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        evaluator: The constructed trainer.
    """
    if cfg.type == "trainer_e2e":

        logging.info("Build optimizer")
        optimizer, scaler = build_optimizer(model._get_params(), cfg.optimizer)
        logging.info(optimizer)
        logging.info(scaler)
        logging.info("Build scheduler")
        lr_scheduler = build_scheduler(optimizer, cfg.scheduler, default_args)
        # optimizer, scaler = model.get_optimizer({'lr': cfg.learning_rate})

        # # Warmup schedule
        # num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        # num_epochs, lr_scheduler = get_lr_scheduler(
        #     cfg, optimizer, num_steps_per_epoch)
        trainer = Trainer_e2e(
            cfg,
            model,
            optimizer,
            scaler,
            lr_scheduler,
            default_args["work_dir"],
            default_args["dali"],
            default_args["repartitions"],
            default_args["cfg_test"],
            default_args["cfg_challenge"],
            default_args["cfg_valid_data_frames"],
        )
    else:
        trainer = Trainer_pl(cfg, default_args["work_dir"])

    return trainer


class Trainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass


class Trainer_pl(Trainer):
    """Trainer class used for models that rely on lightning modules.

    Args:
        cfg (dict): Dict config. It should contain the key 'max_epochs' and the key 'GPU'.
    """

    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        call = MyCallback()
        self.trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            devices=[cfg.GPU],
            callbacks=[call, CustomProgressBar(refresh_rate=1)],
            num_sanity_val_steps=0,
        )

    def train(self, **kwargs):
        self.trainer.fit(**kwargs)

        best_model = kwargs["model"].best_state

        logging.info("Done training")
        logging.info("Best epoch: {}".format(best_model.get("epoch")))
        torch.save(best_model, os.path.join(self.work_dir, "model.pth.tar"))

        logging.info("Model saved")
        logging.info(os.path.join(self.work_dir, "model.pth.tar"))


class Trainer_e2e(Trainer):
    """Trainer class used for the e2e model.

    Args:
        args (dict): Dict of config.
        model.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters. Set to None if validation epoch.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        lr_scheduler : The learning rate scheduler.
        work_dir (string): The folder in which the different files will be saved.
        dali (bool): Whether videos are processed with dali or opencv.
        repartitions (List[int]): List of gpus used data processing.
            Default: None.
        cfg_test (dict): Dict of config for the inference (testing purpose) and evaluation of the test split. Occurs once training is done.
            Default: None.
        cfg_challenge (dict): Dict of config for the inference (testing purpose) of the challenge split. Occurs once training is done.
            Default: None.
        cfg_valid_data_frames (dict): Dict of config for the inference (testing purpose) and evaluation of the valid split. Occurs through the epochs after a certain number of epochs only if the criterion for the valid split is 'map'.
            Default: None.
    """

    def __init__(
        self,
        args,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        work_dir,
        dali,
        repartitions=None,
        cfg_test=None,
        cfg_challenge=None,
        cfg_valid_data_frames=None,
    ):
        self.losses = []
        self.best_epoch = 0
        self.best_criterion_valid = 0 if args.criterion_valid == "map" else float("inf")

        self.num_epochs = args.num_epochs
        self.epoch = 0
        self.model = model

        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        self.acc_grad_iter = args.acc_grad_iter

        self.start_valid_epoch = args.start_valid_epoch
        self.criterion_valid = args.criterion_valid
        self.save_dir = work_dir
        self.dali = dali

        self.repartitions = repartitions
        self.cfg_test = cfg_test
        self.cfg_challenge = cfg_challenge
        self.cfg_valid_data_frames = cfg_valid_data_frames

    def train(self, train_loader, valid_loader, classes):
        """
        Args:
            train_loader : Data loader for the train split.
            valid_loader : Data loader for the valid split.
            classes (dict): Dictionnary with classes associated to indexes.
        """
        if self.criterion_valid == "map":
            dataset_Valid_Frames = build_dataset(
                self.cfg_valid_data_frames,
                None,
                {"repartitions": self.repartitions, "classes": classes},
            )
        for epoch in range(self.epoch, self.num_epochs):
            # train_loss = 0.0
            # valid_loss = 0.0
            train_loss = self.model.epoch(
                train_loader,
                self.dali,
                self.optimizer,
                self.scaler,
                lr_scheduler=self.lr_scheduler,
                acc_grad_iter=self.acc_grad_iter,
            )

            valid_loss = self.model.epoch(
                valid_loader, self.dali, acc_grad_iter=self.acc_grad_iter
            )
            print(
                "[Epoch {}] Train loss: {:0.5f} Valid loss: {:0.5f}".format(
                    epoch, train_loss, valid_loss
                )
            )

            valid_mAP = 0
            # self.start_valid_epoch = 0
            if self.criterion_valid == "loss":
                if valid_loss < self.best_criterion_valid:
                    self.best_criterion_valid = valid_loss
                    self.best_epoch = epoch
                    print("New best epoch!")
            elif self.criterion_valid == "map":
                if epoch >= self.start_valid_epoch:
                    pred_file = None
                    if self.save_dir is not None:
                        pred_file = os.path.join(
                            self.save_dir, "pred-valid.{}".format(epoch)
                        )
                        os.makedirs(self.save_dir, exist_ok=True)
                    valid_mAP = infer_and_process_predictions_e2e(
                        self.model,
                        self.dali,
                        dataset_Valid_Frames,
                        "VALID",
                        classes,
                        pred_file,
                        dataloader_params=self.cfg_valid_data_frames.dataloader,
                    )
                    if valid_mAP > self.best_criterion_valid:
                        self.best_criterion_valid = valid_mAP
                        self.best_epoch = epoch
                        print("New best epoch!")
            else:
                print("Unknown criterion:", self.criterion_valid)

            self.losses.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "valid": valid_loss,
                    "valid_mAP": valid_mAP,
                }
            )
            if self.save_dir is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                store_json(
                    os.path.join(self.save_dir, "loss.json"), self.losses, pretty=True
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, "checkpoint_{:03d}.pt".format(epoch)),
                )

                logging.info("Checkpoint saved")
                logging.info(
                    os.path.join(self.save_dir, "checkpoint_{:03d}.pt".format(epoch))
                )

                clear_files(self.save_dir, r"optim_\d+\.pt")
                torch.save(
                    {
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "lr_state_dict": self.lr_scheduler.state_dict(),
                    },
                    os.path.join(self.save_dir, "optim_{:03d}.pt".format(epoch)),
                )
                logging.info("Optim saved")
                logging.info(
                    os.path.join(self.save_dir, "optim_{:03d}.pt".format(epoch))
                )
            # break
        logging.info("Best epoch: {}".format(self.best_epoch))

        if self.dali:
            train_loader.delete()
            valid_loader.delete()
            if self.criterion_valid == "map":
                dataset_Valid_Frames.delete()

        if self.save_dir is not None:
            self.model.load(
                torch.load(
                    os.path.join(
                        self.save_dir, "checkpoint_{:03d}.pt".format(self.best_epoch)
                    )
                )
            )

            logging.info("Checkpoint of best epoch loaded")
            logging.info(
                os.path.join(
                    self.save_dir, "checkpoint_{:03d}.pt".format(self.best_epoch)
                )
            )

            # Evaluate on valid if not already done
            eval_splits = ["valid"] if self.criterion_valid != "map" else []

            # Evaluate on hold out splits
            eval_splits += ["test", "challenge"]
            for split in eval_splits:
                if split == "valid":
                    cfg_tmp = self.cfg_valid_data_frames
                if split == "test":
                    cfg_tmp = self.cfg_test
                elif split == "challenge":
                    cfg_tmp = self.cfg_challenge
                split_path = os.path.join(cfg_tmp.path)

                if os.path.exists(split_path):
                    split_data = build_dataset(
                        cfg_tmp,
                        None,
                        {"repartitions": self.repartitions, "classes": classes},
                    )
                    split_data.print_info()

                    pred_file = None
                    if self.save_dir is not None:
                        pred_file = os.path.join(
                            self.save_dir, "pred-{}.{}".format(split, self.best_epoch)
                        )

                    infer_and_process_predictions_e2e(
                        self.model,
                        self.dali,
                        split_data,
                        split.upper(),
                        classes,
                        pred_file,
                        calc_stats=split != "challenge",
                        dataloader_params=cfg_tmp.dataloader,
                    )

                    if self.dali:
                        split_data.delete()

        logging.info("Done training")
        logging.info(self.best_epoch)
