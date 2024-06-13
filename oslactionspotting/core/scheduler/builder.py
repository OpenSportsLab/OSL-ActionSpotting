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
import torch
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import logging


def build_scheduler(optimizer, cfg, default_args=None):
    """Build a scheduler from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        scheduler: The constructed scheduler.
    """
    if cfg.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg.mode, verbose=cfg.verbose, patience=cfg.patience
        )
        logging.info("Using ReduceLROnPlateau")
    elif cfg.type == "ChainedSchedulerE2E":
        # Warmup schedule
        num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        cosine_epochs = cfg.num_epochs - cfg.warm_up_epochs
        scheduler = ChainedScheduler(
            [
                LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=cfg.warm_up_epochs * num_steps_per_epoch,
                ),
                CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs),
            ]
        )
        logging.info(
            "Using Linear Warmup ({}) + Cosine Annealing LR ({})".format(
                cfg.warm_up_epochs, cosine_epochs
            )
        )
    else:
        scheduler = None
    return scheduler
