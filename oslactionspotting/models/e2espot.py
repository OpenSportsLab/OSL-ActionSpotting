from oslactionspotting.core.loss.builder import build_criterion
from oslactionspotting.models.backbones import build_backbone
from oslactionspotting.models.common import step, BaseRGBModel
from oslactionspotting.models.heads import build_head
from contextlib import nullcontext
import torch
from tqdm import tqdm
import torch.nn as nn


class E2EModel(BaseRGBModel):
    """Class used to handle the model for the E2E method. This class is not the model but contains the model as a variable.
    This class is used to initialize the model, perform an epoch and predict.
    Args:
        cfg (dict): Dict of config.
        num_classes (int): Number of classes.
        backbone (dict): Dict used to build the backbone.
        head (dict): Dict used to build the head.
        clip_len (int): the length of the clip.
        device (string): The device to use.
            Default :'cuda'.
        multi_gpu: Whether to use one gpu or more for the model.
            Default: False.
    """

    class Impl(nn.Module):
        """Model for the E2E method. Model is build of a backbone and a head.

        Args:
            num_classes (int): Number of classes.
            backbone (dict): Dict used to build the backbone.
            head (dict): Dict used to build the head.
            clip_len (int): the length of the clip.
        """

        def __init__(self, num_classes, backbone, head, clip_len, modality):
            super().__init__()
            is_rgb = modality == "rgb"
            in_channels = {"flow": 2, "bw": 1, "rgb": 3}[modality]

            backbone.clip_len = clip_len
            backbone.is_rgb = is_rgb
            backbone.in_channels = in_channels

            self.backbone = build_backbone(backbone)

            head.num_classes = num_classes
            head.feat_dim = self.backbone._feat_dim

            self.head = build_head(head)

        def forward(self, x):
            im_feat = self.backbone(x)
            return self.head(im_feat)

        def print_stats(self):
            print("Model params:", sum(p.numel() for p in self.parameters()))
            print(
                "  CNN features:",
                sum(p.numel() for p in self.backbone._features.parameters()),
            )
            print(
                "  Temporal:", sum(p.numel() for p in self.head._pred_fine.parameters())
            )

    def __init__(
        self,
        cfg,
        num_classes,
        backbone,
        head,
        clip_len,
        modality,
        device="cuda",
        multi_gpu=False,
    ):

        last_gpu_index = torch.cuda.device_count() - 1

        # self.device = torch.device('cuda:{}'.format(0))
        self.device = device
        self._multi_gpu = multi_gpu
        self._model = E2EModel.Impl(num_classes, backbone, head, clip_len, modality)
        self._model.print_stats()
        self.criterion = build_criterion(cfg.training.criterion)
        if multi_gpu:
            self._model = nn.DataParallel(self._model)
            self._model.to(device)
            # self._model = nn.DataParallel(self._model,device_ids = [1, 0, 2, 3] if torch.cuda.device_count()==4 else [1, 0])
            # self._model.to(f'cuda:{self._model.device_ids[0]}')
            # self.device = torch.device('cuda:{}'.format(1))
        else:
            self._model.to(device)

        self._multi_gpu = multi_gpu
        self._num_classes = num_classes

    def epoch(
        self,
        loader,
        dali,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
        acc_grad_iter=1,
        fg_weight=5,
    ):
        """Performs an epoch for training/validating.

        Args:
            loader: The dataloader.
            dali (bool): Whether dali has been used or opencv for data processing.
            optimizer (torch.optim.Optimizer): The optimizer to update model parameters. Set to None if validation epoch.
                Default: None.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
                Default: None.
            lr_scheduler : The learning rate scheduler.
                Default: None.
            acc_grad_iter (int): Use gradient accumulation.
                Default: 1.
            fg_weight (int): Used to build tensor of weights for the criterion.
                Default: 5.
        Returns:
            (float): The average loss over the batches.
        """
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            if self._multi_gpu:
                ce_kwargs["weight"] = torch.FloatTensor(
                    [1] + [fg_weight] * (self._num_classes - 1)
                ).to(self.device)
            else:
                ce_kwargs["weight"] = torch.FloatTensor(
                    [1] + [fg_weight] * (self._num_classes - 1)
                ).to(torch.device("cuda:{}".format(1)))
        epoch_loss = 0.0

        times = []
        import timeit

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                if dali:
                    frame = batch["frame"].to(self.device)
                    label = batch["label"].to(
                        self.device
                        if self._multi_gpu
                        else torch.device("cuda:{}".format(1))
                    )
                else:
                    frame = loader.dataset.load_frame_gpu(batch, self.device)
                    label = batch["label"].to(self.device)

                label = (
                    label.flatten()
                    if len(label.shape) == 2
                    else label.view(-1, label.shape[-1])
                )

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)

                    pred = (
                        pred.to(self.device)
                        if self._multi_gpu
                        else pred.to(torch.device("cuda:{}".format(1)))
                    )

                    loss = 0.0
                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    # label=label.to(self.device)

                    for i in range(pred.shape[0]):
                        loss += self.criterion(
                            label, pred[i].reshape(-1, self._num_classes), **ce_kwargs
                        )
                        # loss += F.cross_entropy(
                        #     pred[i].reshape(-1, self._num_classes), label,
                        #     **ce_kwargs)

                if optimizer is not None:
                    step(
                        optimizer,
                        scaler,
                        loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                    )

                epoch_loss += loss.detach().item()

        print(epoch_loss, len(loader), epoch_loss / len(loader))
        return epoch_loss / len(loader)  # Avg loss

    def predict(self, seq, use_amp=True):
        """Perform prediction on the input.

        Args:
            seq (torch.tensor): The input.
            use_amp (bool): Whether to use automatic precision.
                Default: True.
        Returns:
            pred_cls (numpy.ndarray): Predicted class indices.
            pred (numpy.ndarray): Predicted probabilities.
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()
