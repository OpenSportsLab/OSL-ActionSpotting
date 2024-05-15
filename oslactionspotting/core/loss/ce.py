import torch
import torch.nn.functional as F


class CELoss(torch.nn.Module):
    """Cross Entropy Loss.
    """

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, labels, output, **ce_kwargs):
        """Forward function.

        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.
            ce_kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        return F.cross_entropy(
            output, labels,
            **ce_kwargs)
