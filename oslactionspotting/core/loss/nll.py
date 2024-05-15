import torch


class NLLLoss(torch.nn.Module):
    """Negative Log LikeLihood Loss.
    """

    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        """Forward function.

        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.

        Returns:
            torch.Tensor: The returned negative log likelihood loss.
        """
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))
