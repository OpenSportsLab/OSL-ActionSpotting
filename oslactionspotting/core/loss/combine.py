import torch


####################################################################################################################################################

# Combined loss function

####################################################################################################################################################


class Combined2x(torch.nn.Module):
    """Combination of two losses.

    Args:
        c_1 : The first criterion.
        c_2 : The second criterion.
        w_1 (float): Weight for the first criterion.
        w_2 (float): Weight for the second criterion.
    """

    def __init__(self, c_1, c_2, w_1, w_2):

        super(Combined2x, self).__init__()

        self.c_1 = c_1
        self.c_2 = c_2
        self.w_1 = w_1
        self.w_2 = w_2

    def forward(self, labels, output):
        """Forward function.

        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.

        Returns:
            torch.Tensor: The returned combined loss.
        """
        return self.w_1 * self.c_1(labels[0], output[0]) + self.w_2 * self.c_2(
            labels[1], output[1]
        )
