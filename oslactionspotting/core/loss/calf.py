import torch


####################################################################################################################################################

# Context-aware loss function

####################################################################################################################################################


class ContextAwareLoss(torch.nn.Module):
    """Context Aware Loss.

    Args:
        K (list[list[int]]): Config dict. It should at least contain the key "type".
        framerate (int): Framerate at which the features have been extracted.
            Default: 2.
        hit_radius (float): The hit radius.
            Default: 0.1.
        miss_radius (float): The miss radius.
            Default: 0.9.

    """

    def __init__(self, K, framerate=2, hit_radius=0.1, miss_radius=0.9):

        super(ContextAwareLoss, self).__init__()

        self.K = torch.FloatTensor(K * framerate).cuda()
        self.hit_radius = float(hit_radius)
        self.miss_radius = float(miss_radius)

    def forward(self, labels, output):
        """Forward function.

        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.

        Returns:
            torch.Tensor: The returned loss.
        """
        K = self.K
        hit_radius = self.hit_radius
        miss_radius = self.miss_radius

        zeros = torch.zeros(output.size()).to(output.device).type(torch.float)
        output = 1.0 - output

        case1 = self.DownStep(labels, K[0]) * torch.max(
            zeros, -torch.log(output) + torch.log(zeros + miss_radius)
        )
        case2 = self.Interval(labels, K[0], K[1]) * torch.max(
            zeros,
            -torch.log(
                output
                + (1.0 - output)
                * (self.PartialIdentity(labels, K[0], K[1]) - K[0])
                / (K[1] - K[0])
            )
            + torch.log(zeros + miss_radius),
        )
        case3 = self.Interval(labels, K[1], 0.0) * zeros
        case4 = self.Interval(labels, 0.0, K[2]) * torch.max(
            zeros,
            -torch.log(
                1.0
                - output
                + output
                * (self.PartialIdentity(labels, 0.0, K[2]) - 0.0)
                / (K[2] - 0.0)
            )
            + torch.log(zeros + 1.0 - hit_radius),
        )
        case5 = self.Interval(labels, K[2], K[3]) * torch.max(
            zeros,
            -torch.log(
                output
                + (1.0 - output)
                * (self.PartialIdentity(labels, K[2], K[3]) - K[3])
                / (K[2] - K[3])
            )
            + torch.log(zeros + miss_radius),
        )
        case6 = self.UpStep(labels, K[3]) * torch.max(
            zeros, -torch.log(output) + torch.log(zeros + miss_radius)
        )

        L = case1 + case2 + case3 + case4 + case5 + case6

        return torch.sum(L)

    def UpStep(self, x, a):
        """
        Args :
            x (torch.Tensor).
            a (torch.Tensor).

        Returns:
            0 if x<a, 1 if x >= a
        """

        return 1.0 - torch.max(0.0 * x, torch.sign(a - x))

    def DownStep(self, x, a):
        """
        Args :
            x (torch.Tensor).
            a (torch.Tensor).

        Returns:
            1 if x < a, 0 if x >=a
        """

        return torch.max(0.0 * x, torch.sign(a - x))

    def Interval(self, x, a, b):
        """
        Args :
            x (torch.Tensor).
            a (torch.Tensor).
            b (torch.Tensor).

        Returns:
            1 if a<= x < b, 0 otherwise
        """

        return self.UpStep(x, a) * self.DownStep(x, b)

    def PartialIdentity(self, x, a, b):
        """
        Args :
            x (torch.Tensor).
            a (torch.Tensor).
            b (torch.Tensor).

        Returns:
            a if x<a, x if a<= x <b, b if x >= b
        """

        return torch.min(torch.max(x, 0.0 * x + a), 0.0 * x + b)


####################################################################################################################################################

# Spotting loss

####################################################################################################################################################


class SpottingLoss(torch.nn.Module):
    """Spotting loss.

    Args:
        lambda_coord (float).
        lambda_noobj (float).

    """

    def __init__(self, lambda_coord, lambda_noobj):
        super(SpottingLoss, self).__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, labels, output):
        """Forward function.

        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.

        Returns:
            torch.Tensor: The returned spotting loss.
        """
        output = self.permute_output_for_matching(labels, output)
        loss = torch.sum(
            labels[:, :, 0]
            * self.lambda_coord
            * torch.square(labels[:, :, 1] - output[:, :, 1])
            + labels[:, :, 0] * torch.square(labels[:, :, 0] - output[:, :, 0])
            + (1 - labels[:, :, 0])
            * self.lambda_noobj
            * torch.square(labels[:, :, 0] - output[:, :, 0])
            + labels[:, :, 0]
            * torch.sum(torch.square(labels[:, :, 2:] - output[:, :, 2:]), axis=-1)
        )  # -labels[:,:,0]*torch.sum(labels[:,:,2:]*torch.log(output[:,:,2:]),axis=-1)
        return loss

    def permute_output_for_matching(self, labels, output):
        """
        Args:
            labels (torch.Tensor): The ground truth labels.
            output (torch.Tensor): The predictions.

        Returns:
            torch.Tensor: The permuted pred.
        """
        alpha = labels[:, :, 0]
        x = labels[:, :, 1]
        p = output[:, :, 1]
        nb_pred = x.shape[-1]

        D = torch.abs(
            x.unsqueeze(-1).repeat(1, 1, nb_pred)
            - p.unsqueeze(-2).repeat(1, nb_pred, 1)
        )
        D1 = 1 - D
        Permut = 0 * D

        alpha_filter = alpha.unsqueeze(-1).repeat(1, 1, nb_pred)

        v_filter = alpha_filter
        h_filter = 0 * v_filter + 1
        D2 = v_filter * D1

        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2, axis=-1), nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B, axis=-2), nb_pred).permute(
                0, 2, 1
            )
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1 - torch.sum(Permut, axis=-1)) * alpha
            v_filter = v_filter.unsqueeze(-1).repeat(1, 1, nb_pred)
            h_filter = 1 - torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1, nb_pred, 1)

        v_filter = 1 - alpha_filter
        D2 = v_filter * D1
        D2 = h_filter * D2

        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2, axis=-1), nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B, axis=-2), nb_pred).permute(
                0, 2, 1
            )
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1 - torch.sum(Permut, axis=-1)) * (
                1 - alpha
            )  # here comes the change
            v_filter = v_filter.unsqueeze(-1).repeat(1, 1, nb_pred)
            h_filter = 1 - torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1, nb_pred, 1)

        permutation = torch.argmax(Permut, axis=-1)
        permuted = torch.gather(
            output, 1, permutation.unsqueeze(-1).repeat(1, 1, labels.shape[-1])
        )

        return permuted
