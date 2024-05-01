import torch


####################################################################################################################################################

# Combined loss function

####################################################################################################################################################

class Combined2x(torch.nn.Module):

    def __init__(self, c_1, c_2, w_1, w_2):

        super(Combined2x, self).__init__()

        self.c_1 = c_1
        self.c_2 = c_2
        self.w_1 = w_1
        self.w_2 = w_2

    def forward(self, gts, preds):
        return self.w_1 * self.c_1(gts[0], preds[0]) + self.w_2 * self.c_2(gts[1], preds[1])