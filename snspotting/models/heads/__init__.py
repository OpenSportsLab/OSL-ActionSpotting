import torch 

def build_head(cfg, default_args=None):
    """Build a head from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        head: The constructed head.
    """
    if cfg.type == "LinearLayer":
        head = LinearLayerHead(input_dim=cfg.input_dim, 
                            output_dim=cfg.num_classes+1)
    else:
        head = None 

    return head



class LinearLayerHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayerHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.drop = torch.nn.Dropout(p=0.4)
        self.head = torch.nn.Linear(input_dim, output_dim)
        self.sigm = torch.nn.Sigmoid()
        

    def forward(self, inputs):
        return self.sigm(self.head(self.drop(inputs)))
        

        
