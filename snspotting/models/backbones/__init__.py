import torch

def build_backbone(cfg, default_args=None):
    """Build a backbone from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        backbone: The constructed backbone.
    """
    if cfg.type == "PreExtactedFeatures":
        backbone = PreExtactedFeatures(
                feature_dim=cfg.feature_dim,
                output_dim=cfg.output_dim)
    else:
        backbone = None 

    return backbone




class PreExtactedFeatures(torch.nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(PreExtactedFeatures, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.reduceDim = not self.feature_dim == self.output_dim
        if self.reduceDim:
            self.feature_extractor = torch.nn.Linear(self.feature_dim, self.output_dim)


    def forward(self, inputs):
        BS, FR, IC = inputs.shape
        if self.reduceDim:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = self.feature_extractor(inputs)
            inputs = inputs.reshape(BS, FR, -1)
        return inputs

        
