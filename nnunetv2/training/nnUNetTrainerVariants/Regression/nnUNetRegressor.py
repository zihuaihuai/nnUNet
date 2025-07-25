from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn

class nnUNetRegressor(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.label_dtype = torch.float32
        self.loss = nn.MSELoss()  # override the segmentation loss with regression loss

    def compute_loss(self, output, target):
        return {'loss': self.loss(output, target)}

