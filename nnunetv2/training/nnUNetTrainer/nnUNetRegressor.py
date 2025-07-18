from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class nnUNetRegressor(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device
        )
        self.label_dtype = torch.float32

    def compute_loss(self, output, target):
        return {'loss': torch.nn.functional.mse_loss(output, target)}
