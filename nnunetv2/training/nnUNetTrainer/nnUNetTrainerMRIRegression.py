from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch.nn as nn

class nnUNetTrainerMRIRegression(nnUNetTrainer):
    def build_loss(self):
        return nn.MSELoss()

    import torch

    def configure_network(self):
        torch._dynamo.config.suppress_errors = True  # avoid Dynamo errors
        torch._dynamo.disable()  # fully disables torch.compile
        super().configure_network()
        self.network.num_classes = 1
        self.network.final_nonlin = nn.Identity()

