from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Register additional trainer variants here
try:
    from nnunetv2.training.nnUNetTrainerVariants.Regression.nnUNetRegressor import nnUNetRegressor
except ImportError:
    pass
