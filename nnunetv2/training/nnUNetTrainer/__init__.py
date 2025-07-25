from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# Register additional trainer variants here
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMRIRegression import nnUNetTrainerMRIRegression
except ImportError as e:
    print(f"❌ Could not import nnUNetTrainerMRIRegression: {e}")


