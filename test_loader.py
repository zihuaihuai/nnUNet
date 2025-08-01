# test_dataloader_speed.py

import time
from nnunetv2.run.run_training import get_trainer_from_args

# 1) Instantiate & initialize trainer
trainer = get_trainer_from_args(
    "555",                       # dataset ID
    "3d_fullres",                # configuration
    0,                           # fold
    "nnUNetTrainerMRIRegression" # trainer class name
)
trainer.initialize()

# 2) Build the dataloaders (this returns the augmenters)
train_loader, val_loader = trainer.get_dataloaders()

# 3) Time one batch load + augmentation
start = time.time()
batch = next(train_loader)
elapsed = time.time() - start
print(f"✔ One batch (data + augmentation) took {elapsed:.3f} seconds")

# 4) Cleanly stop the loader threads
if hasattr(train_loader, "_finish"):
    train_loader._finish()
if hasattr(val_loader, "_finish"):
    val_loader._finish()
