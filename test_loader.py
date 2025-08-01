# test_dataloader_speed.py

import time
from nnunetv2.run.run_training import get_trainer_from_args

# 1) Build & initialize your trainer
trainer = get_trainer_from_args(
    "555",                       # dataset ID
    "3d_fullres",                # configuration
    0,                           # fold
    "nnUNetTrainerMRIRegression" # custom trainer class
)
trainer.initialize()            # sets up data loaders, model, etc.

# 2) Grab the DataLoader and wrap it in an iterator
dataloader = trainer.dataloader_train
data_iter = iter(dataloader)

# 3) Time one batch pull
start = time.time()
batch = next(data_iter)
elapsed = time.time() - start

print(f"✔ One batch load took {elapsed:.3f} seconds")
