# test_dataloader_speed.py

import time
from nnunetv2.run.run_training import get_trainer_from_args

# 1) Instantiate your trainer (positional args only)
trainer = get_trainer_from_args(
    "555",                       # your dataset ID
    "3d_fullres",                # configuration
    0,                           # fold
    "nnUNetTrainerMRIRegression" # your trainer class name
)

# 2) Build network, optimizer, etc.
trainer.initialize()

# 3) Set up the dataloaders
trainer.on_train_start()  # <-- this is what actually sets trainer.dataloader_train

# 4) Grab the DataLoader and time one batch
loader = trainer.dataloader_train
start = time.time()
batch = next(iter(loader))
elapsed = time.time() - start

print(f"✔ One batch (data + augmentation) took {elapsed:.3f} seconds")
