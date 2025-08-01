# test_dataloader_speed.py

import time
import collections.abc
from nnunetv2.run.run_training import get_trainer_from_args

# 1) Build & initialize your trainer (positional args only)
trainer = get_trainer_from_args(
    "555",                       # dataset ID
    "3d_fullres",                # configuration
    0,                           # fold
    "nnUNetTrainerMRIRegression" # your custom trainer class
)
trainer.initialize()            # sets up everything, including the loader

# 2) Auto-discover the loader attribute
loader_attr = None
for name, val in vars(trainer).items():
    is_loader = isinstance(val, collections.abc.Iterator) or hasattr(val, "__iter__")
    if is_loader and any(k in name.lower() for k in ("loader", "dataloader", "generator")):
        loader_attr = name
        break

if loader_attr is None:
    raise RuntimeError(
        "No data-loader attribute found. Available attributes:\n"
        f"{list(vars(trainer).keys())}"
    )

loader = getattr(trainer, loader_attr)
print(f"⏳ Using data-loader attribute: {loader_attr}")

# 3) Time one batch pull
start = time.time()
batch = next(loader)
print(f"✔ One batch load took {time.time() - start:.3f} seconds")
