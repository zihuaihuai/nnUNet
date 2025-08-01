# test_minimal_batch.py

import time
import torch
from torch.cuda import synchronize
from nnunetv2.run.run_training import get_trainer_from_args

# 1) Build & init trainer
trainer = get_trainer_from_args(
    "555",                       # dataset ID
    "3d_fullres",                # configuration
    0,                           # fold
    "nnUNetTrainerMRIRegression" # custom trainer class name
)
trainer.initialize()
trainer.on_train_start()  # sets up trainer.dataloader_train

# 2) Grab one batch
loader = trainer.dataloader_train
batch = next(iter(loader))
data, target = batch['data'], batch['target']

# Move everything to device once (so we’re timing pure compute, not the first-to-GPU paywall)
data = data.to(trainer.device, non_blocking=True)
if isinstance(target, list):
    target = [t.to(trainer.device, non_blocking=True) for t in target]
else:
    target = target.to(trainer.device, non_blocking=True)

net = trainer.network
opt = trainer.optimizer
loss_fn = trainer.loss
scaler = trainer.grad_scaler

# Helper to time on GPU
def timed(step_name, fn):
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    print(f"{step_name:15s}: {(time.time()-t0)*1000:.1f} ms")

# 3) Run timings
timed("Forward",    lambda: net(data))
timed("Backward",   lambda: loss_fn(net(data), target).backward() if scaler is None else scaler.scale(loss_fn(net(data), target)).backward())
timed("Optimizer",  lambda: (scaler.step(opt), scaler.update(), opt.zero_grad()) if scaler is not None else (opt.step(), opt.zero_grad()))

# 4) Clean up loader threads
if hasattr(loader, "_finish"):
    loader._finish()
