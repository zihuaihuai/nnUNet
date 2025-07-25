# nnUNetTrainerMRIRegression.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np

print("✅ Custom trainer nnUNetTrainerMRIRegression loaded")

class nnUNetTrainerMRIRegression(nnUNetTrainer):
    """
    nnU-Net as voxel-wise regression (e.g. 3T -> 7T intensities).
    Changes:
      • MSE loss
      • Single output channel
      • Deep supervision OFF
      • Float targets (no class_locations needed)
      • Custom val logging (no Dice)
    """

    def initialize(self):
        # disable DS before parent init so loss/net are built accordingly
        self.enable_deep_supervision = False
        super().initialize()

        self.batch_dice = False
        self.loss = nn.MSELoss()

        # (Optional) reduce batch to dodge OOM
        self.batch_size = min(self.batch_size, 1)

        # ---- enforce 1 output channel ----
        net = self.network.module if self.is_ddp else self.network
        try:
            # PlainConvUNet path
            in_c = net.decoder.seg_outputs[0].in_channels
            net.decoder.seg_outputs = torch.nn.ModuleList([nn.Conv3d(in_c, 1, 1, bias=True)])
        except Exception:
            # fallback
            if hasattr(net, "seg_output_layer"):
                in_c = net.seg_output_layer.in_channels
                net.seg_output_layer = nn.Conv3d(in_c, 1, 1, bias=True)

        print("✅ Initialized (deep_supervision=False, out_ch=1, MSE loss)")

    # -------- TRAIN --------
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)

        target = batch['target']
        if isinstance(target, (list, tuple)):
            target = target[0]
        target = target.to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = self.device.type == 'cuda'
        with torch.autocast(self.device.type, enabled=use_amp):
            out = self.network(data)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out[:, :1].float()  # make sure single channel & float
            loss = self.loss(out, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}

    # -------- VAL --------
    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)

        target = batch['target']
        if isinstance(target, (list, tuple)):
            target = target[0]
        target = target.to(self.device, non_blocking=True).float()

        use_amp = self.device.type == 'cuda'
        with torch.autocast(self.device.type, enabled=use_amp):
            out = self.network(data)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out[:, :1].float()
            loss = self.loss(out, target)

        return {'loss': loss.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs):
        # collate losses only
        losses = [v['loss'] for v in val_outputs]
        loss_here = float(np.mean(losses))
        self.logger.log('val_losses', loss_here, self.current_epoch)
        # keep compatibility with base class expectations
        self.logger.log('dice_per_class_or_region', [np.nan], self.current_epoch)
        self.logger.log('mean_fg_dice', np.nan, self.current_epoch)

