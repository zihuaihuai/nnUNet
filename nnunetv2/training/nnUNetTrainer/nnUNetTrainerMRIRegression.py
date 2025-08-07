# nnUNetTrainerMRIRegression.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

class UnsqueezeSeg(BasicTransform):
    """Turn a 3D seg volume into (1,X,Y,Z) so SpatialTransform won't one-hot it."""
    def __call__(self, image=None, segmentation=None, **kwargs):
        # only unsqueeze if it's missing the channel dimension
        if isinstance(segmentation, np.ndarray) and segmentation.ndim == image.ndim:
            segmentation = segmentation[np.newaxis, ...]
        return {"image": image, "segmentation": segmentation}


print("✅ Custom trainer nnUNetTrainerMRIRegression loaded")

class nnUNetTrainerMRIRegression(nnUNetTrainer):
    """
    Train nnU‑Net as a voxel‑wise regression model (e.g. 3T → 7T intensity mapping).
    Key changes:
      - MSE loss
      - single output channel
      - no deep supervision
      - only spatial + mirror augmentations (applied to both x and y)
    """

    def initialize(self):
        # turn off DS so network / loss build with single head
        self.enable_deep_supervision = False

        super().initialize()                # builds network, optimizer, loss, etc.
        self.batch_dice = False             # irrelevant for regression
        self.loss = nn.MSELoss()            # voxel‑wise MSE

        # optional: reduce batch if you need to avoid OOM
        self.batch_size = 1

        # swap final head to 1 channel
        net = self.network.module if self.is_ddp else self.network
        # most default nets use decoder.seg_outputs
        if hasattr(net.decoder, "seg_outputs"):
            in_c = net.decoder.seg_outputs[0].in_channels
            net.decoder.seg_outputs = torch.nn.ModuleList([nn.Conv3d(in_c, 1, 1, bias=True)])
        elif hasattr(net, "seg_output_layer"):
            in_c = net.seg_output_layer.in_channels
            net.seg_output_layer = nn.Conv3d(in_c, 1, 1, bias=True)

        print("✅ Initialized (deep_supervision=False, out_ch=1, MSE loss)")

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)

        # pull out the target and cast to float
        target = batch['target']
        if isinstance(target, (list, tuple)):
            target = target[0]
        target = target.to(self.device, non_blocking=True).float()   # <<< add .float()

        self.optimizer.zero_grad(set_to_none=True)
        use_amp = (self.device.type == 'cuda')
        with torch.autocast(self.device.type, enabled=use_amp):
            out = self.network(data)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out[:, :1]  # ensure single channel
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


    @staticmethod
    def get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales,
        mirror_axes, do_dummy_2d_data_aug, **kwargs
    ):
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = []
        # 1) Make sure seg has a channel axis
        transforms.append(UnsqueezeSeg())

        # 2) Your spatial + mirror as before
        transforms.append(SpatialTransform(
            patch_size,
            patch_center_dist_from_border=0,
            random_crop=True,
            p_elastic_deform=0.0,
            p_rotation=0.2,
            rotation=rotation_for_DA,
            p_scaling=0.2,
            scaling=(0.9, 1.1),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
            order_seg=3,  # Use high-quality trilinear interpolation for the target
            border_mode_seg='constant', # Pad with a constant value
            border_cval_seg=0 # The value to use for padding
        ))
        if mirror_axes:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        return ComposeTransforms(transforms)


    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales,
        **kwargs
    ) -> BasicTransform:
        # no augmentation at validation time
        return ComposeTransforms([])


    def configure_loss(self):
        # override base if needed elsewhere
        return nn.MSELoss()

