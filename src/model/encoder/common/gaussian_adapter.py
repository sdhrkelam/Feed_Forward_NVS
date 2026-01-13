from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from src.geometry.projection import get_world_rays
from src.misc.sh_rotation import rotate_sh
from .gaussians import build_covariance

from ...types import Gaussians

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    use_marble_gaussians: bool = True


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.use_marble_gaussians = cfg.use_marble_gaussians

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        print("raw_gaussians dim:", raw_gaussians.shape[-1])
        print("adapter d_in:", self.d_in)
        print("use_marble:", self.cfg.use_marble_gaussians)

        #####################################changed_for_marbels###################################
        # --------------------------------------------------
        # 1. Split raw Gaussian parameters
        # --------------------------------------------------
        if self.cfg.use_marble_gaussians:
            # raw_gaussians = [radius, sh]
            radius, sh = raw_gaussians.split((1, 3 * self.d_sh), dim=-1)

            # isotropic scale in image space
            scales = F.softplus(radius).repeat_interleave(3, dim=-1)

            # identity rotation (no orientation)
            rotations = torch.zeros(
                (*scales.shape[:-1], 4),
                device=device,
                dtype=scales.dtype,
            )
            rotations[..., 0] = 1.0

        else:
            # raw_gaussians = [sx, sy, sz, qx, qy, qz, qw, sh]
            scales, rotations, sh = raw_gaussians.split(
                (3, 4, 3 * self.d_sh), dim=-1
            )

            scales = F.softplus(scales)
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # --------------------------------------------------
        # 2. Convert image-space scale → world-space scale
        # --------------------------------------------------
        h, w = image_shape
        pixel_size = 1.0 / torch.tensor(
            (w, h),
            dtype=scales.dtype,
            device=device,
        )

        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)

        # depth-aware metric scaling (THIS WAS MISSING)
        scales = scales * depths[..., None] * multiplier[..., None]

        # --------------------------------------------------
        # 3. Clamp / bound scales (existing logic)
        # --------------------------------------------------
        scales = scales.clamp(
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
        )

        # --------------------------------------------------
        # 4. Continue as usual
        # --------------------------------------------------
        covariances = build_covariance(scales, rotations)

        #####################################changed_for_marbels###################################

        # scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # # Map scale features to valid scale range.
        # scale_min = self.cfg.gaussian_scale_min
        # scale_max = self.cfg.gaussian_scale_max
        # scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        # h, w = image_shape
        # pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        # multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        # scales = scales * depths[..., None] * multiplier[..., None]

        # # Normalize the quaternion features to yield a valid quaternion.
        # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        # covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
        
    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    # @property
    # def d_in(self) -> int:
    #     return 7 + 3 * self.d_sh

    @property
    def d_in(self) -> int:
        print("use_marble_gaussians:",self.cfg.use_marble_gaussians)
        if self.cfg.use_marble_gaussians:
            return 1 + 3 * self.d_sh  # radius + SH
        return 7 + 3 * self.d_sh



class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        # levels: Float[Tensor, "*#batch"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        # scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # scales = 0.001 * F.softplus(scales)
        # scales = scales.clamp_max(0.3)
        
        # # Normalize the quaternion features to yield a valid quaternion.
        # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        if self.cfg.use_marble_gaussians:
            # raw_gaussians = [radius, SH]
            radius, sh = raw_gaussians.split((1, 3 * self.d_sh), dim=-1)

            scales = F.softplus(radius).repeat_interleave(3, dim=-1)

            rotations = torch.zeros(
                (*scales.shape[:-1], 4),
                device=raw_gaussians.device,
                dtype=raw_gaussians.dtype,
            )
            rotations[..., 0] = 1.0  # identity quaternion

        else:
            scales, rotations, sh = raw_gaussians.split(
                (3, 4, 3 * self.d_sh), dim=-1
            )

            scales = F.softplus(scales)
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

                
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        # print(scales.max())
        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means.float(),
            # levels=levels.int(),
            covariances=covariances.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )

class Unet3dGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        # scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # scales = 0.001 * F.softplus(scales)
        # scales = scales.clamp_max(0.3)
        
        # # Normalize the quaternion features to yield a valid quaternion.
        # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        if self.cfg.use_marble_gaussians:
            # raw_gaussians = [radius, SH]
            radius, sh = raw_gaussians.split((1, 3 * self.d_sh), dim=-1)

            scales = F.softplus(radius).repeat_interleave(3, dim=-1)

            rotations = torch.zeros(
                (*scales.shape[:-1], 4),
                device=raw_gaussians.device,
                dtype=raw_gaussians.dtype,
            )
            rotations[..., 0] = 1.0  # identity quaternion

        else:
            scales, rotations, sh = raw_gaussians.split(
                (3, 4, 3 * self.d_sh), dim=-1
            )

            scales = F.softplus(scales)
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

