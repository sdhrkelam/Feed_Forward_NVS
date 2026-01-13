import os
from pathlib import Path
import sys
import json
import gzip
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from einops import rearrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from misc.image_io import save_image, save_interpolated_video
from src.utils.image import process_image
from src.misc.utils import vis_depth_map


from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.ply_export import export_ply

def setup_args():
    """Set up command-line arguments for the eval NVS script."""
    parser = argparse.ArgumentParser(description='Test AnySplat on NVS evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NVS dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--llffhold', type=int, default=8, help='LLFF holdout')
    parser.add_argument('--ctx_indices', type=str, default=None, help='Context indices (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--tgt_indices', type=str, default=None, help='Target indices (comma-separated, e.g., "4,5,6,7")')
    parser.add_argument('--output_path', type=str, default="outputs/nvs", help='Path to output directory')
    return parser.parse_args()

def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image, image)
    return psnr, ssim, lpips

def evaluate(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint:
        # Load from your checkpoint
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Initialize model architecture first
        model = AnySplat.from_pretrained("lhjiang/anysplat")
        
        # Load state dict from checkpoint
        if 'state_dict' in checkpoint:
            # Remove 'model.' prefix if it exists (Lightning saves with this prefix)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'model.' prefix
                key = k.replace('model.', '')
                # Only load keys that exist in the model (skip distill heads)
                if key in model.state_dict():
                    new_state_dict[key] = v
                # else:
                #     print(f"Skipping key not in model: {key}")
            
            # Load with strict=False to allow missing keys
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"\nMissing keys (will use pretrained weights): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys (skipped): {len(unexpected_keys)} keys")
            
            print(f"✓ Loaded checkpoint from step: {checkpoint.get('global_step', 'unknown')}")
        else:
            print("Checkpoint does not contain 'state_dict'. Loading directly.")
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
    else:
        # Load pretrained model
        print("Loading pretrained model from HuggingFace")
        model = AnySplat.from_pretrained("lhjiang/anysplat")
        model.to(device)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    os.makedirs(args.output_path, exist_ok=True)

    # load images
    image_folder = args.data_dir
    image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [process_image(img_path) for img_path in image_names]
    
    # Get indices from arguments or use llffhold
    if args.ctx_indices is not None and args.tgt_indices is not None:
        ctx_indices = [int(idx) for idx in args.ctx_indices.split(',')]
        tgt_indices = [int(idx) for idx in args.tgt_indices.split(',')]
        print(f"Using provided context indices: {ctx_indices}")
        print(f"Using provided target indices: {tgt_indices}")
    else:
        ctx_indices = [idx for idx, name in enumerate(image_names) if idx % args.llffhold != 0]
        tgt_indices = [idx for idx, name in enumerate(image_names) if idx % args.llffhold == 0]
        print(f"Using llffhold={args.llffhold} for indices")
        print(f"Context indices: {ctx_indices}")
        print(f"Target indices: {tgt_indices}")
    
    ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
    tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)

    ctx_images = (ctx_images+1)*0.5
    tgt_images = (tgt_images+1)*0.5
    b, v, _, h, w = tgt_images.shape

    # run inference
    # Pass pretrained_features=None explicitly to avoid the error
    encoder_output = model.encoder(
        ctx_images,
        global_step=0,
        visualization_dump={},
        pretrained_features=None,  # Add this to use encoder's own feature extraction
    )
    gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
    # save gaussians in .pt format
    # torch.save(gaussians, Path(f"{args.output_path}/gaussians.pt"))
    torch.save(pred_context_pose, Path(f"{args.output_path}/pred_context_pose.pt"))
    # export gaussians in .ply format for visualization
    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(f"{args.output_path}/gaussians.ply"))
    # exit(0)
    num_context_view = ctx_images.shape[1]
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(vggt_input_image, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])

    extrinsic_padding = torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    pred_all_context_extrinsic, pred_all_target_extrinsic = pred_all_extrinsic[:, :num_context_view], pred_all_extrinsic[:, num_context_view:]
    pred_all_context_intrinsic, pred_all_target_intrinsic = pred_all_intrinsic[:, :num_context_view], pred_all_intrinsic[:, num_context_view:]

    scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / pred_all_context_extrinsic[:, :, :3, 3].mean()
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
    print("scale_factor:", scale_factor)
    
    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v, device=device) * 0.01,
        torch.ones(1, v, device=device) * 100,
        (h, w)
        )

    save_interpolated_video(pred_all_context_extrinsic, pred_all_context_intrinsic, b, h, w, gaussians, args.output_path, model.decoder)
    
    # Save original images
    save_path = Path(args.output_path)
    # os.makedirs(save_path, exist_ok=True)
    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
        save_image(gt_image, save_path / "gt" / f"{idx:0>6}.jpg")
        save_image(pred_image, save_path / "pred" / f"{idx:0>6}.jpg")

    # save depth maps
    # for idx, depth_map in enumerate(output.depth[0]):
    #     save_image(depth_map, save_path / "depth" / f"{idx:0>6}.jpg")

    # save depth maps
    for idx, depth_map in enumerate(output.depth[0]):
        vis_depth = vis_depth_map(depth_map.unsqueeze(0))[0]  # vis_depth_map expects (v, h, w), returns (v, 3, h, w)
        save_image(vis_depth, save_path / "depth" / f"{idx:0>6}.jpg")
    # compute metrics
    psnr, ssim, lpips = compute_metrics(output.color[0], tgt_images[0])
    print(f"PSNR: {psnr.mean():.2f}, SSIM: {ssim.mean():.3f}, LPIPS: {lpips.mean():.3f}")
    
    #save metrics
    with open(save_path / "metrics.txt", "w") as f:
        f.write(f"PSNR: {psnr.mean():.2f}\n")
        f.write(f"SSIM: {ssim.mean():.3f}\n")
        f.write(f"LPIPS: {lpips.mean():.3f}\n")

if __name__ == "__main__":
    args = setup_args()
    evaluate(args)
