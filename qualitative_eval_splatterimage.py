import os
import sys
import tqdm
import torch
import torchvision
import argparse
import diffusers
import accelerate
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from scene.wrapper import WrapperDataset
from scene.wrapper_all import WrapperDatasetAll

import itertools
import torchmetrics
import pandas as pd
import torch_fidelity
import matplotlib

sys.path.append('/home/melonie/gaussian-splatting')
from diffusion_ib import ColmapDataset

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

def depth_to_rgb(depth):
    # Ensure depth is a numpy array
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    # Define the colormap
    colourmap = np.asarray(matplotlib.colormaps['magma'].colors)

    # Calculate disparity
    disparity = 1.0 / depth
    is_inf = np.isinf(disparity)
    if not np.all(is_inf):
        disparity[is_inf] = disparity[~is_inf].max()
    else:
        disparity[is_inf] = 1.0

    # Normalize disparity values
    disparity_lower, disparity_upper = np.quantile(disparity, q=[0.02, 0.98])
    disparity_normalised = np.clip((disparity - disparity_lower) / (disparity_upper - disparity_lower + 1.e-4), 0.0, 1.0)

    # Quantize disparity to 255 levels
    disparity_quantised = np.round(disparity_normalised * 255.0).astype(np.int32)

    # Map quantized disparity values to RGB using the colormap
    rgb_image = colourmap[disparity_quantised]

    rgb_image = torch.from_numpy(rgb_image)

    return rgb_image

@torch.inference_mode
def main(args):

    device = torch.device("cuda:{}".format(args.device_idx))
    torch.cuda.set_device(device)

    # load cfg
    cfg = OmegaConf.load(os.path.join(args.ckpt_dir, ".hydra", "config.yaml"))

    # load model
    model = GaussianSplatPredictor(cfg)
    ckpt_loaded = torch.load(os.path.join(args.ckpt_dir, "model_latest.pth"), map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    original_dataset = ColmapDataset(
        path=args.dataset_dir,
        classes=args.dataset_classes,
        suffix=args.dataset_suffix,
        split=args.split,
        num_views=6,
        viewset_sampling='single',  # one canonical viewset per scene
        viewset_multiplier=1,
        target_view_distance='halfway',
        resolution=cfg.data.training_resolution,
        crop_augmentation_min_scale=1.,
        crop_augmentation_max_shift=0,
        overfit=None if args.num_scenes == -1 else args.num_scenes,
        re10k=args.re10k,
    )

    classes = original_dataset.classes

    dataset_all = WrapperDatasetAll(cfg, original_dataset, perm=args.perm)

    dataloader_all = torch.utils.data.DataLoader(dataset_all, batch_size=args.batch_size, shuffle=False)


    if args.out_subdir is None:
        args.out_subdir = f'vis_{args.split}_seed-{args.seed}'
    out_dir = f'{args.ckpt_dir}/{args.out_subdir}'



    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for d_idx, data in enumerate(tqdm.tqdm(dataloader_all)):  
        data = {k: v.to(device) for k, v in data.items()}
        prefixes = [bytes(scene_id_padded).decode('ascii') for scene_id_padded in data['scene_id_padded']]
        
        
        rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]

        if cfg.data.category == "hydrants" or cfg.data.category == "teddybears" or cfg.data.category =="*" or cfg.data.category =="realestate":
            focals_pixels_pred = data["focals_pixels"][:, :cfg.data.input_images, ...]
            input_images = torch.cat([data["gt_images"][:, :cfg.data.input_images, ...],
                                      data["origin_distances"][:, :cfg.data.input_images, ...]],
                                      dim=2)
                   
        else:
            focals_pixels_pred = None
            input_images = data["gt_images"][:, :cfg.data.input_images, ...]


        example_id = dataloader_all.dataset.get_example_id(d_idx)
        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        for n in range(len(prefixes)):
            images_tiled = []
            images_video = []
            depth_tiled = []
            depth_video = []
            gt_images = []
            class_name = classes[data["class_idx"][n]]
            
            for r_idx in range(data["view_to_world_transforms"].shape[1]):
                
                if cfg.data.category == "hydrants" or cfg.data.category == "teddybears" or cfg.data.category =="*" or cfg.data.category =="realestate":
                    focals_pixels_render = data["focals_pixels"][n, r_idx]
                else:
                    focals_pixels_render = None
            
                output = render_predicted({k: v[n].contiguous() for k, v in reconstruction.items()},
                                        data["world_view_transforms"][n, r_idx],
                                        data["full_proj_transforms"][n, r_idx], 
                                        data["camera_centers"][n, r_idx],
                                        background,
                                        cfg,
                                        focals_pixels=focals_pixels_render)
                
                image = output["render"]
                depth = output["depth_map"].reshape((cfg.data.training_resolution,cfg.data.training_resolution))

                if r_idx < data["gt_images"].shape[1]:    
                    images_tiled.append(image)
                    gt_image = data["gt_images"][n, r_idx]
                    gt_images.append(gt_image)
                    depth_tiled.append(depth)
                
                else:                   
                    images_video.append(image)
                    depth_video.append(depth)                   

            os.makedirs(f'{out_dir}/{args.num_cond_views}-im-recon/{class_name}', exist_ok=True)

            rgb_v = torch.stack(images_video, dim=0)
            depth_v = torch.stack([depth_to_rgb(depth.cpu().numpy()) for depth in depth_video], dim=0)
            
            rgb_vis = (rgb_v.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
            depth_vis = (depth_v * 255).cpu().numpy().astype(np.uint8)
            rgb_and_depth_video = np.concatenate([rgb_vis, depth_vis], axis=2)
            rgb_and_depth_video = np.concatenate([rgb_and_depth_video, rgb_and_depth_video[::-1]], axis=0)
            torchvision.io.write_video(
                    f'{out_dir}/{args.num_cond_views}-im-recon/{class_name}/{prefixes[n]}_00.mp4', 
                    rgb_and_depth_video,
                    args.fps,
            )

            rgb_t = torch.stack(images_tiled, dim=0)
            depth_t = torch.stack([depth_to_rgb(depth.cpu().numpy()) for depth in depth_tiled], dim=0)

            gt = torch.stack(gt_images, dim=0)
            if args.perm == "middle":
                permuted_indices = torch.tensor([3, 4, 5, 0, 1, 2])
                gt = gt[permuted_indices]
                rgb_t = rgb_t[permuted_indices]
                depth_t = depth_t[permuted_indices]

            def convert_rgb(x):
                x = x.permute(2, 0, 3, 1).reshape(x.shape[2], -1, 3)
                return (x * 255).cpu().numpy().astype(np.uint8)

            input_image = input_images[n, :, :3, :, :]
            input_rgb_vis = convert_rgb(input_image)
            gt_rgb_vis = convert_rgb(gt)
            pred_rgb_vis = convert_rgb(rgb_t)

            pred_depth_vis = (depth_t.cpu().numpy().transpose(1, 0, 2, 3).reshape(depth_t.shape[1], -1, 3) * 255).astype(np.uint8)

            Image.fromarray(input_rgb_vis).save(f'{out_dir}/{args.num_cond_views}-im-recon/{class_name}/{prefixes[n]}_input.png')

            divider = np.full([input_images.shape[-1], 4, 3], 255, dtype=np.uint8)
            tiled = np.concatenate([input_rgb_vis, divider, gt_rgb_vis, divider, pred_rgb_vis, divider, pred_depth_vis], axis=1)
            Image.fromarray(tiled).save(f'{out_dir}/{args.num_cond_views}-im-recon/{class_name}/{prefixes[n]}_00.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=str, default='1')
    parser.add_argument('--dataset_dir', type=str, default='/scratch/melonie/re10k')
    parser.add_argument('--dataset_classes', type=str, nargs='+', default=["realestate"])  
    parser.add_argument('--selected_classes', type=str, nargs='+')  
    parser.add_argument('--dataset_suffix', type=str, default='256/converted')
    parser.add_argument('--ckpt_dir', type=str, default='/home/melonie/splatter-image/experiments_out/2024-05-18/19-37-52')
    parser.add_argument('--ema_model', action='store_true')
    parser.add_argument('--out_subdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_scenes', type=int, default=5000)
    parser.add_argument('--num_cond_views', type=int, default=1)
    parser.add_argument('--perm', type=str, default="first")
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--batch_size', type=float, default=8)
    parser.add_argument('--re10k', type=bool, default=True)


    main(parser.parse_args())