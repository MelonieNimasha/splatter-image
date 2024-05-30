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
from scene.wrapper_quantitative import WrapperDataset

import itertools
import torchmetrics
import pandas as pd
import torch_fidelity

sys.path.append('/home/melonie/gaussian-splatting')
from diffusion_ib import ColmapDataset

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

def save_img_cond(args, cfg, model, dataloader, out_dir, device):

    os.makedirs(f'{out_dir}/{args.num_cond_views}-im-recon/rgb', exist_ok=True)
    os.makedirs(f'{out_dir}/gt/rgb', exist_ok=True)

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        

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

        example_id = dataloader.dataset.get_example_id(d_idx)
        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        for n in range(len(prefixes)):
            for r_idx in range(data["gt_images"].shape[1]):
                if cfg.data.category == "hydrants" or cfg.data.category == "teddybears" or cfg.data.category =="*" or cfg.data.category =="realestate":
                    focals_pixels_render = data["focals_pixels"][n, r_idx]
                else:
                    focals_pixels_render = None
            
                image = render_predicted({k: v[n].contiguous() for k, v in reconstruction.items()},
                                        data["world_view_transforms"][n, r_idx],
                                        data["full_proj_transforms"][n, r_idx], 
                                        data["camera_centers"][n, r_idx],
                                        background,
                                        cfg,
                                        focals_pixels=focals_pixels_render)["render"]

                if not r_idx == 0:
                    torchvision.utils.save_image(image, f'{out_dir}/{args.num_cond_views}-im-recon/rgb/{prefixes[n]}_00_{r_idx:03}.png')
                torchvision.utils.save_image(data["gt_images"][n, r_idx], f'{out_dir}/gt/rgb/{prefixes[n]}_{r_idx:03}.png')
    
def evaluate_recon(eval_dir):

    # maybe_create_rgb_both_subdir(f'{eval_dir}/1-im-recon')

    split_recon_filenames = list(map(
        lambda filename: filename.rsplit('_', maxsplit=2),
        sorted(os.listdir(f'{eval_dir}/1-im-recon/rgb'))
    ))
    scene_to_sample_to_suffixes = {
        scene_id: {
            sample_id: [bits[-1] for bits in filenames]
            for sample_id, filenames in itertools.groupby(filenames, lambda bits: bits[1])
        }
        for scene_id, filenames in itertools.groupby(split_recon_filenames, lambda bits: bits[0])
    }

    lpips = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').cuda()
    name_to_metric = {
        'psnr': lambda source_views, target_views:
            torchmetrics.functional.image.peak_signal_noise_ratio(
                source_views,
                target_views,
                data_range=(-1., 1.),
                dim=(1, 2, 3),  # take mean over CHW *before* taking log
                reduction='none',  # preserve view dimension after log
            ).mean(dim=0),  # average over views
        'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=(-1., 1.)).cuda(),
        'neg-lpips': lambda source_views, target_views:
            -lpips(source_views, target_views),
    }

    scene_best_metrics = []
    for scene_id, sample_to_suffix in tqdm.tqdm(scene_to_sample_to_suffixes.items()):
        target_views = torch.stack([
            load_image(f'{eval_dir}/gt/rgb/{scene_id}_{suffix}')
            for suffix in next(iter(sample_to_suffix.values()))
        ], dim=0).cuda()
        best_metrics = {metric_name: -torch.inf for metric_name in name_to_metric}
        for sample_id, suffixes in sample_to_suffix.items():
            pred_views = torch.stack([
                load_image(f'{eval_dir}/1-im-recon/rgb/{scene_id}_{sample_id}_{suffix}')
                for suffix in suffixes
            ], dim=0).cuda()
            metrics = {
                name: metric(pred_views, target_views).item()
                for name, metric in name_to_metric.items()
            }
            best_metrics = {
                metric_name: max(best_metrics[metric_name], metrics[metric_name])
                for metric_name in metrics
            }
        scene_best_metrics.append({'scene_id': scene_id, **best_metrics})

    scene_best_metrics = pd.DataFrame(scene_best_metrics)
    scene_best_metrics.sort_values(by='neg-lpips', ascending=False, inplace=True)
    scene_best_metrics.to_csv(f'{eval_dir}/recon-results_best-to-worst-lpips.csv')
    print(scene_best_metrics.describe())
    return scene_best_metrics.mean(numeric_only=True), scene_best_metrics.std(numeric_only=True)

def load_image(filename):
    return torch.from_numpy(np.asarray(Image.open(filename))).to(torch.float32).permute(2, 0, 1) / 255. * 2. - 1.

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
    )

    dataset = WrapperDataset(cfg, original_dataset, perm=args.perm)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.out_subdir is None:
        args.out_subdir = f'eval_{args.split}_seed-{args.seed}'
    out_dir = f'{args.ckpt_dir}/{args.out_subdir}'

    save_img_cond(args, cfg, model, dataloader, out_dir, device)
    evaluate_recon(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=str, default='0')
    parser.add_argument('--dataset_dir', type=str, default='/scratch/melonie/re10k')
    parser.add_argument('--dataset_classes', type=str, nargs='+', default=['realestate'])
    parser.add_argument('--dataset_suffix', type=str, default='256/converted')
    parser.add_argument('--ckpt_dir', type=str, default='/home/melonie/splatter-image/experiments_out/2024-05-18/19-37-52')
    parser.add_argument('--out_subdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_scenes', type=int, default=5000)
    parser.add_argument('--num_cond_views', type=int, default=1)
    parser.add_argument('--perm', type=str, default="first")
    parser.add_argument('--batch_size', type=float, default=8)
    parser.add_argument('--re10k', type=bool, default=True)

    main(parser.parse_args())