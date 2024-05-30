from .srn import SRNDataset
from .wrapper import WrapperDataset

import sys
sys.path.append('/home/melonie/gaussian-splatting')
from diffusion_ib import ColmapDataset

def get_dataset(cfg, name, overfit=None, val = False):
    if cfg.data.category == "cars" or cfg.data.category == "chairs":
        return SRNDataset(cfg, name)

    elif cfg.data.category == "objaverse" or cfg.data.category == "*" or cfg.data.category == "realestate":
        if cfg.data.category == "realestate"
            re10k = True
        else:
            re10k = False

        if not val:
            perm = "random"
            original_dataset = ColmapDataset(
                path=cfg.dataset_dir,
                classes=[cfg.data.category],
                suffix=cfg.dataset_suffix,
                split=name, 
                num_views=cfg.num_views,
                viewset_sampling='stratified' if cfg.randomise_viewsets else 'regular',
                viewset_multiplier=cfg.viewset_multiplier,
                target_view_distance= 0,
                resolution=cfg.data.training_resolution,
                crop_augmentation_min_scale=cfg.crop_augmentation_min_scale,
                crop_augmentation_max_shift=cfg.crop_augmentation_max_shift,
                overfit=overfit,
                re10k = re10k
            )
        else:
            original_dataset = ColmapDataset(
                path=cfg.dataset_dir,
                classes=[cfg.data.category],
                suffix=cfg.dataset_suffix,
                split=name, 
                num_views=cfg.num_views,
                viewset_sampling='single',
                viewset_multiplier=1,
                target_view_distance= 0,
                resolution=cfg.data.training_resolution,
                crop_augmentation_min_scale=cfg.crop_augmentation_min_scale,
                crop_augmentation_max_shift=cfg.crop_augmentation_max_shift,
                overfit=overfit,
                re10k= False
            )
        
        if val and cfg.data.category == "*":
            perm = "middle"
        elif val and cfg.data.category == "realestate":
            perm = "first"

        return WrapperDataset(
            cfg, original_dataset, perm
        )





