import sys
sys.path.append('/home/melonie/gaussian-splatting')

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World
from einops import repeat

class WrapperDataset(Dataset):
    def __init__(self, cfg, original_dataset, perm):
        self.original_dataset = original_dataset
        self.cfg = cfg
        self.perm = perm

    def __getitem__(self, index):
        # Get item from the original dataset
        original_item = self.original_dataset[index]

        # Modify the output as needed
        modified_output = self.modify_output(original_item)

        return modified_output

    def __len__(self):
        # Return the length of the original dataset
        return len(self.original_dataset)


    def get_example_id(self, index):
        # Get item from the original dataset
        original_item = self.original_dataset[index]

        #scene id
        example_id = original_item["scene_name"]
        return example_id

    def make_poses_relative_to_first(self, images_and_camera_poses):
        inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
        for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
            images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                                images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                                inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
            images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
        return images_and_camera_poses

    def get_source_cw2wT(self, source_cameras_view_to_world):
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)

    def get_origin_distances(self, view_to_world_transform):
        # Outputs the origin_distances for each view.
        # view_to_world_transform: Tensor of shape (num_views, 4, 4) containing camera-to-world transformations for each view.
        
        num_views = view_to_world_transform.shape[0]
        origin_distances = torch.zeros((num_views, 1, self.cfg.data.training_resolution, self.cfg.data.training_resolution))
        
        for i in range(self.cfg.opt.imgs_per_obj):
            camera_to_world = view_to_world_transform[i]
            camera_center_to_origin = -camera_to_world[3, :3]
            camera_z_vector = camera_to_world[2, :3]
            origin_distance = torch.dot(camera_center_to_origin, camera_z_vector).unsqueeze(0)
            origin_distances[i] = origin_distance.repeat(1, 1, self.cfg.data.training_resolution, self.cfg.data.training_resolution)

        return origin_distances

    def fov2focal(self, fov, pixels):
        return pixels / (2 * torch.tan(fov / 2))

    def getWorld2View(self, Rt_in):
        R = Rt_in[:3, :3]
        t = Rt_in[:3, 3]
        Rt = torch.zeros((4, 4), dtype=torch.float32)
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return Rt

    # Wrapper to modify original item
    def modify_output(self, original_item):
        noised_images =  torch.cat((original_item["noised_images"],original_item["target_images"]), dim=0)
        noised_cameras = original_item["noised_cameras"]
        target_cameras = original_item["target_cameras"]
        fovy = torch.cat((noised_cameras["FoVy"],target_cameras["FoVy"] ),dim=0)
        fovx = torch.cat((noised_cameras["FoVx"],target_cameras["FoVx"] ),dim=0)
        full_proj_transform = torch.cat((noised_cameras["full_proj_transform"],target_cameras["full_proj_transform"] ),dim=0)
        world_view_transform = torch.cat((noised_cameras["world_view_transform"],target_cameras["world_view_transform"] ),dim=0)
        camera_center = torch.cat((noised_cameras["camera_center"],target_cameras["camera_center"] ),dim=0)

        num_views = world_view_transform.shape[0]

        view_to_world_transform = torch.zeros((num_views,4,4))
        for n in range(num_views):
            view_to_world_transform[n,:,:] = torch.linalg.inv(world_view_transform[n,:,:])
        
        fovx_pixels = self.fov2focal(fovx, self.cfg.data.training_resolution)
        fovy_pixels = self.fov2focal(fovy, self.cfg.data.training_resolution)
        focals_pixels = torch.stack([fovx_pixels, fovy_pixels], dim=1) 

        origin_distances = self.get_origin_distances(view_to_world_transform)

        if self.perm == "first":
            permuted_indices = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11])

        elif self.perm == "random":
            permuted_indices = torch.randperm(world_view_transform.size(0))
        
        else:
            middle_index = 3
            permuted_indices = torch.cat((torch.arange(middle_index, noised_images.size(0)), torch.arange(middle_index)), dim=0)

        images_and_camera_poses = {
            "gt_images": noised_images.float()[permuted_indices] / 255.0,
            "world_view_transforms": world_view_transform[permuted_indices],
            "view_to_world_transforms": view_to_world_transform[permuted_indices],
            "full_proj_transforms": full_proj_transform[permuted_indices],
            "camera_centers": camera_center[permuted_indices],
            "focals_pixels": focals_pixels[permuted_indices],
            "origin_distances": origin_distances[permuted_indices],
            "scene_id_padded" : original_item["scene_id_padded"],
            "class_idx" : original_item["class_idx"]
        }

        #From their code
        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
        
        return images_and_camera_poses