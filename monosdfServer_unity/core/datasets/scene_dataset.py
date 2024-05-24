import os
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')

import core.utils.general as utils
from core.utils import rend_util
from glob import glob
import cv2
import random
import copy
from scipy.spatial.transform import Rotation as R




# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 K = None
                 ):

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views

        self.rgb_images = []
        self.depth_images = []
        self.normal_images = []
        self.intrinsics_all = []
        self.pose_all = []
        self.pose_all_gt = []
        self.pose_all_test = []
        self.mask_images = []
        self.camera_intrinsic = K
        self.K = np.eye(4)
        self.K[:3, :3] = self.camera_intrinsic
        print(self.K)

        self.n_images = 0

    

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_testdata(self, idx, data_type="train"):
        
        if data_type=="train":
            pose = self.pose_all[idx]
        else:
            pose = self.pose_all_test[idx]

        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": pose
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        batch_list = [(idx,sample,ground_truth)]


        return self.collate_fn(batch_list)


    def add_img(self, rgb, depth, normal, pose):
        
        rgb = rgb.reshape(3, -1).transpose(1, 0)
        self.rgb_images.append(torch.from_numpy(rgb).float())
        self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        normal = normal.reshape(3, -1).transpose(1, 0)
        # important as the output of omnidata is normalized
        normal = normal * 2. - 1.
        self.normal_images.append(torch.from_numpy(normal).float())

        self.pose_all.append(torch.from_numpy(pose).float())
        self.pose_all_gt.append(torch.from_numpy(pose).float())
        self.pose_all_test.append(torch.from_numpy(pose).float())

        self.intrinsics_all.append(torch.from_numpy(self.camera_intrinsic).float())

        mask = torch.ones_like(self.depth_images[0])
        self.mask_images.append(mask)

        self.n_images += 1



# Dataset with monocular depth and normal for Unity
class SceneDatasetDNIMUnity(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 K = None
                 ):

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views

        self.rgb_images = []
        self.depth_images = []
        self.normal_images = []
        self.intrinsics_all = []
        self.pose_all = []
        self.pose_all_gt = []
        self.pose_all_test = []
        self.mask_images = []
        self.camera_intrinsic = K
        self.K = np.eye(4)
        self.K[:3, :3] = self.camera_intrinsic
        print(self.K)

        self.n_images = 0

        self.scale_mat = np.loadtxt("../data/scale_mat/hm3d_00804.txt") 
        self.intrinsic = torch.from_numpy(self.camera_intrinsic).float()

    

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]


    def get_testdata(self, idx, data_type="train"):
        
        if data_type=="train":
            pose = self.pose_all[idx]
        else:
            pose = self.pose_all_test[idx]

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": pose
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        batch_list = [(idx,sample,ground_truth)]


        return self.collate_fn(batch_list)
    
    def covert_pose(self,K, scale_mat, pose):
        # print(i)
        # print("before :", pose)
        pose = np.array(pose)
        R = copy.deepcopy(pose[:3,:3])
        pose = K @ np.linalg.inv(pose)
        P = pose @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        # print("after :", pose)
        pose[:3,:3] = R
        # print("after1 :", pose)
        return pose

    def add_img(self, rgb, depth, normal, pose):
        
        # ## room
        # depth = depth[0,:,:]
        # mask = np.ones_like(depth)
        # mask[depth<0.5]=0
        # mask[depth>=6]=0
        # depth = depth/6.0/50
        # # print("shape = ",depth.shape,mask.shape)
        
        
        ## hm3d
        depth = depth[:,:]
        mask = np.ones_like(depth)
        # mask[depth<0.2]=0
        # mask[depth>=10]=0
        # depth = depth/10.0/50
    

        rgb = rgb.reshape(3, -1).transpose(1, 0)
        self.rgb_images.append(torch.from_numpy(rgb).float())
        self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        normal = normal.reshape(3, -1).transpose(1, 0)
        # important as the output of omnidata is normalized
        normal = normal * 2. - 1.
        self.normal_images.append(torch.from_numpy(normal).float())

        ## Blender
        T = np.array([[-1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
        pose = T @ pose
        pose = self.covert_pose(self.K, self.scale_mat, pose)
        # print(pose)

        self.pose_all.append(torch.from_numpy(pose).float())
        self.pose_all_gt.append(torch.from_numpy(pose).float())
        self.pose_all_test.append(torch.from_numpy(pose).float())

        self.intrinsics_all.append(torch.from_numpy(self.camera_intrinsic).float())

        self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

        self.n_images += 1

