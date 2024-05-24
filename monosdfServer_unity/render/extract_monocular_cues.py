import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

from core.modules.unet import UNet
from core.modules.midas.dpt_depth import DPTDepthModel
from core.data.transforms import get_transform

class Extract_monocular_cues():
    
    def __init__(self, task, model_path, output_path = "logs"):

        self.task = task
        self.model_path = model_path
        self.output_path = output_path

        self.trans_topil = transforms.ToPILImage()

        # os.system(f"mkdir -p {self.args.output_path}")
        self.map_location = (lambda storage, loc: storage.cuda(1)) if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


        # get target task and model
        if self.task == 'normal':
            image_size = 384
            
            pretrained_weights_path = self.model_path
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
            print("loading omnidata normal model ...")
            checkpoint = torch.load(pretrained_weights_path, map_location=self.map_location)
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(image_size),
                                                get_transform('rgb', image_size=None)])

        elif self.task == 'depth':
            image_size = 384
            pretrained_weights_path =self.model_path  # 'omnidata_dpt_depth_v1.ckpt'
            # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
            self.model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
            print("loading omnidata depth model ...")
            checkpoint = torch.load(pretrained_weights_path, map_location=self.map_location)
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=0.5, std=0.5)])

        else:
            print("task should be one of the following: normal, depth")
            sys.exit()

        self.trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        ])


    def standardize_depth_map(self, img, mask_valid=None, trunc_value=0.1):
        if mask_valid is not None:
            img[~mask_valid] = torch.nan
        sorted_img = torch.sort(torch.flatten(img))[0]
        # Remove nan, nan at the end of sort
        num_nan = sorted_img.isnan().sum()
        if num_nan > 0:
            sorted_img = sorted_img[:-num_nan]
        # Remove outliers
        trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
        trunc_mean = trunc_img.mean()
        trunc_var = trunc_img.var()
        eps = 1e-6
        # Replace nan by mean
        img = torch.nan_to_num(img, nan=trunc_mean)
        # Standardize
        img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
        return img


    def save_outputs(self,img_path, output_file_name):
        with torch.no_grad():
            save_path = os.path.join(self.output_path, f'{output_file_name}_{self.task}.png')

            print(f'Reading input {img_path} ...')
            img = Image.open(img_path)

            img_tensor = self.trans_totensor(img)[:3].unsqueeze(0).to(self.device)

            rgb_path = os.path.join(self.output_path, f'{output_file_name}_rgb.png')
            self.trans_rgb(img).save(rgb_path)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3,1)

            output = self.model(img_tensor).clamp(min=0, max=1)

            if self.task == 'depth':
                output = output.clamp(0,1)
                np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
                output = 1 - output
                plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
                
            else:
                np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
                self.trans_topil(output[0]).save(save_path)
                
            print(f'Writing output {save_path} ...')


