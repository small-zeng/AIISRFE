import os
from PIL import Image
import numpy as np
import json
import threading
from skimage import io
from scipy.spatial.transform import Rotation as R
import cv2
import time

from render.monosdf_forrender import *
from render.extract_monocular_cues import Extract_monocular_cues
from render.utils import rend_util
import threading



H = 384
W = 384
focal = 384/800*600
K = np.array([[focal, 0, 0.5*W],[0, focal, 0.5*H],[0, 0, 1]])
print("K = ",K)

running_flag = [False]
imgs = []
depth_imgs = []
normal_imgs = []
poses = []
global count
count = [0]
global controller
controller = [None]

# data_dir = "../data/HM3D/traj"
# iteration = 95000
# N = 300

iters_list = [229,415,775,981,1000,1326,1624,1807,2000,2050,2239,2488,3000,4000,5000,6000,
              7000,8000,9000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,
              60000,65000,70000,75000,80000,85000,90000,95000,100000]
data_dir ="../data/HM3D/hm3d_00804_240"
# data_dir = "../data/Gibson/gibson_Convoy_240"
# data_dir = "../data/Room/childroom_random_240"
# iteration = 100000
N = 1

## 加载测试集
for i in range(0,N,1):
    print(i)
    rgb = rend_util.load_rgb(os.path.join(data_dir,f"{i}main.png"),[384,384])[:3,:,:]
    # print(rgb.shape)
    # depth = rend_util.load_depth(os.path.join(data_dir,f"{i}depth.png"),[384,384])/255.0*6.0
    depth = cv2.imread(os.path.join(data_dir,f"{i}depth.png"),cv2.IMREAD_ANYDEPTH)/65535.0*10.0
    depth = cv2.resize(depth, (384,384), interpolation=cv2.INTER_LINEAR)
            
    # normal = np.load(os.path.join(data_dir,f"{i}_om_normal.npy"))
    normal = np.ones((3,384,384))
    pose = np.loadtxt(os.path.join(data_dir,f"{i}.txt"),delimiter=" ")
    imgs.append(rgb)
    depth_imgs.append(depth)
    normal_imgs.append(normal)
    poses.append(pose)

# controller[0] = Controller(H,W,focal,imgs,depth_imgs,normal_imgs,poses)
# controller[0].load_model(iteration)


# #### scale pose
# ## Blender
# poses_scale = []
# K = controller[0].train_dataset.K
# scale_mat = controller[0].train_dataset.scale_mat
# T = np.array([[-1,0,0,0],
#               [0,1,0,0],
#               [0,0,1,0],
#               [0,0,0,1]])
# for pose in poses:
#   pose_scale = T @ pose
#   pose_scale = controller[0].train_dataset.covert_pose(K, scale_mat, pose_scale)
#   poses_scale.append(pose_scale)


# for index in range(0,N,1):
#   t0 = time.time()
#   uncer_all=controller[0].render_test(iteration,"test_all", index,device2)


## x渲染多个mesh
for iteration in iters_list:
    controller[0] = Controller(H,W,focal,imgs,depth_imgs,normal_imgs,poses)
    controller[0].load_model(iteration)

    #### scale pose
    ## Blender
    poses_scale = []
    K = controller[0].train_dataset.K
    scale_mat = controller[0].train_dataset.scale_mat
    T = np.array([[-1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    for pose in poses:
      pose_scale = T @ pose
      pose_scale = controller[0].train_dataset.covert_pose(K, scale_mat, pose_scale)
      poses_scale.append(pose_scale)


    for index in range(0,N,1):
      t0 = time.time()
      uncer_all=controller[0].render_test(iteration,"test_all", index,device2)