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

data_dir = "../data/HM3D/traj"
data_dir = "../data/HM3D/random_480"
iteration = 1000
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
    pose = np.loadtxt(os.path.join(data_dir,f"{i}.txt"),delimiter="\t")
    imgs.append(rgb)
    depth_imgs.append(depth)
    normal_imgs.append(normal)
    poses.append(pose)

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



for index in range(0,1,1):
  t0 = time.time()
  # uncer_all=controller[0].render_test(iteration,"test_all", index,device2)
  
  location = [0,1.7,-2]
  u = 0
  v = np.pi/3.0
  pose = np.eye(4)
  pose[:3,3] =np.array(location)
  pose = controller[0].train_dataset.covert_pose(K,scale_mat,pose)
  location = pose[:3,3].tolist()
  pose = get_pose(location,u,v)
  uncer_all= controller[0].get_all_uncertainty(pose,iteration,"test_all", index,device2)
  
  
  
  t1 = time.time()
  uncers = controller[0].get_uncertainty([pose],device2,NUM=500)
  print("t_sample",time.time()-t1)
  for i in range(1):
    print(i,uncers[i])







## uncer test
# t0 = time.time()
# for index in range(0,N,1):
#   data = []
#   uncer_all=controller[0].render_test(iteration,"test_all", index,device2)
#   uncer_all = uncer_all.cpu().numpy()
#   # print(type(poses[0]))
#   for n in range(1,10000,10):
#     uncers = controller[0].get_uncertainty([poses_scale[index]],device2,NUM=n)
#     uncers = uncers.cpu().numpy()[0]
#     # print("uncers = ", uncers)
#     err_abs = abs(uncers-uncer_all)
#     err_ratio = err_abs/uncer_all
#     print("t_sample",time.time()-t0,n,uncer_all,uncers,err_abs,err_ratio)
#     data.append([n,uncer_all,uncers,err_abs,err_ratio])
#   data = np.array(data)
#   np.savetxt(f"logs/uncer_err/uncer{index}.txt",data)






