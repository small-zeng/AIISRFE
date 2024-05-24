from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.http import request
import os
from PIL import Image
import numpy as np
import json
import threading
from skimage import io
from scipy.spatial.transform import Rotation as R
import cv2
import time

from core.monosdf import *
from core.extract_monocular_cues import Extract_monocular_cues
from core.utils import rend_util
import threading
import core.utils.plots as plt

# Create your views here.

# 接受图片并根据训练策略加入训练
# 向planner发送收敛完成信号
# 根据坐标方向返回到表面的距离
# 根据起始终点坐标返回是否直线通路
# 根据观测点及观测方向返回信息增益或不确定性

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
############################################################################  delete
global count
count = [0]
img_index = [0]
global controller
controller = [None]

# torch.autograd.set_detect_anomaly(True)
base_dir = "../exps/red_house_0/" + scene_name + version
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

base_dir = base_dir + "/trainset"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    


def start_mission(img,depth,normal,pose):
    
    if len(imgs) < 1:
        imgs.append(img)
        depth_imgs.append(depth)
        normal_imgs.append(normal)
        poses.append(pose)  
    else:
        if running_flag[0]:
            controller[0].add_img(img,depth,normal,pose,False)
        else:
            controller[0] = Controller(H,W,focal,imgs,depth_imgs,normal_imgs,poses)
            controller[0].add_img(img,depth,normal,pose,False)
            # time.sleep(1)
            threading.Thread(target=controller[0].train,args=()).start()
            running_flag[0] = True
            
    
##################  test online train ###################
# for i in range(247):
#     rgb = rend_util.load_rgb(os.path.join(base_dir,f"{i}_main.png"),[384,384])
#     depth = np.load(os.path.join(base_dir,f"{i}_om_depth.npy"))
#     # depth = cv2.imread(os.path.join(base_dir,f"{i}_depth.png"),cv2.IMREAD_ANYDEPTH)/65535.0*10.0
#     # depth = cv2.resize(depth, (384,384), interpolation=cv2.INTER_LINEAR)
#     depth = depth[:,:]
#     normal = np.load(os.path.join(base_dir,f"{i}_om_normal.npy"))
#     pose = np.loadtxt(os.path.join(base_dir,f"{i}_pose.txt"))
#     # imgs.append(rgb)
#     # depth_imgs.append(depth)
#     # normal_imgs.append(normal)
#     # poses.append(pose)
#     start_mission(rgb,depth,normal,pose)
#     time.sleep(1)

###########################################################
 
def load_data():
    # print("index = ", img_index[0],", ", count[0])
    # try:
    if img_index[0] < count[0]:
        print("img_index = ", img_index[0])
        ts = time.time()
        ##  ominidata output 
        depth_extractor.save_outputs(base_dir+f"/{img_index[0]}_main.png",f"{img_index[0]}_om")
        normal_extractor.save_outputs(base_dir+f"/{img_index[0]}_main.png",f"{img_index[0]}_om")
        print("t2 = ", time.time()-ts)

        pose = np.loadtxt(os.path.join(base_dir,f"{img_index[0]}_pose.txt"),delimiter=" ") 
        rgb = rend_util.load_rgb(os.path.join(base_dir,f"{img_index[0]}_main.png"),[384,384])
        depth = cv2.imread(os.path.join(base_dir,f"{img_index[0]}_depth.png"),cv2.IMREAD_ANYDEPTH)/65535.0*10.0
        depth = cv2.resize(depth, (384,384), interpolation=cv2.INTER_LINEAR)
        depth = depth[:,:] / 100.0
        # depth = np.load(os.path.join(base_dir,f"{img_index[0]}_om_depth.npy"))
        normal = np.load(os.path.join(base_dir,f"{img_index[0]}_om_normal.npy"))
        ## strat mapping
        start_mission(rgb,depth,normal,pose)
        img_index[0] = img_index[0] + 1
    # except:
    #     print("load data error")
    process_data_timer()

def process_data_timer():
    # print("process_data_timer")
    t1 = threading.Timer(0.2, load_data)
    t1.start()


## load model and start mapping
depth_model_path = "core/pretrained_models/omnidata_dpt_depth_v2.ckpt"
normal_model_path = "core/pretrained_models/omnidata_dpt_normal_v2.ckpt"
depth_extractor = Extract_monocular_cues(task="depth",model_path=depth_model_path,output_path = base_dir)
normal_extractor = Extract_monocular_cues(task="normal",model_path=normal_model_path,output_path = base_dir)
process_data_timer()



def get_picture(request):
    if request.method == 'POST':
        t0 = time.time()
        img = request.FILES.get("rgb", None)
        depth_img = request.FILES.get("depth",None)    
        
        pose = request.FILES.get("pose",None)
        pose = np.loadtxt(pose,delimiter=" ") 
        # print(pose.shape)
        img = np.array(Image.open(img))
        # print(img.shape)
        depth_img = np.array(io.imread(depth_img))
        # print(depth_img.shape) 

        cv2.imwrite(base_dir+f"/{count[0]}_main.png",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(base_dir+f"/{count[0]}_depth.png",depth_img)
        np.savetxt(base_dir+f"/{count[0]}_pose.txt",pose)
            
        print("receive image:  count[0] = ", count[0])
        count[0] = count[0] + 1
        
        if count[0] == 2:
            info_planner_timer()
            
    
    return HttpResponse("get image")


        

def terminate_mission(request):
    controller[0].terminate_work()
    running_flag[0] = False
    return HttpResponse("terminate successfully")

def save_model(request):
    controller[0].save_model()
    return HttpResponse("save successfully")

def get_uncertainty(request):
    # print("come in")
    t0 = time.time()
    if request.method == 'POST':
        # print("come into get_uncertainty")
        json_data = json.loads(request.body.decode('utf-8'))
        # 解析JSON数据
        locations = []
        us = []
        vs = []
        for view in json_data:
            x = view['x']
            y = view['y']
            z = view['z']
            u = view['u']
            v = view['v']
            locations.append([x,y,z])
            us.append(u)
            vs.append(v)
            
        poses = []
        K = controller[0].train_dataset.K
        scale_mat = controller[0].train_dataset.scale_mat
        for i in range(len(locations)):
            pose = np.eye(4)
            pose[:3,3] =np.array(locations[i])
            pose = controller[0].train_dataset.covert_pose(K,scale_mat,pose)
            locations[i] = pose[:3,3].tolist()
            pose = get_pose(locations[i],us[i],vs[i])
            poses.append(pose)
            # print(pose)
        
        uncers = controller[0].get_uncertainty(poses,device2,NUM=500)
        # print("uncers = ")
        # print(uncers.cpu().numpy().tolist())
        re = {"uncers":uncers.cpu().numpy().tolist()}
        # re = {"uncers":[1,2]}
        print("get_uncertainty time = ", time.time()-t0)
        return JsonResponse(re)
    

def get_surface_uncertainty(request):
    
    t0 = time.time()
    
    epoch, path = controller[0].get_surface_uncertainty()
    
    
    re = {"model_index":epoch,"model_dir":path}
    print("get_surface_uncertainty time = ", time.time()-t0)
    return JsonResponse(re)
    




def info_planner_timer():
    time.sleep(2)
    t1 = threading.Timer(3, info_planner)
    t1.start()

    
def info_planner():
    # response = requests.get("http://192.168.31.25:7300/isfinish") 
    response = requests.get("http://127.0.0.1:7300/isfinish") 
    print(response)
    print("send info to planner")
    # start_timer()
    

