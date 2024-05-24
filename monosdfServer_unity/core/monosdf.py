import os
import time
from tqdm.auto import tqdm

import json, random
from core.utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import requests
import copy
import threading

from pyhocon import ConfigFactory


from core.utils import rend_util
import core.utils.plots as plt
import core.utils.general as utils
from core.common import *
from core.model.loss import compute_scale_and_shift
from core.utils.general import BackprojectDepth

from torch.autograd import Variable
import imageio
from datetime import datetime
import cv2
import configargparse

import argparse


scene_name = 'test_'
version = '0'

class Controller():

    # prepare condition HWF and at least 4 imgs
    def __init__(self, H, W, focal, imgs, depth_imgs,normal_imgs, poses):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.H = H
        self.W = W
        self.focal = focal
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.newimg = False
        self.GPU_INDEX = 0
        
        parser = config_parser()
        self.args, self.argv = parser.parse_known_args()
        self.conf = ConfigFactory.parse_file(self.args.conf)
        self.batch_size = self.args.batch_size
        self.exps_folder_name = self.args.exps_folder
        print("render_only = ", self.args.render_only)


        self.expname = self.conf.get_string('train.expname') 
        scan_id = self.args.scan_id if self.args.scan_id != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        is_continue =self.args.is_continue

        if self.GPU_INDEX == 0 and not is_continue:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            self.timestamp = scene_name + version
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
            
            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(self.args.conf, os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
           
            
        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')
        dataset_conf = self.conf.get_config('dataset')
        if self.args.scan_id != -1:
            dataset_conf['scan_id'] = self.args.scan_id
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf,K = self.K)
        
        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        self.nepochs = 1000
        
        self.add_img_index = 0
        for i in np.arange(0, len(imgs), 1):
            print(i)
            self.add_img(imgs[i], depth_imgs[i], normal_imgs[i],poses[i],False)


        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        self.Grid_MLP = self.model.Grid_MLP
        self.model = self.model.to(device)

        self.model2 = self.conf.get_config('model')
        self.model2 = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        self.model2 = self.model2.to(device2)

        ## 加载位姿
        self.is_gt_pose = self.conf.get_float('train.is_gt_pose')
        if self.is_gt_pose:
            self.pose_all_est = self.train_dataset.pose_all_gt
        else:
            self.pose_all_est = self.train_dataset.pose_all
        self.pose_all_gt = self.train_dataset.pose_all_gt
        self.pose_all_init = copy.deepcopy(self.pose_all_est)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))


        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        self.BA_cam_lr = self.conf.get_float('train.BA_cam_lr')
        self.BA = self.conf.get_float('train.BA')
        self.BA_cam_size = int(self.conf.get_float('train.BA_cam_size'))
       
        self.grid_para_list = list(self.model.implicit_network.grid_parameters())
        self.net_para_list = list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters())
        self.density_para_list = list(self.model.density.parameters())
        camera_tensor = torch.tensor([1.0,0.,0.,0.,0.,0.,0.])
        self.camera_tensor_list = [Variable(camera_tensor.to(device), requires_grad=True)]

        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': self.grid_para_list, 'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': self.net_para_list,'lr': self.lr},
                {'name': 'density', 'params': self.density_para_list, 'lr': self.lr},
                {'name': 'camera', 'params': self.camera_tensor_list, 'lr': 0},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * 100
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))


        if is_continue:
            self.scheduler_params_subdir = "SchedulerParameters"
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            timestamp =self.args.timestamp
            self.plots_dir = os.path.join(self.expdir, timestamp, 'plots')
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(self.args.checkpoint) + ".pth"))
            self.model2.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(self.args.checkpoint) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(self.args.checkpoint) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        self.do_vis =True
        self.start_epoch = 0
        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).to(device)


        self.newimg = False
        self.NBV_step = 0
        self.is_uncer_cal = False
        self.is_model2_save = False
        
   

    def add_img(self, img, depth_img, normal, pose, test):
        self.train_dataset.add_img(img, depth_img, normal, pose)
        self.add_img_index += 1
        self.newimg = True

 
    def terminate_work(self):
        self.terminate = True
    
    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def get_uncertainty(self,poses,device = device,NUM=1000):
        self.is_uncer_cal = True
        # print("pose = ",poses[0])
        t0 = time.time()
        sample_num = NUM
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        indices = torch.tensor(0).to(device)
        res = []
        ray_dirs_all = []
        cam_loc_all = []
        depth_scale_all = []
        poses_all = []
        uncers = 0.0
        sequences = torch.randint(0,self.total_pixels,(len(poses),sample_num))
        for i in range(len(poses)):
            ts = time.time()
            pose = torch.from_numpy(poses[i][np.newaxis,...]).float().to(device)
            sampling_idx = sequences[i]
            model_input = {
                "uv": uv[sampling_idx, :][np.newaxis,...].to(device),
                "intrinsics": self.train_dataset.intrinsic[np.newaxis,...].to(device),
                "pose": pose
            }

            sample = self.get_sampler(model_input, device=device)
            
            ray_dirs_all.append(sample["ray_dirs"])
            cam_loc_all.append(sample["cam_loc"])
            depth_scale_all.append(sample["depth_scale"])
            poses_all.append(sample["pose"])
        
        ray_dirs_all = torch.cat(ray_dirs_all,0)
        cam_loc_all = torch.cat(cam_loc_all,0)
        depth_scale_all = torch.cat(depth_scale_all,0)
        poses_all = torch.cat(poses_all,0)
        print(ray_dirs_all.shape,cam_loc_all.shape,depth_scale_all.shape,poses_all.shape)
        
        
        sample_rays_num = 4000
        iter_num = int(NUM*len(poses)/sample_rays_num)
        print(iter_num)
        for k in range(iter_num+1):
            start = k*sample_rays_num
            end = min((k+1)*sample_rays_num,NUM*len(poses))
            if start >= end:
                break
            ray_dirs = ray_dirs_all[start:end]
            cam_loc = cam_loc_all[start:end]
            depth_scale = depth_scale_all[start:end]
            pose = poses_all[start:end]
            sample = {
                    "ray_dirs": ray_dirs,
                    "cam_loc": cam_loc,
                    "depth_scale": depth_scale,
                    "pose":pose    
                }
                
            out = self.model2(sample, indices,device=device,is_uncer_only=True)
            if 'uncer_map' in out:
                uncer = out['uncer_map'].detach()
                res.append(uncer)
        uncers = torch.cat(res,-1)
        # print(uncers.shape)
        uncers = uncers.reshape((len(poses),-1) )
        uncers = torch.sum(uncers,-1)/sample_num
        print(uncers.shape)
        self.is_uncer_cal = False
        torch.cuda.empty_cache()

        return uncers#,avg_distance,ratio
        
    def info_planner(self):
        # response = requests.get("http://192.168.31.18:7300/isfinish")
        response = requests.get("http://127.0.0.1:7300/isfinish")
        # response = requests.get("http://127.0.0.1:7300/isfinish")
        print(response)
        print("send info to planner")
        # start_timer()

    def train(self):

        self.iter_step = 0
        print(self.add_img_index)
        currindex = self.add_img_index - 1
        torch.cuda.empty_cache()
        PSNRs,PSNRs_test,Uncers_test = [],[0],[0]

        ts = time.time()
        while True :
            
            if self.is_model2_save:
                continue
            
            if self.iter_step == 80:
                self.info_planner()

            if self.newimg:
                self.newimg = False
                currindex = self.add_img_index - 1
                
            self.train_dataset.change_sampling_idx(self.num_pixels)
            if self.iter_step % 2 == 0 and self.iter_step < 1000:
                min_index = max(0,currindex - 9)
                max_index = self.add_img_index
                n_index = np.random.randint(min_index,max_index,size=1)[0]
            else :
                n_index = np.random.randint(self.add_img_index,size=1)[0]
            
            # print(n_index,self.add_img_index,len(self.train_dataset))

            indices, model_input, ground_truth = self.train_dataset.get_testdata(n_index,data_type = "train")
            index = indices.cpu().numpy()[0]
            model_input["intrinsics"] = model_input["intrinsics"].to(device)
            model_input["uv"] = model_input["uv"].to(device)
            c2w = self.pose_all_est[index]
            model_input['pose'] = c2w.to(device)[None,:,:]

            sample = self.get_sampler(model_input,device=device)
            # print("t1_0 =",time.time()-ts)
            model_outputs = self.model(sample, indices)
            loss_output = self.loss(model_outputs, ground_truth)
            loss = loss_output['loss']
            # with torch.autograd.detect_anomaly():
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if self.iter_step%50==0 and self.iter_step>0:
                # print(self.GPU_INDEX)
                if self.GPU_INDEX == 0 and ((self.iter_step%1000==0 and self.iter_step<10000) 
                   or (self.iter_step%5000==0 and self.iter_step>=10000)):
                 
                    self.model.eval()
                    self.model2.load_state_dict(self.model.state_dict())
                    # self.rensder_test(self.iter_step,"train", n_index,device2)
                    # threading.Thread(target=self.render_test,args=(self.iter_step,"train", n_index,device2)).start()
                    self.save_checkpoints(self.iter_step)

                    self.model.train()
                # print(n_index)
            
            if self.iter_step >= 150000:
                break


            if self.GPU_INDEX == 0:
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],ground_truth['rgb'].to(device).reshape(-1,3))
                if self.iter_step % 100 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7},smooth_loss = {8}, depth_loss = {9}, normal_l1 = {10}, normal_cos = {11}, psnr = {12},uncer = {13}, bete={14}, alpha={15},speed={16}'
                            .format(self.expname, self.timestamp,  self.add_img_index, self.iter_step, self.nepochs* self.train_dataset.n_images, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    loss_output['smooth_loss'].item(),
                                    loss_output['depth_loss'].item(),
                                    loss_output['normal_l1'].item(), 
                                    loss_output['normal_cos'].item(),
                                    psnr.item(),
                                    loss_output['uncer'].item(),
                                    self.model.density.get_beta().item(),
                                    1. / self.model.density.get_beta().item(),
                                    (self.iter_step+1)/(time.time()-ts)))
                # prinst("time = ", time.time()-t0)
                logfolder = os.path.join(self.plots_dir, 'logs')
                pose_loss = 0.0
                data = np.array([[self.iter_step,time.time()-ts,psnr.item(),loss.item(),loss_output['rgb_loss'].item(),loss_output['eikonal_loss'].item(),\
                                loss_output['smooth_loss'].item(),loss_output['depth_loss'].item(), loss_output['normal_l1'].item(), loss_output['normal_cos'].item(),\
                                pose_loss,loss_output['uncer'].item(),\
                                self.model.density.get_beta().item(),1. / self.model.density.get_beta().item()]])

                with open(f'{logfolder}/loss.txt', "ab") as f:
                    np.savetxt(f, data)
                
                if self.iter_step % 20 == 0 and self.is_uncer_cal == False :  
                    # tms = time.time()
                    self.model.eval()
                    self.model2.load_state_dict(self.model.state_dict())
                    self.model.train() 
                    # print("tm = ",time.time()-tms)
                
                    


            self.iter_step = self.iter_step+1
    


    def render_test(self,epoch,data_type = "train",index = 0,device=device):
        ts = time.time()

        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = self.train_dataset.get_testdata(index,data_type = data_type)
        print(indices)
        model_input["intrinsics"] = model_input["intrinsics"].to(device)
        model_input["uv"] = model_input["uv"].to(device)
        model_input['pose'] = model_input['pose'].to(device)

        sample = self.get_sampler(model_input, device=device)
        
        # print("sample = ",sample["ray_dirs"].shape,sample["cam_loc"].shape,sample["depth_scale"].shape)
        split = utils.split_input(sample, self.total_pixels, n_pixels=self.split_n_pixels,device=device)
        # print("t1 =",time.time()-ts)
        res = []
        # print(model_input)
        for s in tqdm(split):
            # print("t1-1 =",time.time()-ts)
            out = self.model2(s, indices,device=device)
            # print("t1-2 =",time.time()-ts)
            d = {'rgb_values': out['rgb_values'].detach(),
                    'normal_map': out['normal_map'].detach(),
                    'depth_values': out['depth_values'].detach()}
            if 'rgb_un_values' in out:
                d['rgb_un_values'] = out['rgb_un_values'].detach()
            
            if 'uncer_map' in out:
                d['uncer_map'] = out['uncer_map'].detach()
            res.append(d)
        # torch.cuda.empty_cache()
        print("t_all =",time.time()-ts)
        # print(res)
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        # print(model_outputs)
        depth_map = model_outputs["depth_values"].reshape(384,384).cpu().numpy()*(-5.244145)
        # print("depth_map shape = ", depth_map.shape,np.max(depth_map),np.min(depth_map))
        # print(depth_map[::10,::10])
        depth_map = (depth_map/10.0*65535.0).astype(np.uint16)
        
        
        plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'],device=device)

        uncer_map = plot_data["uncer_map"].reshape(384,384).cpu().numpy()
        uncer =  torch.sum(plot_data["uncer_map"],-1)/self.total_pixels
        # print(np.max(uncer_map),np.min(uncer_map))
        # print(uncer_map)
        uncer_data = 1/uncer_map
        rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
        
    
        plt.plot_test(self.model2.implicit_network,
                indices,
                plot_data,
                self.plots_dir,
                epoch,
                self.img_res,
                **self.plot_conf,
                data_type = data_type,
                device = device
                )

        if data_type == "test_all":
            print(self.plots_dir)
            imageio.imwrite('{0}/test_all/{1}/renderdepth_{2}.png'.format(self.plots_dir, epoch, indices[0]),depth_map)
            imageio.imwrite('{0}/test_all/{1}/uncer_{2}.png'.format(self.plots_dir, epoch, indices[0]),rgb8)

        torch.cuda.empty_cache()
        return uncer


    def get_all_uncertainty(self,location,u,v,epoch,data_type = "train",index = 0,device=device):
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = self.train_dataset.get_testdata(index,data_type = data_type)
        print(indices)
        model_input["intrinsics"] = model_input["intrinsics"].to(device)
        model_input["uv"] = model_input["uv"].to(device)
        print( model_input["uv"], model_input["uv"].shape)

        pose = get_pose(location,u,v)[np.newaxis,...]
        # print(pose,pose.shape)
        model_input['pose'] = torch.from_numpy(pose).float().to(device)
        
        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels,device=device)

        res = []
        for s in tqdm(split):   
            out = self.model2(s, indices,device=device)
            d = {'rgb_values': out['rgb_values'].detach(),
                    'normal_map': out['normal_map'].detach(),
                    'depth_values': out['depth_values'].detach()}
            if 'rgb_un_values' in out:
                d['rgb_un_values'] = out['rgb_un_values'].detach()
            
            if 'uncer_map' in out:
                d['uncer_map'] = out['uncer_map'].detach()
            res.append(d)

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'],device=device)
        
        print("uncer shape = ",plot_data["uncer_map"].shape)
        uncer =  torch.sum(plot_data["uncer_map"],-1)/self.total_pixels
        uncer_map = plot_data["uncer_map"].reshape(384,384).cpu().numpy()
        uncer_data = 1/uncer_map
        rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
        

        plt.plot(self.model2.implicit_network,
                indices,
                plot_data,
                self.plots_dir,
                epoch,
                self.img_res,
                **self.plot_conf,
                data_type = data_type,
                is_vs= True,
                device = device
                )

        if data_type == "test_all":
            imageio.imwrite('{0}/test_all/uncer_{1}_{2}.png'.format(self.plots_dir, epoch, indices[0]),rgb8)

        torch.cuda.empty_cache()
        time.sleep(0.5)
        
        return uncer


    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt,device=device):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        # print(depth_map)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        uncer_map = model_outputs['uncer_map']

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs,device=device)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs,device=device)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'uncer_map': uncer_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs,device=device):
        self.backproject = self.backproject.to(device)
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()

    def get_sampler(self, model_input, device = device):
        ray_dirs, cam_loc = rend_util.get_camera_params(model_input["uv"], model_input["pose"], 
                                                             model_input["intrinsics"],device=device)
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(model_input["uv"], torch.eye(4).to(device)[None], 
                                                        model_input["intrinsics"],device=device)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        # print(model_input['pose'].shape)
        
        sample = {
                "ray_dirs": ray_dirs[0],
                "cam_loc": cam_loc.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1).reshape(-1, 3),
                "depth_scale": depth_scale,
                "pose": model_input['pose'].repeat(ray_dirs.shape[1],1,1)
                
            }
    
        return sample
    
    def get_surface_uncertainty(self):
        
        # self.model.eval()
        self.model2.load_state_dict(self.model.state_dict())
        self.save_checkpoints(self.iter_step)
        # self.model.train()
                    
        epoch = self.iter_step
        # path = self.plots_dir + "/surface_all" 
        path = os.path.join("/home/drivers/5/czbbzc/repos/monosdf_planning",self.exps_folder_name, self.expname, self.timestamp, 'plots',"surface_all" )
        if not os.path.exists(path):
            os.makedirs(path)
        indices = torch.tensor(0).to(device)
        
        surface_traces = plt.get_surface_uncertainty(path=path,
                                indices=indices, 
                                epoch=epoch,
                                model=self.model2,
                                resolution=256,
                                grid_boundary=self.plot_conf.get('grid_boundary'),
                                level=0,
                                device =device2
                                )
        
        self.is_model2_save = False
        path = '{0}/surface_{1}.txt'.format(path, epoch)
        
        return epoch, path


def config_parser():
    parser = configargparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='hm3d_00804_im_grids.conf') # childroom_im_grids.conf
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='test_0', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='8000', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=0, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--render_only', type=int, default=0, help='render only for test views')

    

    return parser