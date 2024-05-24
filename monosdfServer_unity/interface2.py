import threading
import requests
import numpy as np
import sys
import json
import torch
from torch.functional import Tensor
import time


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def send_NBV(path_view):
    n = path_view.shape[0]
    nbv_data = []
    for i in range(n):
        u = path_view[i,3]
        v = path_view[i,4]
        nbv_data.append({'x':path_view[i,0],'y':path_view[i,1],'z':path_view[i,2],'pitch':u,'yaw':v})
    nbv_json = {'length':n,'nbv':nbv_data}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://10.192.133.74:7300/get_nbv/", headers= headers,data=json.dumps(nbv_json))
    return response.text


def send_Path(path):
    data = {'path':path}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://10.13.21.208:7300/get_path/", headers= headers,data=json.dumps(data))
    return response.text



#### 给定空间点及观测方向下的整幅图像的uncertainty
#### location nparray [x,y,z]
#### direction nparray vector 3 [x,y,z]
#### return uncertainty which normnized uncer of whole image
#### return distance float average(reached)
#### return ratio reached/unreachale float
def get_uncertainty(locations,us,vs):
    data = {'locations':locations,'us':us,'vs':vs}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:7000/get_uncertainty/", headers= headers,data=json.dumps(data))
    re = response.json()
    # print(response.status_code)
    # print(response.text)
    # distances = re['distances']
    uncertaintys = re['uncers']
    # ratios = re['ratios']
    # distances = torch.Tensor(distances)
    # uncertaintys = torch.Tensor(uncertaintys)
    distances = []
    ratios = []
    return uncertaintys,distances,ratios




if __name__=='__main__':
    # creat_window()
    # time.sleep(5)
    path_view = [[-1.0,1.0,5.0,0.0,0.0],[1.0,0.0,6.1,0.0,0.0]]
    send_NBV(np.array(path_view))
