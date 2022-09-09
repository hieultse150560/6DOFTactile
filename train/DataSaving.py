# threeD_dataLoader.py: Chuẩn bị train và val dataset
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from progressbar import ProgressBar
import logging
import pickle
import glob
from utils import normalize
from heatmap_from_keypoint3D import heatmap_from_keypoint

# Trả về 1 đoạn input signal có size là 2 * window
def window_select(data,timestep,window):
    if window ==0:
        return data[timestep : timestep + 1, :, :]
    max_len = data.shape[0]
    l = max(0,timestep-window) 
    u = min(max_len,timestep+window)
    if l == 0:
        return (data[:2*window,:,:]) # Nếu không đủ data để lùi timestep thì lấy từ đầu x2 windows
    elif u == max_len:
        return (data[-2*window:,:,:]) # Nếu không đủ data để tiến thì lấy từ cuối x2 windows
    else:
        return(data[l:u,:,:]) # Nếu đủ thì lấy x2 window với vị trí giữa là timestep


def get_subsample(touch, subsample): # Tính trung bình theo từng cụm subsample * subsample size theo chiều đầu tiên và thay thế 
    for x in range(0, touch.shape[1], subsample):
        for y in range(0, touch.shape[2], subsample):
            v = np.mean(touch[:, x:x+subsample, y:y+subsample], (1, 2))
            touch[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

    return touch


class sample_data_diffTask_Data(Dataset):
    def __init__(self, path, window, subsample, index):
        self.path = path
        self.touchs = glob.glob(os.path.join(path, "[P]*", "touch_normalized.p"))
        self.keypoints = glob.glob(os.path.join(path, "[P]*", "keypoint_transform.p"))
        self.subsample = subsample
        self.index = index
        touch = np.empty((1,96,96))
        heatmap = np.empty((1,21,20,20,18))
        keypoint = np.empty((1,21,3))
        xyz_range = [[-100,1900],[-100,1900],[-1800,0]]
        size = [20, 20, 18] #define 3D space
        print ("Load data from: ", self.touchs[self.index], self.keypoints[self.index])
        tactile = np.array(pickle.load(open(self.touchs[self.index], "rb")))
        keypointN, heatmapN = heatmap_from_keypoint(self.keypoints[self.index], xyz_range, size)
        touch = np.append(touch, tactile, axis=0) # Đọc dữ liệu và xếp chồng
        heatmap = np.append(heatmap, heatmapN, axis=0) # Đọc dữ liệu và xếp chống
        keypoint = np.append(keypoint, keypointN, axis=0) # Đọc dữ liệu và xếp chống

        self.touch = touch[1:,:,:]
        self.heatmap = heatmap[1:,:,:,:,:]
        self.keypoint = keypoint[1:,:,:] # Tất cả data trừ sample đầu tiên
        self.window = window

    def __len__(self):
        # return self.length
        return self.heatmap.shape[0] # Lấy timestamps của camera làm độ dài dataset

    def __getitem__(self, idx): #idx là iterator
        tactileU = window_select(self.touch,idx,self.window) # Frame of tactiles
        heatmapU = self.heatmap[idx,:,:,:,:] # Headmap
        keypointU = self.keypoint[idx,:,:] # Keypoint
        tactile_frameU = self.touch[idx,:,:] # Middle Frame

        if self.subsample > 1:
            tactileU = get_subsample(tactileU, self.subsample) # Nếu có chia theo subsample thì tính trung bình cacs pixel theo giá trị subsample

        return tactileU, heatmapU, keypointU, tactile_frameU # Lấy M frames xung quanh 1 middle frame + heatmap + keypoint của middle frame

num = 4
train_dataset = sample_data_diffTask_Data("./tactile_keypoint_data/", 10, 1, num)
train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True)
bar = ProgressBar(max_value=len(train_dataloader))
print ("Training set size:", len(train_dataset))
with open("index.p", "rb") as f:
  i = pickle.load(f)
for sample_batched in bar(train_dataloader):
  with open(f"./batch_data/train/{i}.p", "wb") as f2:
    pickle.dump(sample_batched, f2)
    print(f"Saving to {i}.p file") 
  i += 1
with open("index.p", "wb") as f3:
  pickle.dump(i, f3)
  print(f"Saving index: {i}")
