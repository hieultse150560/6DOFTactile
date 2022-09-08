# threeD_dataLoader.py: Chuẩn bị train và val dataset
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
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


class sample_data_diffTask(Dataset):
    def __init__(self, path, window, subsample, mode):
        self.path = path
        self.touchs = glob.glob(os.path.join(path, "[P]*", "touch_normalized.p"))
        self.keypoints = glob.glob(os.path.join(path, "[P]*", "keypoint_transform.p"))
        self.subsample = subsample
        touch = np.empty((1,96,96))
        heatmap = np.empty((1,21,20,20,18))
        keypoint = np.empty((1,21,3))
        touch = torch.empty((1,96,96))
        heatmap = torch.empty((1,21,20,20,18))
        keypoint = torch.empty((1,21,3))
        xyz_range = [[-100,1900],[-100,1900],[-1800,0]]
        size = [20, 20, 18] #define 3D space
        if mode == "train":
            for i in range(0, len(self.touchs)-4): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print ("Load data from: ", self.touchs[i], self.keypoints[i])
                tactile = np.array(pickle.load(open(self.touchs[i], "rb")))
                keypointN, heatmapN = heatmap_from_keypoint(self.keypoints[i], xyz_range, size)
                # print(tactile.shape, keypointN.shape, heatmapN.shape)
                # print(np.max(tactile), np.min(tactile)) #1.0 and 0.0
#                 touch = np.append(touch, tactile, axis=0) # Đọc dữ liệu và xếp chồng
#                 heatmap = np.append(heatmap, heatmapN, axis=0) # Đọc dữ liệu và xếp chống
#                 keypoint = np.append(keypoint, keypointN, axis=0) # Đọc dữ liệu và xếp chống
                touch = torch.cat((torch.from_numpy(touch), torch.from_numpy(tactile)), 0) # Đọc dữ liệu và xếp chồng
                heatmap = torch.cat((torch.from_numpy(heatmap), torch.from_numpy(heatmapN)), 0) # Đọc dữ liệu và xếp chống
                keypoint = torch.cat((torch.from_numpy(keypoint), torch.from_numpy(keypointN)), 0) # Đọc dữ liệu và xếp chống
            
        elif mode == "val":
            for i in range(len(self.touchs)-4, len(self.touchs) - 2): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print ("Load data from: ", self.touchs[i], self.keypoints[i])
                tactile = np.array(pickle.load(open(self.touchs[i], "rb")))
                keypointN, heatmapN = heatmap_from_keypoint(self.keypoints[i], xyz_range, size)
                # print(tactile.shape, keypointN.shape, heatmapN.shape)
                # print(np.max(tactile), np.min(tactile)) #1.0 and 0.0
#                 touch = np.append(touch, tactile, axis=0) # Đọc dữ liệu và xếp chồng
#                 heatmap = np.append(heatmap, heatmapN, axis=0) # Đọc dữ liệu và xếp chống
#                 keypoint = np.append(keypoint, keypointN, axis=0) # Đọc dữ liệu và xếp chống
                touch = torch.cat((torch.from_numpy(touch), torch.from_numpy(tactile)), 0) # Đọc dữ liệu và xếp chồng
                heatmap = torch.cat((torch.from_numpy(heatmap), torch.from_numpy(heatmapN)), 0) # Đọc dữ liệu và xếp chống
                keypoint = torch.cat((torch.from_numpy(keypoint), torch.from_numpy(keypointN)), 0) # Đọc dữ liệu và xếp chống
        else:
            for i in range(len(self.touchs)-2, len(self.touchs)): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print ("Load data from: ", self.touchs[i], self.keypoints[i])
                tactile = np.array(pickle.load(open(self.touchs[i], "rb")))
                keypointN, heatmapN = heatmap_from_keypoint(self.keypoints[i], xyz_range, size)
                # print(tactile.shape, keypointN.shape, heatmapN.shape)
                # print(np.max(tactile), np.min(tactile)) #1.0 and 0.0
#                 touch = np.append(touch, tactile, axis=0) # Đọc dữ liệu và xếp chồng
#                 heatmap = np.append(heatmap, heatmapN, axis=0) # Đọc dữ liệu và xếp chống
#                 keypoint = np.append(keypoint, keypointN, axis=0) # Đọc dữ liệu và xếp chống
                touch = torch.cat((torch.from_numpy(touch), torch.from_numpy(tactile)), 0) # Đọc dữ liệu và xếp chồng
                heatmap = torch.cat((torch.from_numpy(heatmap), torch.from_numpy(heatmapN)), 0) # Đọc dữ liệu và xếp chống
                keypoint = torch.cat((torch.from_numpy(keypoint), torch.from_numpy(keypointN)), 0) # Đọc dữ liệu và xếp chống

        touch = touch[1:,:,:]
        heatmap = heatmap[1:,:,:,:,:]
        keypoint = keypoint[1:,:,:] # Tất cả data trừ sample đầu tiên
        self.window = window

    def __len__(self):
        # return self.length
        return heatmap.shape[0] # Lấy timestamps của camera làm độ dài dataset

    def __getitem__(self, idx): #idx là iterator
        tactileU = window_select(touch,idx,self.window) # Frame of tactiles
        heatmapU = heatmap[idx,:,:,:,:] # Headmap
        keypointU = keypoint[idx,:,:] # Keypoint
        tactile_frameU = touch[idx,:,:] # Middle Frame

        if self.subsample > 1:
            tactileU = get_subsample(tactileU, self.subsample) # Nếu có chia theo subsample thì tính trung bình cacs pixel theo giá trị subsample

        return tactileU, heatmapU, keypointU, tactile_frameU # Lấy M frames xung quanh 1 middle frame + heatmap + keypoint của middle frame

class sample_data_diffTask2(Dataset):
    def __init__(self, path, window, subsample):
        self.path = path
        self.files = os.listdir(self.path)
        self.subsample = subsample
        touch = np.empty((1,96,96))
        heatmap = np.empty((1,21,20,20,18))
        keypoint = np.empty((1,21,3))
        count = 0

        for f in self.files:
            count += 1
            print (f, count)
            data = pickle.load(open(self.path + f, "rb"))
            touch = np.append(touch, data[0], axis=0)
            heatmap = np.append(heatmap, data[1], axis=0)
            keypoint = np.append(keypoint, data[2], axis=0)


        self.data_in = [touch[1:,:,:], heatmap[1:,:,:,:,:], keypoint[1:,:,:]]
        self.window = window

    def __len__(self):
        # return self.length
        return self.data_in[0].shape[0]

    def __getitem__(self, idx):
        tactile = window_select(self.data_in[0],idx,self.window)
        heatmap = self.data_in[1][idx,:,:,:,:]
        keypoint = self.data_in[2][idx,:,:]
        tactile_frame = self.data_in[0][idx,:,:]

        if self.subsample > 1:
            tactile = get_subsample(tactile, self.subsample)

        return tactile, heatmap, keypoint, tactile_frame
# dataset = sample_data_diffTask('/content/tactile_keypoint_data/tactile_keypoint_data', 10, 1)
# i = 0
# for i_batch, sample_batched in enumerate(dataset):
#   i += 1
#   print(i_batch, len(sample_batched))
#   print(sample_batched[0].shape, sample_batched[1].shape,sample_batched[2].shape,sample_batched[3].shape)
#   #check heatmap viz
#   print ("Time Step: ",i_batch)

#   plt.imshow(sample_batched[3])

#   img1 = plotKeypoint(np.reshape(sample_batched[2], (21,3)))

#   # if shift_to_9tiles:
#   #   img2 = plot3Dheatmap(round_to_1(np.reshape(heatmap[i,:,:,:,:]
#   #                      ,(heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4])), 2), seperate=False)
#   # else:
#   #   img2 = plot3Dheatmap(round_to_1(np.reshape(heatmap[i,:,:,:,:]
#   #                      ,(heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4])), 2), seperate=False)
    
#   plt.show()
#   if i == 10:
#     break
