import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np

class SPECT(Dataset):
    def __init__(self, image_path, sample_list):
        self.image_path = image_path
        self.sample_list = sample_list
    def __getitem__(self, index):
        data = loadmat(os.path.join(self.image_path, self.sample_list[index]))
        x = data['x1']        
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(data['y1'], dtype=torch.float).squeeze()
        return x, y

    def __len__(self):
        return len(self.sample_list)

class SPECT_LAD(Dataset):
    def __init__(self, image_path, sample_list, get_channel=[0,1,2,3]):
        self.image_path = image_path
        self.sample_list = sample_list
        self.get_channel = get_channel
    def __getitem__(self, index):
        data = loadmat(os.path.join(self.image_path, self.sample_list[index]))
        x = data['x1']
        if self.get_channel != [0,1,2,3]:
            x = x[self.get_channel,:,:,:]
            if len(self.get_channel) == 1:
                x = np.expand_dims(x, axis=0)
                
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(data['y1'][:, 0], dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.sample_list)
    
class SPECT_LCX(Dataset):
    def __init__(self, image_path, sample_list, get_channel=[0,1,2,3]):
        self.image_path = image_path
        self.sample_list = sample_list
        self.get_channel = get_channel
    def __getitem__(self, index):
        data = loadmat(os.path.join(self.image_path, self.sample_list[index]))
        x = data['x1']    
        if self.get_channel != [0,1,2,3]:
            x = x[self.get_channel,:,:,:] 
            if len(self.get_channel) == 1:
                x = np.expand_dims(x, axis=0)   
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(data['y1'][:, 1], dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.sample_list)

class SPECT_RCA(Dataset):
    def __init__(self, image_path, sample_list, get_channel=[0,1,2,3]):
        self.image_path = image_path
        self.sample_list = sample_list
        self.get_channel = get_channel
    def __getitem__(self, index):
        data = loadmat(os.path.join(self.image_path, self.sample_list[index]))
        x = data['x1']   
        if self.get_channel != [0,1,2,3]:
            x = x[self.get_channel,:,:,:]    
            if len(self.get_channel) == 1:
                x = np.expand_dims(x, axis=0) 
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(data['y1'][:, 2], dtype=torch.float)
        return x, y

    def __len__(self):
        return len(self.sample_list)