import torch
import torchvision.datasets as dset
from torch.utils.data import TensorDataset,DataLoader, Dataset
import numpy as np


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data) #들어온 데이터를 텐서로 
        #self.x_data = self.x_data.permute(0,3,1,2) #이미지 개수, 채널 수, 이미지 너비, 높이
        self.y_data = torch.FloatTensor(y_data)  #들어온 데이터를 텐서로 
        self.len = self.y_data.shape[0]
        
    def __getitem__ (self, index): #
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    


def make_sequence_train_dataset(feature, label, window_size, predict_size):
    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list
    for i in range(len(feature)-window_size-predict_size+1):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size:i+window_size+predict_size]) #predict만큼 여러 개 
        #label_list.append(label[i+window_size+predict_size-1:i+window_size+predict_size]) #predict이후 시점 딱 한개 
    return np.array(feature_list), np.array(label_list)
    
    