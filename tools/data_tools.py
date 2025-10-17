import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from Dataset import STDataset,TrafficDataset
import os
# from Dataset import Auxility_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml,logging
# # from models.GWNET import gwnet
# from models.MTGNN import MTGNN
# from models import TGCN,ASTGCNCommon,CCRNN,STGCN,AGCRN,STTN,DCRNN
import copy
class CustomStandardScaler:
    def __init__(self, axis=None):
        self.axis = axis
        self.mean = None
        self.std = None

    def fit(self, data):
        if self.axis is None:
            # If axis is not specified, calculate mean and std over the entire data
            self.mean = np.mean(data)
            self.std = np.std(data)
        else:
            # Calculate mean and std along the specified axis
            self.mean = np.mean(data, axis=self.axis)
            self.std = np.std(data, axis=self.axis)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' method first.")
        
        # Standardize the data using the calculated mean and std
        standardized_data = (data - self.mean) / self.std
        return standardized_data
    
    def inverse_transform(self, standardized_data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' method first.")
        
        # Reverse the standardization process
        original_data = standardized_data * self.std + self.mean
        return original_data
def filter_data(data):
    daily_totals = np.sum(data, axis=(0,2, 3))
    
    # 找到总数不为0的天的索引
    valid_days_idx = np.where(daily_totals > 0)[0]
    
    # 根据索引创建新的data数组
    return data[:,valid_days_idx]
def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1

    for index in range(day_start//interval, data.shape[0]+day_start//interval):
        if (index) % time_slot == 0 and index!=0:
            day_init = 0
        day_init = day_init + interval
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

def load_data( config):
    # List all the available files in the data directory


    # Sort the files by name to ensure chronological order
    # config=vars(config)


    train_data = np.load(os.path.join('dataset\\',config['dataset_name'], 'train.npy'))
    val_data = np.load(os.path.join('dataset\\',config['dataset_name'], 'val.npy'))
    test_data = np.load(os.path.join('dataset\\',config['dataset_name'], 'test.npy'))
    # print(train_data.shape)
    
    # if config['dataset_name'] == 'nycbike':
        
    #     valid_grid=np.load('data\\nycbike\\valid_grid_bike.npy')
    # else:
    # if config['model']!='UniST':
        # if config['dataset_name'] in ['bosbike','baybike','torbike','chitaxi','chibike']:z9[;90                   ]            valid_grid=np.where(train_data.mean(axis=(0,2))>2)[0]
        #     train_data,val_data,test_data=train_data[:,valid_grid],val_data[:,valid_grid],test_data[:,valid_grid]  #(hour_num,grid_num,feature_num)
        #     train_data=train_data.transpose(1,0,2).reshape(train_data.shape[1],-1,24,2)    #(grid_num,day_num,24，feature_num)
        #     val_data=val_data.transpose(1,0,2).reshape(val_data.shape[1],-1,24,2)
        #     test_data=test_data.transpose(1,0,2)[:,:(test_data.shape[0]//24)*24].reshape(test_data.shape[1],-1,24,2)
        # else:
        #     valid_grid=np.where(train_data.mean(axis=(1,2,3))>2)[0]
        #     train_data,val_data,test_data=train_data[valid_grid],val_data[valid_grid],test_data[valid_grid]
    #     pass
    # else:
    if config['dataset_name'] in ['bosbike','baybike','torbike','chitaxi','chibike']:
        valid_grid=np.where(train_data.mean(axis=(0,2))>2)[0]
        train_data,val_data,test_data=train_data[:,valid_grid],val_data[:,valid_grid],test_data[:,valid_grid]  #(hour_num,grid_num,feature_num)
        train_data=train_data.transpose(1,0,2).reshape(train_data.shape[1],-1,24,2)    #(grid_num,day_num,24，feature_num)
        val_data=val_data.transpose(1,0,2).reshape(val_data.shape[1],-1,24,2)
        test_data=test_data.transpose(1,0,2)[:,:(test_data.shape[0]//24)*24].reshape(test_data.shape[1],-1,24,2)
    else:
        valid_grid=np.where(train_data.mean(axis=(1,2,3))>2)[0]
        train_data,val_data,test_data=train_data[valid_grid],val_data[valid_grid],test_data[valid_grid]
    
    




   
    
   
        
    scaler = CustomStandardScaler()  # Specify the axis over which to calculate mean and std
    scaler.fit(train_data)

    # Standardize the data
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    if 'tc_num_heads' in config.keys():
        test_data=test_data.transpose(1,2,0,3)
        test_data=test_data.reshape(test_data.shape[0]*test_data.shape[1],-1)
        week_start = 3
        interval = 5
        week_day = 7
        holiday_list = None
        interval = 60
  
        day_data, week_data, holiday_data = time_add(test_data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
        test_data= np.expand_dims(test_data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        test_data = np.concatenate([test_data, day_data, week_data], axis=-1)
        train_data=train_data.transpose(1,2,0,3)
        train_data=train_data.reshape(train_data.shape[0]*train_data.shape[1],-1)
        day_data, week_data, holiday_data = time_add(train_data, 2, interval=interval, weekday_only=False, holiday_list=holiday_list)
        train_data= np.expand_dims(train_data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        train_data = np.concatenate([train_data, day_data, week_data], axis=-1)



    return train_data, val_data, test_data, scaler,valid_grid

def get_datasets( config):
    # Load and preprocess the data using load_data function
    train_data, val_data, test_data, scaler,valid_gird = load_data( config)
    # if config.model=='iVAE':
    #     train_dataset=iVAE.iVAEDataset(train_data,config)
    #     val_dataset=iVAE.iVAEDataset(val_data,config)
    #     test_dataset=iVAE.iVAEDataset(test_data,config)
    #     return train_dataset, val_dataset, test_dataset, scaler,valid_gird

    # Create datasets using the STDataset class
    train_dataset = STDataset(train_data, config)
    val_dataset = STDataset(val_data, config,if_train=False,index=len(train_dataset)//(30*24))
    if 'tc_num_heads' in config.keys():
        test_dataset=TrafficDataset(test_data,batch_size=1)
        train_dataset=TrafficDataset(train_data,batch_size=1)
    else:
        test_dataset = STDataset(test_data, config,if_train=False,index=len(train_dataset)//(30*24))

    return train_dataset, val_dataset, test_dataset, scaler,valid_gird

def expand_adjacency_matrix(adj_matrix, m):
    n = adj_matrix.shape[0]
    
    if m < n:
        m=n
    
    expanded_adj_matrix = np.zeros((m, m), dtype=int)
    expanded_adj_matrix[:n, :n] = adj_matrix
    
    # Add self-loops
    np.fill_diagonal(expanded_adj_matrix, 1)
    
    return expanded_adj_matrix-np.eye(len(expanded_adj_matrix))
from typing import Union
class Dataset_Recent(Dataset):
    def __init__(self, dataset, gap: Union[int, tuple, list], recent_num=1, take_post=0, strength=0, **kwargs):
        super().__init__()
        self.more = gap - recent_num + 1
        self.dataset = dataset
        self.gap = gap
        self.recent_num = recent_num
        if strength:
            print("Modify time series with strength =", strength)
            for i in range(3, len(self.dataset.data_y)):
                self.dataset.data_x[i] *= 1 + 0.1 * (i // 24 % strength)

    def _stack(self, data):
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        else:
            return torch.stack(data, 0)

    def __getitem__(self, index):
        if self.recent_num == 1:
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            current_data = self.dataset[index + self.gap + self.recent_num - 1]
            if not isinstance(current_data, tuple):
                recent_data = tuple(self.dataset[index + n] for n in range(self.recent_num))
                recent_data = self._stack(recent_data)
                return current_data, recent_data
            else:
                recent_data = tuple([] for _ in range(len(current_data)))
                for past in range(self.recent_num):
                    for j, past_data in enumerate(self.dataset[index + past]):
                        recent_data[j].append(past_data)
                recent_data = tuple(self._stack(recent_d) for recent_d in recent_data)
            return recent_data, current_data

    def __len__(self):
        return len(self.dataset) - self.more
# def align_data(data):
#     # align  data to have the same mean and variance of data[:,:360,:,:]
#     #data (node_number,day_number,24,2)
    
#     target_mean = data[:,-365:].mean()
#     target_std = data[:,:-365:].std()
#     day_number=data.shape[1]
    
#     for i in range(day_number//365):
#         this_mean = data[:,i*365:(i+1)*365,:,:].mean()
#         this_std = data[:,i*365:(i+1)*365,:,:].std()
        
#         data[:,i*365:(i+1)*365,:,:] = (data[:,i*365:(i+1)*365,:,:]-this_mean)/this_std*target_std+target_mean
        
#     return data

# if __name__ == '__main__':
#     # Test the functions
#     with open('models\config2.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     train_data, val_data, test_data, scaler,valid_grid=load_data(config['data_dir'],config)
#     align_data(train_data)