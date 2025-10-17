import torch
from torch.utils.data import Dataset
import numpy as np
class STDataset(Dataset):
    def __init__(self, data, config,if_train=True,index=100):
        '''
        data:nparray (num_nodes,day_num,24,2)
        '''

        
        # 根据索引创建新的data数组
        self.data = data
        self.data = self.data.reshape(data.shape[0],-1,2)
        self.input_window = 6
        self.output_window = 1
        self.if_train = if_train
        self.index=index

        self.max_index=(self.data.shape[1] - (self.input_window + self.output_window) + 1)//(30*24)
    def __len__(self):
        return self.data.shape[1] - (self.input_window + self.output_window) + 1

    def __getitem__(self, index):
        
        x = self.data[:, index:index + self.input_window, :].transpose(1,0,2)
        y = self.data[:, index + self.input_window:index + self.input_window + self.output_window, :].reshape(-1,self.output_window,2).transpose(1,0,2)
      
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)
class Auxility_dataset(Dataset):
    def __init__(self, dataset, classindex,config):
        super(Auxility_dataset, self).__init__()
        self.input_window = config['input_window']
        self.output_window = config['output_window']
        self.data = dataset
        self.classindex = torch.zeros(self.data.shape[1] - (self.input_window + self.output_window) + 1,dtype=torch.float32)
        self.classindex[-classindex:] = 1
    
    def __getitem__(self, index):
        x = self.data[:, index:index + self.input_window, :].transpose(1,0,2)
        return torch.tensor(x,dtype=torch.float32),self.classindex[index]
    
    def __len__(self):
        return self.data.shape[1] - (self.input_window + self.output_window) + 1
import random
class TrafficDataset(Dataset):
    def __init__(self, data, batch_size, input_window=288, output_window=1, eval_only=1):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

        # preprocess
        self.windows = [
            (data[i:i + input_window], data[i + input_window:i + input_window + output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]

        # drop_last & shuffle
        if eval_only==False:
            random.shuffle(self.windows)
            if len(self.windows) % batch_size != 0:
                self.windows = self.windows[:-(len(self.windows) % batch_size)]


        # batch
        self.batches = [
            self.windows[i:i + batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_x, batch_y = zip(*self.batches[idx])
        return torch.from_numpy(np.stack(batch_x)).float(), torch.from_numpy(np.stack(batch_y)).float()
    
