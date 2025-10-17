import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
class OGD:
    ## online gradient descent
    def __init__(self, model: nn.Module,lr=0.01,device='cuda:0'):
        self.model=model
        self.lr=lr
        self.optimizer=optim.SGD(model.parameters(),lr=lr)
        self.device=device
    
    def online(self,test_loader:torch.utils.data.DataLoader):
        preds=[]
        labels=[]
        for data in tqdm.tqdm(test_loader):
            
            
            for i in range(5):
                self.optimizer.zero_grad()
                x,y=data
                x=x.to(self.device)
                y=y.to(self.device)
            
                output=self.model(x)
                if i==0:
                    preds.append(output.detach().cpu().numpy())
                loss=F.mse_loss(output,y)
                loss.backward()
                self.optimizer.step()
                
            labels.append(y.detach().cpu().numpy())
        return preds,labels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import tqdm

class ER:
    """Experience Replay with buffer to store historical data batches."""
    def __init__(self, model: nn.Module, lr=0.01, buffer_size=100, device='cuda:0'):
        self.model = model
        self.lr = lr
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.device = device
        self.buffer = deque(maxlen=buffer_size)  # 固定大小的经验缓冲区
    
    def online(self, test_loader: torch.utils.data.DataLoader):
        """Process test data with experience replay."""
        preds = []
        labels = []
        for data in tqdm.tqdm(test_loader):
            self.optimizer.zero_grad()
            
            # 处理当前批次数据
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 将当前批次存入缓冲区（自动淘汰旧数据）
            self.buffer.append((x.detach().clone(), y.detach().clone()))
            
            # 从缓冲区随机采样旧数据（至少1个批次）
            replay_x, replay_y = [], []
            if len(self.buffer) > 0:
                num_replay = min(1, len(self.buffer))  # 采样1个旧批次
                replay_samples = random.sample(self.buffer, num_replay)
                replay_x, replay_y = zip(*replay_samples)
                replay_x = torch.cat(replay_x, dim=0)
                replay_y = torch.cat(replay_y, dim=0)
            
            # 合并当前数据与回放数据
            if len(replay_x) > 0:
                combined_x = torch.cat([x, replay_x], dim=0)
                combined_y = torch.cat([y, replay_y], dim=0)
            else:
                combined_x, combined_y = x, y
            
            # 前向传播与损失计算
            output = self.model(combined_x)
            loss = F.mse_loss(output, combined_y)
            
            # 反向传播与参数更新
            loss.backward()
            self.optimizer.step()
            
            # 记录预测结果
            preds.append(output[0].detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        
        return preds, labels
    
class OGD_output:
    ## online gradient descent for output
    def __init__(self, model: nn.Module,num_node,lr=0.01,device='cuda:0'):
        self.model=model
        self.lr=lr
        self.delta=nn.Parameter(torch.zeros((1,1,num_node,2),device=device),requires_grad=True)
        self.optimizer=optim.SGD([self.delta],lr=lr)
        self.device=device

    def online(self,test_loader:torch.utils.data.DataLoader):
        preds=[]
        labels=[]
        for data in tqdm.tqdm(test_loader):
            self.optimizer.zero_grad()
            x,y=data
            x=x.to(self.device)
            y=y.to(self.device)
            output=self.model(x)+self.delta
            loss=abs(output-y).mean()
            loss.backward()
            self.optimizer.step()
            preds.append(output.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        return preds,labels
    
if __name__ == "__main__":
    model='GWNET'
    train_month=12
    import yaml
    import os
    import yaml
    import torch
    import numpy as np
    from sklearn.linear_model import Ridge
    from collections import defaultdict, deque
    import tqdm
    import numpy as np
    from torch.utils.data import DataLoader
    from datetime import datetime
    from models import STGCN,AGCRN,DCRNN,GWNET,MTGNN,causal_model
    from tools.data_tools import load_data, get_datasets, expand_adjacency_matrix
    from Dataset import STDataset
    import csv
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import logging
    print(torch.version.cuda)
    model_dict={'STGCN':STGCN,'AGCRN':AGCRN,'DCRNN':DCRNN,'GWNET':GWNET,'MTGNN':MTGNN}
    with open(r'models\nyctaxi_config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
    config['model'],config['train_months']=model,train_month
    # Configure logging
    # log_dir = config['log_dir']  # You can specify the directory for log files


    # logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.info('{}_training_month_{}_.log'.format(config['model'],config['train_months']))



    # Check if a GPU is available, otherwise use CPU
    device = config['device']

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets( config)
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset,1, shuffle=False)
    config['adj_mx_file']=os.path.join('dataset\\',config['dataset_name'], 'adj_mx.npy')
    adj_mx =  np.load(config['adj_mx_file'])[valid_gird][:,valid_gird]
    config['num_nodes']=len(valid_gird)
    # 初始化基础模型（假设已有STGCN模型）
    base_stgcn = causal_model.CausalModel(model_dict[config['model']].Model,config,adj_mx).to(device)
    base_stgcn.eval()
    base_stgcn.load_state_dict(torch.load(f'saved_models\{model}_nyctaxi_final_model.pth'))
    ogd=OGD(base_stgcn,lr=0.01,device='cuda:0')
    preds, labels = ogd.online(test_loader)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    preds = scaler.inverse_transform(preds).reshape(-1)
    labels = scaler.inverse_transform(labels).reshape(-1)
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


      