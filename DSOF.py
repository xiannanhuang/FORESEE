import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm
import numpy as np
import torch
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error
model='STGCN'
train_month=12
import yaml
import os
import yaml
import torch
import numpy as np
from sklearn.linear_model import Ridge
from collections import defaultdict, deque
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from models import STGCN,AGCRN,DCRNN,GWNET,MTGNN,causal_model
from tools.data_tools import load_data, get_datasets, expand_adjacency_matrix
from Dataset import STDataset
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from collections import deque
import numpy as np
model_dict={'STGCN':STGCN,'AGCRN':AGCRN,'DCRNN':DCRNN,'GWNET':GWNET,'MTGNN':MTGNN}
def init_model(model_name='STGCN',model_path='saved_models\STGCN_nycbike_train_months_12final_model.pth',config_path=r'models\nycbike_config.yaml'):
    
    with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

    device = config['device']

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets( config)
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset,1, shuffle=False)
    config['adj_mx_file']=os.path.join('dataset\\',config['dataset_name'], 'adj_mx.npy')
    adj_mx =  np.load(config['adj_mx_file'])[valid_gird][:,valid_gird]
    config['num_nodes']=len(valid_gird)
    model=causal_model.CausalModel(model_dict[model_name].Model,config,adj_mx).to(device)
    model.load_state_dict(torch.load(model_path))
    return model,train_loader,val_loader,test_loader,scaler,config,adj_mx

class StudentModel(nn.Module):
    def __init__(self, input_len, output_len, num_nodes):
        super(StudentModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        
        # 学生网络：轻量级模型，处理残差
        self.fc = nn.Sequential(
            nn.Linear((input_len + output_len) * num_nodes * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_len * num_nodes * 2)
        )
        
    def forward(self, x_hist, teacher_pred):
        """学生网络前向传播
        Args:
            x_hist (torch.Tensor): 历史数据 (B, L, N, 2)
            teacher_pred (torch.Tensor): 教师预测 (B, H, N, 2)
        Returns:
            torch.Tensor: 残差预测 (B, H, N, 2)
        """
        B = x_hist.shape[0]
        # 展平历史数据和教师预测
        x_flat = x_hist.reshape(B, -1)
        t_flat = teacher_pred.reshape(B, -1)
        
        # 拼接作为学生网络输入
        x = torch.cat([x_flat, t_flat], dim=1)
        out = self.fc(x)
        
        # 重塑为输出形状
        return out.reshape(B, self.output_len, self.num_nodes, 2)

class DSOF(nn.Module):
    def __init__(self, base_model: nn.Module, input_len, output_len, num_nodes, device,
                 buffer_size=100, batch_size=32, gamma=0.7):
        """
        DSOF在线预测模型
        
        Args:
            base_model: 基础教师模型 (B, L, N, 2) -> (B, H, N, 2)
            input_len: 输入序列长度L
            output_len: 预测序列长度H
            num_nodes: 区域节点数
            device: 运行设备
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            gamma: 时间差分衰减因子
        """
        super(DSOF, self).__init__()
        self.base_model = base_model
        self.student = StudentModel(input_len, output_len, num_nodes).to(device)
        self.device = device
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        
        # 在线学习参数
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 不同优化器设置
        self.er_optimizer = optim.Adam(
            list(self.base_model.parameters()) + list(self.student.parameters()),
            lr=1e-4
        )
        self.td_optimizer = optim.Adam(
            self.student.parameters(),
            lr=1e-3
        )
        
    def forward(self, x_hist):
        """完整前向预测（教师+学生）
        Args:
            x_hist (torch.Tensor): 历史数据 (B, L, N, 2)
        Returns:
            torch.Tensor: 最终预测 (B, H, N, 2)
        """
        with torch.no_grad():
            teacher_pred = self.base_model(x_hist)
        student_pred = self.student(x_hist, teacher_pred)
        return teacher_pred + student_pred
    
    def _teacher_prediction(self, x_hist):
        """单独教师预测"""
        return self.base_model(x_hist)
    
    def online(self, test_loader, scaler):
        """在线预测流程（含双流更新）
        Args:
            test_loader: 测试数据加载器
            scaler: 数据归一化器
        Returns:
            tuple: 预测结果和真实值
        """
        self.base_model.train()
        self.student.train()
        preds, truths = [], []
        batch_idx=0
        # 迭代处理每个时间点
        for  (batch_x, batch_y) in tqdm.tqdm(test_loader):
            batch_x = batch_x.to(self.device)  # (B, L, N, 2)
            batch_y = batch_y.to(self.device)  # (B, H, N, 2)
            
            # === 1. 模型预测当前时序 ===
            teacher_pred = self.base_model(batch_x)
            student_pred = self.student(batch_x, teacher_pred)
            total_pred = teacher_pred + student_pred
            preds.append(total_pred.detach().cpu().numpy())
            truths.append(batch_y.detach().cpu().numpy())
            
            # === 2. 经验回放更新（慢速流）===
            if len(self.buffer) >= self.batch_size+10000000:
                # 从缓冲区随机采样
                indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
                batch_samples = [self.buffer[i] for i in indices]
                
                # 解压数据
                x_batch, y_batch = zip(*batch_samples)
                x_batch = torch.concatenate(x_batch,dim=0).to(self.device)
                y_batch = torch.concatenate(y_batch,dim=0).to(self.device)
                
                # 教师+学生联合预测
                teacher_batch = self.base_model(x_batch)
                student_batch = self.student(x_batch, teacher_batch)
                pred_batch = teacher_batch + student_batch
                
                # 计算损失并更新
                loss_er = nn.MSELoss()(pred_batch, y_batch)
                self.er_optimizer.zero_grad()
                loss_er.backward()
                self.er_optimizer.step()
            
            # === 3. 时间差分更新（快速流）===
            if batch_idx > 10000000:  # 需至少有一次历史预测
                # 获取历史预测（上一时间点）
                prev_x, prev_pred = self.buffer[-1]
                
                # 构建伪标签 [y_t, teacher_pred_t+1:t+H]
                # with torch.no_grad():
                    # 教师对当前时序预测（包含新数据）
                current_teacher = self.base_model(batch_x)
                
                # 真实值仅取第一个时间点 (y_t)
                real_first = batch_y[:, :1, :, :]
                # 教师预测后续时间点 (y_t+1 到 y_t+H)
                teacher_rest = current_teacher[:, :self.output_len-1, :, :]
                
                # 拼接伪标签
                pseudo_label = torch.cat([real_first, teacher_rest], dim=1)
                
                # 计算加权TD损失
                weights = torch.tensor(
                    [self.gamma ** i for i in range(self.output_len)],
                    device=self.device
                ).view(1, -1, 1, 1)
                
                # 计算上一预测与伪标签差异
                td_loss = (prev_pred - pseudo_label).pow(2) * weights
                td_loss = td_loss.mean()
                
                # 仅更新学生模型
                self.td_optimizer.zero_grad()
                td_loss.backward()
                self.td_optimizer.step()
            
            # === 4. 更新经验回放缓冲区 ===
            # 存入当前预测（等待未来真实值）
            self.buffer.append((
                batch_x.detach().clone(),
                teacher_pred.detach().clone()  # 存储基础教师预测
            ))
            
            # === 5. 添加新数据到缓冲区（当获得完整真实值）===
            # 注意：实际应用中需等待H步后获取完整真实值
            if batch_idx >= self.output_len:
                # 获取完整真实值序列（H步前预测）
                past_index = batch_idx - self.output_len
                (x_past, y_past) = test_loader.dataset[past_index]
                self.buffer.append((
                    x_past.unsqueeze(0).to(self.device),
                    y_past.unsqueeze(0).to(self.device)
                ))
            batch_idx=batch_idx+1
        
        # 返回逆归一化的结果
        preds = scaler.inverse_transform(np.concatenate(preds, axis=0))
        truths = scaler.inverse_transform(np.concatenate(truths, axis=0))
        mae= np.mean(np.abs(preds-truths))
        rmse= np.sqrt(np.mean((preds-truths)**2))
        print('mae:' ,mae)
        print('rmse:' ,rmse)
        return preds, truths
    
if __name__=='__main__':
    model,train_loader,val_loader,test_loader,scaler,config,adj_mx=init_model(model_name='GWNET',model_path='saved_models\GWNET_chitaxi_final_model.pth',config_path=r'models\chitaxi_config.yaml')
    from types import SimpleNamespace
    config = SimpleNamespace(**config)
    online_model= DSOF( model,6,1,len(adj_mx),'cuda:0',batch_size=32)
    online_model.online(test_loader,scaler)
    

            