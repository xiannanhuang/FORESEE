import numpy as np
import torch
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error
model='STGCN'
train_month=12
import yaml
# from sklearn_extra.cluster import KMedoids
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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import tqdm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import time

def normalized_adjacency_matrix(A):
    """计算归一化的邻接矩阵（随机游走归一化）"""
    degrees = np.sum(A, axis=1)  # 计算每个节点的度
    # 处理孤立节点（度为0），避免除以0
    isolated_nodes = degrees == 0
    degrees[isolated_nodes] = 1  # 设置为1（不影响结果）
    
    # 计算度矩阵的逆
    D_inv = np.diag(1.0 / degrees)
    A_norm = D_inv @ A  # 归一化邻接矩阵
    
    # 确保孤立节点不会影响结果
    A_norm[isolated_nodes, :] = 0  # 孤立节点无法传播信息
    
    return A_norm



def graph_smooth_torch(X,A,alpha):
    """
    x: torch.tensor (n,d)
    A: torch.tensor (n,n)
    alpha: tensor
    """

    A=torch.tensor(A).to(X.device).float()
    degrees = torch.sum(A, dim=1)  # 计算每个节点的度
    # 处理孤立节点（度为0），避免除以0
    isolated_nodes = degrees == 0
    degrees[isolated_nodes] = 1  # 设置为1（不影响结果）
    
    # 计算度矩阵的逆
    D_inv = torch.diag(1.0 / degrees)
    A_norm = D_inv @ A  # 归一化邻接矩阵
    
    # 确保孤立节点不会影响结果
    A_norm[isolated_nodes, :] = 0  # 孤立节点无法传播信息
    X_smooth = X.clone()
    # 迭代平滑过程
    for i in range(3):
      
        # 计算邻居信息扩散
        neighbor_agg = A_norm @ X_smooth
        
        # 应用带参数的平滑
        X_smooth = (1 - alpha) * X_smooth + alpha * neighbor_agg
        
        
        
    return X_smooth


    
from opencity.opencity import OpenCity

import numpy as np

import argparse





model_dict={'STGCN':STGCN,'AGCRN':AGCRN,'DCRNN':DCRNN,'GWNET':GWNET,'MTGNN':MTGNN}

def init_model(model_name='STGCN',model_path='saved_models\STGCN_nyctaxi_train_months_12final_model.pth',config_path=r'models\nyctaxi_config.yaml',llm=False,device=None):
    
    with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    if device is None:
        device = config['device']

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets( config)
    # logging.info("Configuration:")
    # for key, value in config.items():
    #     logging.info(f"{key}: {value}")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset,24, shuffle=False)
    config['adj_mx_file']=os.path.join('dataset\\',config['dataset_name'], 'adj_mx.npy')
    adj_mx =  np.load(config['adj_mx_file'])[valid_gird][:,valid_gird]
    config['num_nodes']=len(valid_gird)
    re_adj=adj_mx.copy()
    if llm==False:
        model=causal_model.CausalModel(model_dict[model_name].Model,config,adj_mx).to(device)
        model.load_state_dict(torch.load(model_path))
    elif llm=='opencity':
        adj_mx=adj_mx+np.eye(len(adj_mx))
        adj_mx=np.repeat(adj_mx, 2, axis=0)  # 复制每一行
        adj_mx=np.repeat(adj_mx, 2, axis=1)  # 复制每一行
        
    
    
        model=OpenCity(argparse.Namespace(**config),[config['dataset_name']],adj_mx,device,1)
        model_weights = {k.replace('module.', '').replace('predictor.', ''): v for k, v in torch.load(r'finetuned_opencity_chibike.pth').items()}
        model.load_state_dict(model_weights)
        # model.load_state_dict(torch.load(r'opencity\OpenCity-base.pth'))
    return model,train_loader,val_loader,test_loader,scaler,config,re_adj

import torch.nn.functional as F
from sklearn.cluster import KMeans
def smooth_tensor2(tensor, kernel):
    """
    对输入张量的第一个维度进行移动平均平滑
    Args:
        tensor: 输入张量，形状为 (D0, D1, D2, D3)
        kernel: 平滑核，形状为 (window_size,)
    Returns:
        平滑后的张量，形状与输入相同
    """
    # 确保窗口大小为奇数
    window_size = kernel.shape[0]
    assert window_size % 2 == 1, "Window size must be odd"
    
    # 保存原始形状
    orig_shape = tensor.shape
    
    # 将张量展平为 (时间步, 特征) 形状
    flattened = tensor.reshape(orig_shape[0], -1)  # (24, 1 * 234 * 2) = (24, 468)
    
    # 获取特征数量 (468)
    num_features = flattened.shape[1]
    
    # 计算需要添加的边界填充量
    padding = window_size // 2
    
    # 方法：使用1D分组卷积实现移动平均
    # 1. 调整张量形状为卷积所需的格式: (batch=1, channels=特征数, length=时间步)
    input_for_conv = flattened.t().unsqueeze(0)  # (1, 468, 24)
    
    # 2. 创建卷积核: 每个特征使用相同的平均卷积核
    # 卷积核形状: (输出通道数, 输入通道数/组, 卷积核长度)
    # 因为我们要对每个特征独立处理，使用 groups=num_features
    # kernel = torch.tensor([0.2,0.6,0.2]).cuda()
    kernel = kernel.expand(num_features, 1, window_size).float()  # (468, 1, 3)
    
    # 3. 应用复制填充
    padded = F.pad(input_for_conv, (padding, padding), mode='replicate').float()  # (1, 468, 26)
    
    # 4. 应用分组卷积进行平滑
    smoothed = F.conv1d(
        input=padded,
        weight=kernel,
        bias=None,
        stride=1,
        padding=0,  # 因为我们已经手动填充了
        groups=num_features  # 关键：每个特征独立处理
    )  # 形状 (1, 468, 24)
    
    # 5. 恢复形状: (24, 468)
    smoothed = smoothed.squeeze(0).t()
    
    # 6. 恢复原始形状 (24, 1, 234, 2)
    return smoothed.view(orig_shape)


from torch import nn


class Online_day():
    def __init__(self,model:torch.nn.Module,ema=[0.7,0.8,0.9,1],test_sample=None,adj_mx=None,llm='dsd',eta=10):
        super(Online_day,self).__init__()
        self.model=model
        self.ema=ema
        # self.ema=[0.7,0.8,0.9,1]
        # self.eta=10
        # self.ema=[ema]
        self.eta=eta
        if llm=='opencity':
            self.delta=torch.zeros_like(model(test_sample,test_sample).reshape(test_sample.shape[0],1,-1,2)).unsqueeze(0).repeat(len(self.ema),1,1,1,1)
        else:
            self.delta=torch.zeros_like(model(test_sample)).unsqueeze(0).repeat(len(self.ema),1,1,1,1)
        self.weights=(torch.ones(len(self.ema))/len(self.ema)).to(self.model.device)
        self.adj_mx=adj_mx
        self.num_nodes=adj_mx.shape[0]
        self.kernel=nn.Parameter(torch.tensor([0.2,0.6,0.2]).to(self.model.device))
        self.alpha=nn.Parameter(torch.tensor(0.01).to(self.model.device))
        

        

    def online(self,test_loader,scaler,llm='none',smooth=True):
        self.model.eval()
        y_pred=[]
        y_truth=[]
       
        opt=torch.optim.SGD([self.kernel,self.alpha],lr=0.01)
        err_past=None
        
        
        
        j=0
        for (xi, yi) in tqdm.tqdm(test_loader):
            j=j+1*test_loader.batch_size
            y=[]
            y_hat=[]
            with torch.no_grad():
                
                xi=xi.to(self.model.device)
                yi=yi.to(self.model.device)
                
                if llm=='opencity':
                    y_hati=self.model(xi,xi)
                    y_hati=y_hati.reshape(yi.shape[0],1,-1,2)
                    yi=yi[:,0,0,:,0].reshape(yi.shape[0],1,-1,2)
                else:
                    y_hati=self.model(xi)
                y_hat.append(y_hati)
                y.append(yi)
            if j%24==0:
                pred=[]
                loss=[]
                y=torch.cat(y)
                y_hat=torch.cat(y_hat)
               
          
            
            
                now_err=(y-y_hat).reshape(-1,self.num_nodes,2).transpose(1,0).reshape(self.num_nodes,-1) 
                if smooth:
                    now_err=graph_smooth_torch(now_err,self.adj_mx,self.alpha).reshape(self.num_nodes,-1,2).transpose(1,0).reshape(-1,1,self.num_nodes,2)

                    now_err=smooth_tensor2(now_err,kernel=self.kernel)
                else:

                    now_err=(y-y_hat)
                    
                if len(y_pred)>1 and smooth:
                    loss_=((now_err-err_past[:now_err.shape[0]])**2).mean()
                    opt.zero_grad()
                    loss_.backward()
                    opt.step()
                err_past=(y-y_hat).clone()
                    # self.delta=smooth_tensor(self.delta,window_size=3)
                    # self.delta=smooth_tensor(self.delta,window_size=3)
                    # now_err=smooth_tensor(now_err,window_size=3)
                    
                for i in range(self.delta.shape[0]):
                    predi=y_hat+self.delta[i,:y.shape[0]]
                    lossi=abs(predi-y).mean() 
                    self.delta[i,:y.shape[0]]=((1-self.ema[i])*now_err[:y.shape[0]]+self.delta[i,:y.shape[0]]*self.ema[i])
                    pred.append(predi*self.weights[i])
                    loss.append(lossi)
                j=0
                # pred=torch.stack(pred).sum(0)/self.weights.sum()
                pred=torch.stack(pred).sum(0)
                loss=torch.exp(-self.eta*torch.stack(loss))*self.weights #(len(self.ema))
                self.weights=loss/loss.sum()
                # self.weights=loss
                
               
                
                y_pred.append(pred.cpu().detach().numpy())
                y_truth.append(y.cpu().detach().numpy())
       
        y_pred=np.concatenate(y_pred).reshape(-1)
        y_truth=np.concatenate(y_truth).reshape(-1)
        y_pred=scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
        y_truth=scaler.inverse_transform(y_truth.reshape(-1,1)).reshape(-1)
        mae=mean_absolute_error(y_pred,y_truth)
        rmse=np.sqrt(mean_squared_error(y_pred,y_truth))
        print('mae:',mae,'rmse:',rmse)
        return y_pred,y_truth,mae,rmse
import pandas as pd
if __name__=='__main__':
    res=[]
    for model_name in ['GWNET']:
        for dataset in ['nycbike','chibike','chitaxi','baybike','bosbike']:
            model,train_loader,val_loader,test_loader,scaler,config,adj_mx=init_model(model_name,model_path=f'saved_models\{model_name}_{dataset}_final_model.pth',config_path=fr'opencity\{dataset}.yaml',llm='opencity')
            # from types import SimpleNamespace
            import matplotlib.pyplot as plt
            model.to('cuda:0')
            # config['device']='cuda:0'
            # config = SimpleNamespace(**config)
            # 获取第一个批次的输入数据
            data_iter = iter(test_loader)
            model.eval()
            first_batch = next(data_iter)
            # 假设批次结构为 (inputs, labels)，取inputs部分移至GPU
            inputs = first_batch[0].to('cuda:0')

            # 初始化并调用online方法
            maes=[]
            ema=[.7,.8,.9,1]
        
            
            online_instance=Online_day(model,ema,inputs,adj_mx+np.eye(adj_mx.shape[0]),llm='opencity')
            outputs,_,mae,rmse = online_instance.online(test_loader,scaler,llm='opencity',smooth=True)
            maes.append(mae)
            res.append([dataset,model_name,mae,rmse])
            # print(f'{dataset},{model},{mae},{rmse}')
            # print('maes:',maes)
            # print('ema:',ema)
            # plt.plot(outputs)
    pd.DataFrame(res,columns=['dataset','model','mae','rmse']).to_csv('online.csv',index=False)
      
    
