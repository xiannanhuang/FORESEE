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
import argparse
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
from opencity.opencity import OpenCity
model_dict={'STGCN':STGCN,'AGCRN':AGCRN,'DCRNN':DCRNN,'GWNET':GWNET,'MTGNN':MTGNN}

def init_model(model_name='STGCN',model_path='saved_models\STGCN_nycbike_train_months_12final_model.pth',config_path=r'models\nycbike_config.yaml',llm=False):
    
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
    if llm==False:
        model=causal_model.CausalModel(model_dict[model_name].Model,config,adj_mx).to(device)
        model.load_state_dict(torch.load(model_path))
    elif llm=='opencity':
        adj_mx=np.repeat(adj_mx, 2, axis=0)  # 复制每一行
        adj_mx=np.repeat(adj_mx, 2, axis=1)  # 复制每一行
        adj_mx=adj_mx+np.eye(len(adj_mx))
    
    
        model=OpenCity(argparse.Namespace(**config),[config['dataset_name']],adj_mx,device,1)
        model_weights = {k.replace('module.', '').replace('predictor.', ''): v for k, v in torch.load(r'finetuned_opencity_chibike.pth').items()}
        model.load_state_dict(model_weights)
        # model.load_state_dict(torch.load(r'opencity\OpenCity-base.pth'))
    return model,train_loader,val_loader,test_loader,scaler,config

import time
import numpy as np
import torch
import os

from ray import tune

import scipy.sparse as sp

from tqdm import tqdm


class moving_avg(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.avg = torch.nn.AvgPool2d(kernel_size=(1, kernel_size), stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ADCSDModule(torch.nn.Module):
    def __init__(self, output_dim, output_window, num_nodes, moving_avg=5):
        super(ADCSDModule, self).__init__()
        self.decomp = series_decomp(moving_avg)
        hidden_ratio = 128
        FWL_list_1 = [torch.nn.Linear(output_dim, hidden_ratio), 
                      torch.nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
                      torch.nn.GELU(), 
                      torch.nn.Linear(hidden_ratio, output_dim)]
        self.FWL_1 = torch.nn.Sequential(*FWL_list_1)
        self.learned_lambda_1 = torch.nn.Parameter(torch.zeros(num_nodes, 1))
        FWL_list_2 = [torch.nn.Linear(output_dim, hidden_ratio), 
                      torch.nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
                      torch.nn.GELU(), 
                      torch.nn.Linear(hidden_ratio, output_dim)]
        self.FWL_2 = torch.nn.Sequential(*FWL_list_2)
        self.learned_lambda_2 = torch.nn.Parameter(torch.zeros(num_nodes, 1))

    def forward(self, x):
        output_1, output_2 = self.decomp(x)
        output = x + self.learned_lambda_1 * self.FWL_1(output_1) + self.learned_lambda_2 * self.FWL_2(output_2)
        return output[:,-1:,:,:]




   
    


def ADCSD(model, test_dataloader,config,scaler,loss_func,llm=False):
    '''
    y = F(x) + lambda_1 * g_1(F(x)_1) + lambda_2 * g_2(F(x)_2), finetuning g_1, g_2, and lambda
    '''
    
    model.eval()
    model.to(config.device)


    FWL = ADCSDModule(output_dim=2,
                        output_window=12,
                        num_nodes=config.num_nodes,
                        moving_avg=5).to(config.device)
    optimizer = torch.optim.Adam(FWL.parameters(), lr=0.01, eps=1.0e-8, weight_decay=0, amsgrad=False)

    data_number = 0
    y_truths = []
    y_preds = []
   
    q = []
    for batch,y in tqdm(test_dataloader,total=len(test_dataloader)):
        batch=batch.to(config.device)
        y=y.to(config.device)
        data_number += 1

        with torch.no_grad():
            if llm==False:
                output = model(batch)
            if llm=='opencity':
                output = model(batch,batch)
                output=output.reshape(1,1,-1,2)
                y=y[:,0,0,:,0].reshape(1,1,-1,2)
        FWL.eval()
        # output = FWL(output)

       

        # for Training
        q.append(output.cpu().detach().numpy()) #(1, out_window, num_nodes, 2)
        if len(q)>13:
            FWL.train()
            batch = np.concatenate(q[-12:], axis=0) #(12, out_window, num_nodes, 2)
            output= torch.tensor(batch).transpose(0,1).to(config.device) #(1, 12, num_nodes, 2)
            
            # with torch.no_grad():
            #     output =model(batch)
            output = FWL(output)
            
            y_true = scaler.inverse_transform(y[..., :])
            y_pred = scaler.inverse_transform(output[..., :])

            loss = loss_func(y_true, y_pred)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
 

            y_truths.append(y_true.cpu().detach().numpy())
            y_preds.append(y_pred.cpu().detach().numpy())
        else:
            y_true = scaler.inverse_transform(y[..., :])
            y_pred = scaler.inverse_transform(output[..., :])
            y_truths.append(y_true[..., :].cpu().detach().numpy())
            y_preds.append(y_pred[..., :].cpu().detach().numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_truths = np.concatenate(y_truths, axis=0)
    outputs = {'prediction': y_preds, 'truth': y_truths}
    print('mae:' , abs(y_truths.reshape(-1)- y_preds.reshape(-1)).mean())
    print('rmse:', np.sqrt(np.mean((y_truths.reshape(-1)-y_preds.reshape(-1))**2)))
    
    
    return  outputs
model,train_loader,val_loader,test_loader,scaler,config=init_model('GWNET',model_path='saved_models\GWNET_nyctaxi_final_model.pth',config_path=r'models\nyctaxi_config.yaml')
from types import SimpleNamespace
config = SimpleNamespace(**config)
outputs = ADCSD( model,test_loader,config,scaler,torch.nn.MSELoss())