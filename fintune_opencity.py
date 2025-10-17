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
from tqdm import tqdm
from opencity.opencity import OpenCity
import argparse
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
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
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
        model_weights = {k.replace('module.', '').replace('predictor.', ''): v for k, v in torch.load(r'opencity\OpenCity-base.pth').items()}
        model.load_state_dict(model_weights)
        # model.load_state_dict(torch.load(r'opencity\OpenCity-base.pth'))
    return model,train_loader,val_loader,test_loader,scaler,config
model,train_loader,val_loader,test_loader,scaler,config=init_model(model_path='saved_models\STGCN_nycbike_train_months_12final_model.pth',config_path=r'opencity\chibike.yaml',llm='opencity')
from types import SimpleNamespace
model.cuda()
# 1. 冻结除最后一层外的所有参数
# for name, param in model.named_parameters():
    # if not name.startswith('linear'):  # 根据实际模型结构调整这里
    #     param.requires_grad = False
    # else:
    #     print(f"Parameter {name} requires grad")

# 2. 确认需要优化的参数
# trainable_params = [param for param in model.parameters() if param.requires_grad]
# print(f"Number of trainable parameters: {len(trainable_params)}")

# 3. 配置优化器（仅优化最后一层）
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3,               # 建议使用较小的学习率
      # 可选的正则化
)

# 4. 损失函数（根据任务选择）
criterion = torch.nn.MSELoss()  # 回归任务常用MSE
# 分类任务可用: criterion = torch.nn.CrossEntropyLoss()

# 5. 微调训练循环
num_epochs = 15  # 调整微调周期数

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}'):
        data, target = data.cuda(), target.cuda()
        target=target[:,0,0,:,0].reshape(1,1,-1,2)
        
        optimizer.zero_grad()
        output = model(data,data).reshape(1,1,-1,2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # if batch_idx % 50 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
    
    epoch_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.6f}')
    
    # 可选：添加验证步骤
    model.eval()
    val_loss = 0.0
    if 1:  # 每5个epoch进行一次验证
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                target=target[:,0,0,:,0].reshape(1,1,-1,2)
                output = model(data,data).reshape(1,1,-1,2)
                val_loss += criterion(output, target).item()
    
            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.6f}\n')

# 6. 保存微调后的模型
# torch.save(model.state_dict(), 'finetuned_opencity_chibike.pth')