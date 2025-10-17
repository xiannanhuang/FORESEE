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
from sklearn.linear_model import Ridge
class ADATS:


    def __init__(self, output_dim, output_window, num_nodes,lam,lr=0.2):
        self.output_dim = output_dim
        self.output_window = output_window
        self.num_nodes = num_nodes
        self.lam = lam
        self.models=[Ridge(alpha=lam) for _ in range(self.num_nodes*2)]
        self.weights = np.ones((self.num_nodes*2))
        self.lr = lr
    def fit(self, X, y):
        # X(his_len, num_nodes,2),y (batch,pre_len:1, num_nodes,2)
        his_len=X.shape[0]
        train_x,train_y=[],[]
        for i in range(his_len-self.output_window):
            train_x.append(X[i:i+self.output_window,:,:])
            train_y.append(X[i+self.output_window:i+self.output_window+1,:,:])
        
        train_x=np.array(train_x).reshape(-1,self.output_window,self.num_nodes*2)
        train_y=np.array(train_y).reshape(-1,1,self.num_nodes*2)
        b=train_x.shape[0]
        for i in range(self.num_nodes*2):
            self.models[i].fit(train_x[:,:,i].reshape(b,-1), train_y[:,:,i].reshape(b,-1))
    def predict(self, X):
        # X(batch,his_len, num_nodes*2)
        preds = []
        for i in range(self.num_nodes*2):
            preds.append(self.models[i].predict(X[:,:,i]))
        preds = np.array(preds).reshape((self.num_nodes, 2))
        return preds
    def update_weight(self,loss_adap,loss_base):
        weight=np.ones_like(self.weights)
        for i in range(self.num_nodes*2):
            weight[i] = np.exp(-loss_adap[i]) / (np.exp(-loss_adap[i])+  np.exp(-loss_base[i]))
        self.weights = self.lr*weight + (1 - self.lr) * self.weights
        

        


def AdaTS(model, test_dataloader,config,scaler,loss_func,llm='mo'):
    '''
    y = F(x) + lambda_1 * g_1(F(x)_1) + lambda_2 * g_2(F(x)_2), finetuning g_1, g_2, and lambda
    '''
    
    model.eval()


    FWL = ADATS(output_dim=2,
                        output_window=12,
                        num_nodes=config.num_nodes,
                        lam=0.5
                        )
   
    data_number = 0
    y_truths = []
    y_preds = []

   
    q = []
    idx=0
    for batch,y in tqdm(test_dataloader,total=len(test_dataloader)):
        batch=batch.to(config.device)
        y=y.to(config.device)
        data_number += 1

        with torch.no_grad():
            if llm=='opencity':
                output = model(batch,batch)
                output=output.reshape(y.shape[0],1,-1,2)
                y=y[:,0,0,:,0].reshape(y.shape[0],1,-1,2)
                q=batch[0,0,:,:,0].reshape(288,1,-1,2)
            else:
                output = model(batch)
                q.append(batch[:,-1:].cpu().detach().numpy()) #(1, out_window, num_nodes, 2)

        loss_base=(output-y).abs().detach().cpu().numpy().reshape(-1)


       
        y_true = scaler.inverse_transform(y[..., :])
        y_truths.append(y_true.cpu().detach().numpy())
        # for Training
        
        if (len(q)>480000 and len(q)%24==1) or (len(q)==2881 and idx%24==0):
            if isinstance(q,list):
                batch = np.concatenate(q[-48:], axis=0) #(48, out_window, num_nodes, 2)
                X= torch.tensor(batch).transpose(0,1)
            # with torch.no_grad():
            #     output =model(batch)
                FWL.fit(X[0].numpy(),np.concatenate(q[-49:-1], axis=0).transpose(1,0,2,3)[0])
                output2= FWL.predict(np.concatenate(q[-12:], axis=0).transpose(1,0,2,3).reshape(1,12,-1))
            elif isinstance(q,torch.Tensor):
                batch = q[-48:].cpu().detach().numpy() #(48, out_window, num_nodes, 2)
                X= torch.tensor(batch).transpose(0,1)
                # with torch.no_grad():
                #     output =model(batch)
                FWL.fit(X[0].numpy(),q[-49:-1].cpu().detach().numpy().transpose(1,0,2,3)[0])
                output2= FWL.predict(q[-12:].cpu().detach().numpy().transpose(1,0,2,3).reshape(1,12,-1))

            
            
            output=output2.reshape(config.num_nodes*2)*FWL.weights+output.reshape(config.num_nodes*2).detach().cpu().numpy()*(1-FWL.weights)
            output=output.reshape(1,1,config.num_nodes,2)
            y_pred = scaler.inverse_transform(output[..., :])
         
           
            y_preds.append(y_pred)
            loss_adap=abs(output2-y.detach().cpu().numpy()).reshape(-1)
            FWL.update_weight(loss_adap,loss_base)
        elif len(q)>4800000 and len(q)%24!=1:
            if isinstance(q,list):
                output2= FWL.predict(np.concatenate(q[-12:], axis=0).transpose(1,0,2,3).reshape(1,12,-1))
            elif isinstance(q,torch.Tensor):
                output2= FWL.predict((q[-12:].cpu().detach().numpy().reshape(1,12,-1)))
            # output2= FWL.predict(np.concatenate(q[-12:], axis=0).transpose(1,0,2,3).reshape(1,12,-1))
            output=output2.reshape(config.num_nodes*2)*FWL.weights+output.reshape(config.num_nodes*2).detach().cpu().numpy()*(1-FWL.weights)
            output=output.reshape(1,1,config.num_nodes,2)
           
            y_pred = scaler.inverse_transform(output[..., :])
          
            y_preds.append(y_pred)
            loss_adap=abs(output2-y.detach().cpu().numpy()).reshape(-1)
            FWL.update_weight(loss_adap,loss_base)
        else:
          
            y_pred = scaler.inverse_transform(output[..., :])
           
            y_preds.append(y_pred.cpu().detach().numpy())
        idx+=1
        

    y_preds = np.concatenate(y_preds, axis=0)
    y_truths = np.concatenate(y_truths, axis=0)
    outputs = {'prediction': y_preds, 'truth': y_truths}
    print('mae:' , abs(y_truths.reshape(-1)- y_preds.reshape(-1)).mean())
    print('rmse:', np.sqrt(np.mean((y_truths.reshape(-1)-y_preds.reshape(-1))**2)))
    
    
    return  outputs
model,train_loader,val_loader,test_loader,scaler,config=init_model('GWNET',model_path=r'saved_models\GWNET_nyctaxi_final_model.pth',config_path=r'models\nyctaxi_config.yaml')
from types import SimpleNamespace
device='cuda:0'
config['device']=device
model.to(device)
config = SimpleNamespace(**config)
outputs = AdaTS( model,test_loader,config,scaler,torch.nn.MSELoss())