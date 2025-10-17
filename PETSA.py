# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.optimizer import get_optimizer
# from models.forecast import forecast
# from datasets.loader import get_test_dataloader
# from utils.misc import prepare_inputs
# from config import get_norm_method
import math
from argparse import Namespace

def get_optimizer(optim_params, lr):
#     if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
#         return torch.optim.SGD(
#             optim_params,
#             lr=cfg.SOLVER.BASE_LR,
#             momentum=cfg.SOLVER.MOMENTUM,
#             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             dampening=cfg.SOLVER.DAMPENING,
#             nesterov=cfg.SOLVER.NESTEROV,
#         )
#     elif cfg.SOLVER.OPTIMIZING_METHOD == 'adam':
    return torch.optim.Adam(
            optim_params,
            lr=lr,
            # weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    # elif cfg.SOLVER.OPTIMIZING_METHOD == 'Radam':
    #     return torch.optim.RAdam(
    #         optim_params,
    #         lr=cfg.SOLVER.BASE_LR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY
    #     )
    # elif cfg.SOLVER.OPTIMIZING_METHOD == 'adamw':
    #     return torch.optim.AdamW(
    #         optim_params,
    #         lr=cfg.SOLVER.BASE_LR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY
    #     )
    # else:
    #     raise NotImplementedError(
    #         "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
    #     )

class CorrCoefLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        x = preds.reshape(-1)
        y = target.reshape(-1)
        
        data = torch.stack([x, y], dim=0)
        corrmat = torch.corrcoef(data)
        corr_xy = corrmat[0, 1]
        
        return  -corr_xy


class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        cfg=Namespace(**cfg)
        self.cfg = cfg
        # self.model_cfg = cfg.MODEL
        self.model = model
        # self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.device=self.model.device
        # self.test_loader = get_test_dataloader(cfg)
        # self.test_data = self.test_loader.dataset.test

        # if self.cfg.TTA.PETSA.CALI_MODULE:
        self.cali = Calibration(cfg).cuda()
        
        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), 0.0005)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        # cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        # self.test_loader = get_test_dataloader(cfg)
        # self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

        self.person_cor = CorrCoefLoss()   
        self.count_parameters()             
    
    def count_parameters(self):
        print("------- PARAMETERS -------")
        total_sum = 0
        for name, param in self.cali.named_parameters():
            print (param.requires_grad, name, param.size(), param.numel())
            if param.requires_grad == True:
                total_sum = total_sum + int(param.numel())
        print("Total: ", total_sum)

    def forward(self, x):
        # if self.cfg.TTA.PETSA.CALI_MODULE:
        #     inputs = self.cali.input_calibration(inputs)
        # pred_after_adapt, ground_truth = self(self.cfg, inputs, self.model, self.norm_module)
        # if self.cfg.TTA.PETSA.CALI_MODULE:
        #     pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        # for i in range(batch_size-1):
        #     pred[i, period-i:] = pred_after_adapt[i, period-i:]
        inputs=self.cali.input_calibration(x)
        pred_after_adapt=self.model(inputs)
        pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        
        return pred_after_adapt
    
    def reset(self):
        self._load_model_and_optimizer()
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        # if self.cfg.TTA.PETSA.CALI_MODULE:
        models.append(self.cali)
        return models

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)
    
    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules
    
    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        # if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
        return named_modules
        
        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact_match = '(exact)' in module_name
            module_name = module_name.replace('(exact)', '')
            if exact_match:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if name == module_name]
            else:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if module_name in name]

        assert len(named_modules_to_adapt) > 0
        return named_modules_to_adapt
    
    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)
    
    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt
    
    def switch_model_to_train(self):
        for model in self._get_all_models():
            model.train()
    
    def switch_model_to_eval(self):
        for model in self._get_all_models():
            model.eval()
    
    @torch.enable_grad()
    def online(self,test_loader:torch.utils.data.DataLoader,scaler):
        
        xs=[]
        ys=[]
        preds=[]
        self.switch_model_to_eval()
        for idx,(inputs,targets) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs=self(inputs)
                xs.append(inputs.detach().cpu().numpy())
                ys.append(targets.detach().cpu().numpy())
                preds.append(outputs.detach().cpu().numpy())
                if np.concatenate(preds).shape[0] >= self.cfg.pred_len:
                    inputs = xs[-self.cfg.pred_len]
                    targets = ys[-self.cfg.pred_len]
                    inputs=torch.tensor(inputs).to(self.device)
                    targets=torch.tensor(targets).to(self.device)
                    pred=self(inputs)
                    loss=F.l1_loss(pred, targets, reduction='mean')
                    loss=loss+F.huber_loss(pred, targets, reduction='mean')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        preds=np.concatenate(preds)
        targets=np.concatenate(ys)
        preds=scaler.inverse_transform(preds)
        targets=scaler.inverse_transform(targets)
        rmse=np.sqrt(((preds-targets)**2).mean())
        mae=abs(preds-targets).mean()
                    
    
                
              
        
        print('After TSF-TTA of PETSA')
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}')
        print()
        
        self.model.eval()
        return rmse, mae
    
   






class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        b,t,n,d=x.shape
        x = x.reshape(b,t,n*d)
        
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        if self.var_wise:
            x_1 = torch.tanh(self.gating * x)
            new_x =  (torch.einsum('biv,iov->bov', x_1,  weight) + self.bias)
        else:
            x_1 = torch.tanh(self.gating * x)
            new_x =  (torch.einsum('biv,io->bov', x_1,  weight) + self.bias)


        x = x + new_x
        x=x.reshape(b,t,n,d)

        return x




class Calibration(nn.Module):
    def __init__(self, cfg):
        super(Calibration, self).__init__()
     
        self.cfg = cfg
        self.seq_len = cfg.input_window
        self.pred_len = cfg.pred_len
        self.n_var = cfg.num_nodes*2
        self.hidden_dim = 128
        self.gating_init = 0.01
        self.var_wise = 1
        self.low_rank =16

        # if cfg.MODEL.NAME == 'PatchTST':
        #     self.in_cali = GCM(self.seq_len, 1, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)
        #     self.out_cali = GCM(self.pred_len, 1, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)
        # else:
        self.in_cali = GCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)
        self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)
        
    def input_calibration(self, inputs):
        # enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(inputs)
        return enc_window

    def output_calibration(self, outputs):
        return self.out_cali(outputs)
import pandas as pd
if __name__ == "__main__":
    res=[]
    for model in ['GWNET']:
        for dataset in ['nyctaxi']:
          
   
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
            with open(fr'models\{dataset}_config.yaml', 'r') as config_file:
                    config = yaml.safe_load(config_file)
            config['model'],config['train_months']=model,train_month
            
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
            base_stgcn.load_state_dict(torch.load(f'saved_models\{model}_{dataset}_final_model_big.pth'))
            adpter=Adapter(config,base_stgcn)
            mae,mse=adpter.online(test_loader,scaler)
            res.append([model,dataset,mae,mse])
    res=pd.DataFrame(res,columns=['model','dataset','mae','mse'])
    res.to_csv('res2.csv',index=False)