
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: (batch, channel, nodes, timesteps)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class nconv2(nn.Module):
    def __init__(self):
        super(nconv2, self).__init__()

    def forward(self, x, A):
        # x: (batch, channel, nodes, timesteps)
        x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # 使用1x1的卷积核替代Linear层
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        # x: (batch, channel, nodes, timesteps)
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout,multi_adj=False, support_len=3, order=2):
        super(gcn, self).__init__()
        if multi_adj:
            self.nconv = nconv2()
        else:
            self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        :param x: (batch, channel, nodes, timesteps)
        :param support: list of adjacent matrix
        """
        out = [x]
        # Multi-Graph
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            # MixHop: n-order
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # Putting it together in the channel dimension
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Model(nn.Module):
    def __init__(self,
                 config, 
                 adj_mx,
                 dropout=0.3,
                 supports=None,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 in_dim=2,
                 out_dim=16,
                 residual_channels=16,
                 dilation_channels=16,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                #  residual_channels=32,
                #  dilation_channels=32,
                #  skip_channels=1024,
                #  end_channels=1024,
                #  kernel_size=2,
                #  blocks=4,
                #  layers=3,
                 out_window=1,
                 input_window=6):
        # skip_channels = dilation_channels * (blocks * layers)
      
        if config.get('model_confidence', False):
            out_dim=out_dim+2
        super(Model, self).__init__()
        self.dropout = dropout
        self.gcn_bool = gcn_bool
        self.out_window=1
        self.addaptadj = addaptadj
        self.blocks = blocks
        self.layers = layers
        device = config.get('device', torch.device('cpu'))
        self.device=device
        self.adj_mx=adj_mx
        out_dim=config.get('d_model', 2)
        # self.num_nodes=num_nodes
        supports=[torch.tensor(adj_mx,dtype=torch.float32).to(device)]
        # if config['multi_adj']:
        #     supports=[torch.tensor(adj_mx,dtype=torch.float32).to(device).unsqueeze(0).repeat(128,1,1)]
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        # 1.list of adjacency matrix
        self.supports = supports
        self.supports_len = 0
        self.z_dim=config.get('d_model', 2)
        if supports is not None:
            self.supports_len += len(supports)
        else:
            self.supports = []
        num_nodes=adj_mx.shape[0]
        
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                # ===================================================================
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                # ===================================================================
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # 2.Stacked Gated Temporal Convolutional Layers
        receptive_field = 1
        for b in range(blocks):
            # Here the convolution kernel is fixed.
            additional_scope = kernel_size - 1   # 1
            new_dilation = 1
            # Each layer requires padding = 3 data fills keeping the original length unchanged.
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # padding = (kernel_size - 1) * dilation
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                # Graph Convolution Network
                if self.gcn_bool:
                    self.gconv.append(gcn(c_in=dilation_channels,
                                          c_out=residual_channels,
                                          multi_adj=0,
                                          dropout=dropout,
                                          support_len=self.supports_len))

        # 3.Output prediction layer
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        )  # channels from 256 to 512

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True
        )  # channels from 512 to 12

        self.receptive_field = receptive_field  # 1 + (4 * 3) = 
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, input,adj_emb=None):
        """
        Here one-dimensional convolutional kernels are used to extract temporal information,
        and the size of the convolutional kernels is constant at 2.
        :param input: (batch, in_channel, nodes, timesteps)
        :return:
        """
         # (batch_size, input_window, num_nodes, feature_dim)
        b, t, n, d = input.size()
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)   # timesteps
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the adaptive adjacent matrix
        new_supports = None
        if adj_emb is not None:
            self.supports=[torch.tensor(self.adj_mx,dtype=torch.float32).to(self.device).unsqueeze(0).repeat(adj_emb.shape[0],1,1)]
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            if adj_emb is None:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]
            else:
                adp=F.softmax(F.relu(torch.bmm(adj_emb,adj_emb.transpose(1,2))),dim=2)
                new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|   *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # the length of timesteps decreases after each temporal convolution.
            # skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # residual connnection
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # skip: (batch, channel, nodes, 1)  It can be understood as aggregating temporal features.
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).transpose(1,3)
        return x[:,-1:,:,:].reshape(b,1,n,self.z_dim,-1).transpose(1,-1).reshape(b,-1,n,self.z_dim)
    
    def get_fe(self,input,adj_emb=None):
        """
        Here one-dimensional convolutional kernels are used to extract temporal information,
        and the size of the convolutional kernels is constant at 2.
        :param input: (batch, in_channel, nodes, timesteps)
        :return:
        """
         # (batch_size, input_window, num_nodes, feature_dim)
        b, t, n, d = input.size()
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)   # timesteps
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the adaptive adjacent matrix
        new_supports = None
        if adj_emb is not None:
            self.supports=[torch.tensor(self.adj_mx,dtype=torch.float32).to(self.device).unsqueeze(0).repeat(adj_emb.shape[0],1,1)]
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            if adj_emb is None:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]
            else:
                adp=F.softmax(F.relu(torch.bmm(adj_emb,adj_emb.transpose(1,2))),dim=2)
                new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|   *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # the length of timesteps decreases after each temporal convolution.
            # skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # residual connnection
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        return F.relu(skip)

    def get_pred_and_feature(self, x):
         # (batch_size, input_window, num_nodes, feature_dim)
        b=x.shape[0]
        input = x.permute(0, 3, 2, 1)

        in_len = input.size(3)   # timesteps
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the adaptive adjacent matrix
        new_supports = None
        
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]
            

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|   *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # the length of timesteps decreases after each temporal convolution.
            # skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # residual connnection
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            if i==0:
                feature = x.clone()

        # skip: (batch, channel, nodes, 1)  It can be understood as aggregating temporal features.
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).transpose(1,3)
        return x[:,-self.out_window:,:,:],feature.reshape(b,-1)