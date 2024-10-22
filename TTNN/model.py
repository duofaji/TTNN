# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:12:49 2023

@author: fangwenji
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#%%data normalization
def normal(x):
    
    max_x = np.max(x, axis=2)[:,:,np.newaxis]
    min_x = np.min(x, axis=2)[:,:,np.newaxis]
    y = (x - min_x)/(max_x - min_x)
    
    return y

#%%network architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block):
        super().__init__()
        
        #model for estimate depth
        self.conv1_h = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.maxpooling1_h = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv2_x_h = self._make_layer(block, 16, 32, num_block[0], 2)
        self.maxpooling2_h = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv3_x_h = self._make_layer(block, 32, 64, num_block[1], 2)
        self.maxpooling3_h = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv4_x_h = self._make_layer(block, 64, 64, num_block[2], 2)
        self.maxpooling4_h = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv5_x_h = self._make_layer(block, 64, 64, num_block[3], 2)
        self.avg_pool_h = nn.AdaptiveAvgPool1d((1))
        
        self.fc_h = nn.Linear(64, 1)     
        
        #model for estimate epicenter distance
        self.conv1_d = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.maxpooling1_d = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv2_x_d = self._make_layer(block, 16, 32, num_block[0], 2)
        self.maxpooling2_d = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv3_x_d = self._make_layer(block, 32, 64, num_block[1], 2)
        self.maxpooling3_d = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv4_x_d = self._make_layer(block, 64, 64, num_block[2], 2)
        self.maxpooling4_d = nn.Sequential(nn.MaxPool1d(kernel_size=2))
        self.conv5_x_d = self._make_layer(block, 64, 64, num_block[3], 2)
        self.avg_pool_d = nn.AdaptiveAvgPool1d((1))
        
        self.fc_d = nn.Linear(64, 1) 
        
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        
        #model for estimate depth
        #for station1
        output1 = self.conv1_h(x1)
        output1 = self.maxpooling1_h(output1)
        output1 = self.conv2_x_h(output1)
        output1 = self.maxpooling2_h(output1)
        output1 = self.conv3_x_h(output1)
        output1 = self.maxpooling3_h(output1)
        output1 = self.conv4_x_h(output1)
        output1 = self.maxpooling4_h(output1)
        output1 = self.conv5_x_h(output1)
        output1 = self.avg_pool_h(output1)
        output1 = output1.view(output1.size(0), -1)
        #for station2
        output2 = self.conv1_h(x2)
        output2 = self.maxpooling1_h(output2)
        output2 = self.conv2_x_h(output2)
        output2 = self.maxpooling2_h(output2)
        output2 = self.conv3_x_h(output2)
        output2 = self.maxpooling3_h(output2)
        output2 = self.conv4_x_h(output2)
        output2 = self.maxpooling4_h(output2)
        output2 = self.conv5_x_h(output2)
        output2 = self.avg_pool_h(output2)
        output2 = output2.view(output2.size(0), -1)
        #for station3
        output3 = self.conv1_h(x3)
        output3 = self.maxpooling1_h(output3)
        output3 = self.conv2_x_h(output3)
        output3 = self.maxpooling2_h(output3)
        output3 = self.conv3_x_h(output3)
        output3 = self.maxpooling3_h(output3)
        output3 = self.conv4_x_h(output3)
        output3 = self.maxpooling4_h(output3)
        output3 = self.conv5_x_h(output3)
        output3 = self.avg_pool_h(output3)
        output3 = output3.view(output3.size(0), -1)
        #for station4
        output4 = self.conv1_h(x4)
        output4 = self.maxpooling1_h(output4)
        output4 = self.conv2_x_h(output4)
        output4 = self.maxpooling2_h(output4)
        output4 = self.conv3_x_h(output4)
        output4 = self.maxpooling3_h(output4)
        output4 = self.conv4_x_h(output4)
        output4 = self.maxpooling4_h(output4)
        output4 = self.conv5_x_h(output4)
        output4 = self.avg_pool_h(output4)
        output4 = output4.view(output4.size(0), -1)
        
        #fusion of information from four stations
        output = output1 + output2 + output3 + output4
        output = self.fc_h(output)
        output = torch.nn.ReLU()(output) #output corresponding to the estimated depth
        
        #model for estimate epicenter distance
        #for station1
        y1 = self.conv1_d(x1)
        y1 = self.maxpooling1_d(y1)
        y1 = self.conv2_x_d(y1)
        y1 = self.maxpooling2_d(y1)
        y1 = self.conv3_x_d(y1)
        y1 = self.maxpooling3_d(y1)
        y1 = self.conv4_x_d(y1)
        y1 = self.maxpooling4_d(y1)
        y1 = self.conv5_x_d(y1)
        y1 = self.avg_pool_d(y1)
        y1 = y1.view(y1.size(0), -1)
        y1 = self.fc_d(y1)
        y1 = torch.nn.ReLU()(y1) #output corresponding to the estimated epicenter distance of station1
        #for station1
        y2 = self.conv1_d(x2)
        y2 = self.maxpooling1_d(y2)
        y2 = self.conv2_x_d(y2)
        y2 = self.maxpooling2_d(y2)
        y2 = self.conv3_x_d(y2)
        y2 = self.maxpooling3_d(y2)
        y2 = self.conv4_x_d(y2)
        y2 = self.maxpooling4_d(y2)
        y2 = self.conv5_x_d(y2)
        y2 = self.avg_pool_d(y2)
        y2 = y2.view(y2.size(0), -1)
        y2 = self.fc_d(y2)
        y2 = torch.nn.ReLU()(y2) #output corresponding to the estimated epicenter distance of station2
        #for station1
        y3 = self.conv1_d(x3)
        y3 = self.maxpooling1_d(y3)
        y3 = self.conv2_x_d(y3)
        y3 = self.maxpooling2_d(y3)
        y3 = self.conv3_x_d(y3)
        y3 = self.maxpooling3_d(y3)
        y3 = self.conv4_x_d(y3)
        y3 = self.maxpooling4_d(y3)
        y3 = self.conv5_x_d(y3)
        y3 = self.avg_pool_d(y3)
        y3 = y3.view(y3.size(0), -1)
        y3 = self.fc_d(y3)
        y3 = torch.nn.ReLU()(y3) #output corresponding to the estimated epicenter distance of station3
        #for station1
        y4 = self.conv1_d(x4)
        y4 = self.maxpooling1_d(y4)
        y4 = self.conv2_x_d(y4)
        y4 = self.maxpooling2_d(y4)
        y4 = self.conv3_x_d(y4)
        y4 = self.maxpooling3_d(y4)
        y4 = self.conv4_x_d(y4)
        y4 = self.maxpooling4_d(y4)
        y4 = self.conv5_x_d(y4)
        y4 = self.avg_pool_d(y4)
        y4 = y4.view(y4.size(0), -1)    
        y4 = self.fc_d(y4)
        y4 = torch.nn.ReLU()(y4) #output corresponding to the estimated epicenter distance of station4
        
        return output, y1, y2, y3, y4

def rsnet34():
    """ return a RSNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3])

#%%network predictor
def predictor(input_waveform1='sampleData/waveform1.npy',
              input_waveform2='sampleData/waveform2.npy',
              input_waveform3='sampleData/waveform3.npy',
              input_waveform4='sampleData/waveform4.npy',
              input_model="Model/model_TTNN.pth"): 
    
    """ 
    
    To perform estimation on waveform data.
    
    Parameters
    ----------
    input_waveform1: str
        Path to the npy file containing waveforms of the first station.
    
    input_waveform2: str
        Path to the npy file containing waveforms of the second station.
        
    input_waveform3: str
        Path to the npy file containing waveforms of the third station.
        
    input_waveform4: str
        Path to the npy file containing waveforms of the fourth station.
            
    metadata: str
        Path to a CSV file containing metadata corresponding to the waveforms.
            
    input_model: str
        Path to the trained model. 
           
    Returns
    --------        
    dep_predict: npy file
        The depth estimation by TTNN.
    
    dis1_predict: npy file
        The epicenter distance estimation by TTNN corresponding to the first station.
    
    dis2_predict: npy file
        The epicenter distance estimation by TTNN corresponding to the second station.
    
    dis3_predict: npy file
        The epicenter distance estimation by TTNN corresponding to the thrid station.
        
    dis4_predict: npy file
        The epicenter distance estimation by TTNN corresponding to the fourth station.
    
    """  
    
    ##import waveform data
    waveform1 = np.load(input_waveform1)
    waveform2 = np.load(input_waveform2)
    waveform3 = np.load(input_waveform3)
    waveform4 = np.load(input_waveform4)
    
    waveform1 = normal(waveform1)
    waveform2 = normal(waveform2)
    waveform3 = normal(waveform3)
    waveform4 = normal(waveform4)
    
    ##CUDA support
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    ##Convert waveform data into tensor format
    waveform1_t = torch.tensor(waveform1, dtype=torch.float32).to(device)
    waveform2_t = torch.tensor(waveform2, dtype=torch.float32).to(device)
    waveform3_t = torch.tensor(waveform3, dtype=torch.float32).to(device)
    waveform4_t = torch.tensor(waveform4, dtype=torch.float32).to(device)
    
    ##  The shape of the waveform is (number of waveforms, 3, 6000). The second dimension represents the number of channels, 
    ##  and the third dimension represents the number of sampling points.
    
    ##prediction
    model = rsnet34().to(device)
    model.load_state_dict(torch.load(input_model))
    model.eval()

    with torch.no_grad():
        dep_predict, dis1_predict, dis2_predict, dis3_predict, dis4_predict = model(waveform1_t, waveform2_t, waveform3_t, waveform4_t)
    
    dep_predict = dep_predict.detach().cpu().numpy()
    dis1_predict = dis1_predict.detach().cpu().numpy()
    dis2_predict = dis2_predict.detach().cpu().numpy()
    dis3_predict = dis3_predict.detach().cpu().numpy()
    dis4_predict = dis4_predict.detach().cpu().numpy()
    
    return dep_predict, dis1_predict, dis2_predict, dis3_predict, dis4_predict
