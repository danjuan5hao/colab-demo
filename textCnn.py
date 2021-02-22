# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 

class TextCnn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TextCnn, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, 3)
        self.conv2 = nn.Conv1d(in_dim, out_dim, 4)
        self.conv3 = nn.Conv1d(in_dim, out_dim, 5)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        output = torch.cat((x1,x2,x3), dim=2)
        output = output.permute(0,2,1)
        return self.dropout(output)
        
    

