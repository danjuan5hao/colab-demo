# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class TextCnn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TextCnn, self).__init__()
        self.conv11 = nn.Conv1d(in_dim, out_dim, 3)
        self.maxpool11 = nn.MaxPool1d(3, 3)
        self.conv12 = nn.Conv1d(out_dim, out_dim, 3)
        self.maxpool12 = nn.MaxPool1d(3, 3)
        self.conv13 = nn.Conv1d(out_dim, out_dim, 3)
        self.maxpool13 = nn.MaxPool1d(3, 3) 
        self.bn1 =  nn.BatchNorm1d(out_dim)

        self.conv21 = nn.Conv1d(in_dim, out_dim, 4)
        self.maxpool21 = nn.MaxPool1d(2, 2)
        self.conv22= nn.Conv1d(out_dim, out_dim, 4)
        self.maxpool22 = nn.MaxPool1d(2, 2)
        self.conv23= nn.Conv1d(out_dim, out_dim, 4)
        self.maxpool23 = nn.MaxPool1d(2, 2)
        self.bn2 =  nn.BatchNorm1d(out_dim)

        self.conv31 = nn.Conv1d(in_dim, out_dim, 5)
        self.maxpool31 = nn.MaxPool1d(2, 2)
        self.conv32 = nn.Conv1d(out_dim, out_dim, 5)
        self.maxpool32 = nn.MaxPool1d(2, 2)
        self.conv33 = nn.Conv1d(out_dim, out_dim, 5)
        self.maxpool33 = nn.MaxPool1d(2, 2)
        self.bn3 =  nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x1 = self.conv11(x)
        x1 = F.relu(x1)
        x1 = self.maxpool11(x1)
        # x1 = self.conv12(x1)
        # x1 = F.relu(x1)
        # x1 = self.maxpool12(x1)
        x1 = self.bn1(x1)
        # print(x1.size())
        # x1 = self.conv13(x1)
        # x1 = self.maxpool13(x1)

        x2 = self.conv21(x)
        x2 = F.relu(x2)
        x2 = self.maxpool21(x2)
        # x2 = self.conv22(x2)
        # x2 = F.relu(x2)
        # x2 = self.maxpool22(x2)
        x2 = self.bn2(x2)

        # print(x2.size())

        # x2 = self.conv23(x2)
        # x2 = self.maxpool23(x2)
        
        x3 = self.conv31(x)
        x3 = F.relu(x3)
        x3 = self.maxpool31(x3)
        # x3 = self.conv32(x3)
        # x3 = F.relu(x3)
        # x3 = self.maxpool32(x3)
        x3 = self.bn3(x3)
        # x3 = self.conv33(x3)
        # x3 = self.maxpool33(x3)
        output = torch.cat((x1,x2,x3), dim=2)
        output = output.permute(0,2,1)
        print(output.size())
        exit()
        return self.dropout(output)
        
    

