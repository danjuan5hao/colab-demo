# -*- coding: utf-8 -*-
import torch.nn as nn 
import torch 

class TextAtt(nn.Module):
    def __init__(self, char_emb_dim):
        super(TextAtt, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=char_emb_dim, nhead=6)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        return self.encoder(x)