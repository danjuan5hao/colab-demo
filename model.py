# -*- coding: utf-8 -*-
import torch
import torch.nn as nn  

from textCnn import TextCnn
from embedding import FastEmbedding

CHAR_DIM = 300


class CharModel(nn.Module):
    def __init__(self, char_file_path, max_len, cnn_out_dim, num_label):
        super(CharModel, self).__init__()
        self.char_embedding = FastEmbedding(char_file_path, dim=CHAR_DIM, fine_tune=False)
        self.encoder = TextCnn(CHAR_DIM, cnn_out_dim)
        self.dense_1 = nn.Linear(256)
        self.dense_2 = nn.Linear(256, num_label)

    def forward(self, x):
        x = self.char_embedding(x)
        x = self.encoder(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x 

