# -*- coding: utf-8 -*-
import torch
import torch.nn as nn  

from textCnn import TextCnn
from embedding import FastEmbedding
from selfAtt import TextAtt


class CharModel(nn.Module):
    def __init__(self, char_file_path, embedding_dim, cnn_out_dim, num_label):
        super(CharModel, self).__init__()
        self.char_embedding = FastEmbedding(char_file_path, dim=embedding_dim, fine_tune=False)
        self.encoder = TextCnn(embedding_dim, cnn_out_dim)
        self.dense_1 = nn.Linear(22*cnn_out_dim, 256)
        self.dense_2 = nn.Linear(256, num_label)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.char_embedding(x)

        x = self.encoder(x)

        x = x.reshape(batch_size, -1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

class CharSelfAttModel(nn.Module):
    def __init__(self, char_file_path, embedding_dim, max_len, num_label,):
        super(CharSelfAttModel, self).__init__()
        self.char_embedding = FastEmbedding(char_file_path, dim=embedding_dim, fine_tune=False)
        self.encoder = TextAtt(embedding_dim)
        self.dense_1 = nn.Linear(embedding_dim*max_len, 256)
        self.dense_2 = nn.Linear(256, num_label)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.char_embedding(x)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x


