# -*- coding: utf-8 -*-

from gensim.models.KeyedVectors import load_word2vec_format
import torch 
import torch.nn as nn 
from transformers import AutoModel

# class TransformerEmbedding:
#     def __init__(self, pretrain_weight):
#         self.model = AutoModel.from_pretrain(pretrain_weight)

#     def forward(self, kwargs):
#         return self.model(**kwargs)[0]


class FastEmbedding(nn.Module):
    def __init__(self, file, dim=300, fine_tune=False):
        super(FastEmbedding, self).__init__()
        model = load_word2vec_format(file, binary=False)
        self.itos = model.index2word
        vocab_size = len(self.itos)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding.from_pretrained(model.vectors)
        if fine_tune == False:
            for param in self.embedding.parameters():
                param.requires_grad = False
        
    def foward(self, x):
        return self.embedding(x)

class MergeEmbedding(nn.Module):
    def __init__(self, char_file, word_file, char_dim=300, word_dim=300, fine_tune=False):
        self.char_embedding = FastEmbedding(char_file, char_dim, fine_tune=fine_tune)
        self.word_embedding = FastEmbedding(word_file, word_dim, fine_tune=fine_tune)

        self.char_lstm = nn.LSTM(char_dim, char_dim/2, bidirectional=True)

    def forward(self, word_seq, char_seq, char_len_seq):
        """
        word_seq [batch, seq_len]
        char_seq [s]
        """
        

    
        