# -*- coding: utf-8 -*-
import torch.optim as optim 
import torch.nn as nn 
import torch 

import gensim 
from torch.utils.data import DataLoader

from model import CharModel
from dataLoader import MydataSet
from prepare import load_texts_and_labels

EPOCH_NUM = 60
MAX_LEN = 25

def load_gensim_model(char_emb_file):
    return gensim.models.KeyedVectors.load_word2vec_format(char_emb_file, binary=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    train_path = "./data/train.txt"
    char_emb_file = r'D:\data\预训练模型\ChineseEmbedding\model\token_vec_300.bin'
    char_emb_model = load_gensim_model(char_emb_file)
    itos = char_emb_model.index2word
    UNK_IDX = 0
    stoi = {item: idx for idx, item in enumerate(itos)}


    all_texts, all_labels = load_texts_and_labels(train_path)
    dataset = MydataSet(all_texts, all_labels, 
                       stoi, UNK_IDX, UNK_IDX, MAX_LEN)

    data_loader = DataLoader(dataset, batch_size=200, shuffle=True)

    EMBEDDING_DIM = 300
    CNN_OUT_DIM = 10
    NUM_LABEL = 15
    model = CharModel(char_emb_file, embedding_dim=EMBEDDING_DIM,
                     cnn_out_dim=CNN_OUT_DIM, 
                     num_label=NUM_LABEL)

    model.to(device)
    
    certieria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=EPOCH_NUM)
    for i in range(EPOCH_NUM):
        for idx, batch in enumerate(data_loader):
            model.train()

            texts = batch[0]
            texts.to(device)
            label = batch[1]
            label.to(device)

            rst = model(texts)
            
            loss = certieria(rst, label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.1)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if idx % 1000 == 0:
                print(loss.data)






