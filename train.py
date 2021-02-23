# -*- coding: utf-8 -*-
import numpy as np 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F
import torch 

import random
import gensim 
from torch.utils.data import DataLoader, Subset

from model import CharModel, CharSelfAttModel
from dataLoader import MydataSet
from prepare import load_texts_and_labels

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_fscore_support


EPOCH_NUM = 200
MAX_LEN = 20

def load_gensim_model(char_emb_file):
    return gensim.models.KeyedVectors.load_word2vec_format(char_emb_file, binary=False)


def val(model, dev_dataloader, certieria):
    # dataloader = DataLoader(dev_dataset, batch_size=100)
    model.eval()
    eval_loss = 0.0
    nb_step = 0

    preds = None
    out_label_ids = None

    for batch in dev_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            texts = batch[0]
            labels = batch[1]
            rst = model(texts)
            loss = certieria(rst, labels)
            nb_step += 1
            eval_loss += loss.item()

            logits = F.softmax(rst, dim=1)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                # print(preds.shape)
                out_label_ids = labels.detach().cpu().numpy()

            else:
                tmp_rst = logits.detach().cpu().numpy()
                tmp_rst = np.argmax(tmp_rst, axis=1)
                preds = np.append(preds, tmp_rst, axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                # masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)
    # print(preds.shape)
    # print(out_label_ids.shape)
    eval_loss = eval_loss / nb_step
    return eval_loss, score_and_matric(out_label_ids, preds)


def score_and_matric(truth, preds):
    return precision_recall_fscore_support(truth, preds, average="macro")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
# default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('./runs')

    train_path = "./data/train.txt"
    char_emb_file = './data/token_vec_300.bin'
    char_emb_model = load_gensim_model(char_emb_file)
    itos = char_emb_model.index2word
    UNK_IDX = 0
    stoi = {item: idx for idx, item in enumerate(itos)}


    all_texts, all_labels = load_texts_and_labels(train_path)
    dataset = MydataSet(all_texts, all_labels, 
                       stoi, UNK_IDX, UNK_IDX, MAX_LEN)
    test_size = 10000
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # subdata = Subset(dataset, [random.randint(0, 700000) for i in range(0, 5000)])
    # data_loader = DataLoader(subdata, batch_size=64, shuffle=True)
    data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_data_loader = DataLoader(test_dataset, batch_size=100)

    EMBEDDING_DIM = 300
    CNN_OUT_DIM = 5
    # CNN_OUT_DIM = 100
    NUM_LABEL = 14
    # model = CharModel(char_emb_file, embedding_dim=EMBEDDING_DIM,
    #                  cnn_out_dim=CNN_OUT_DIM, 
    #                  num_label=NUM_LABEL)

    model = CharSelfAttModel(char_emb_file, embedding_dim=EMBEDDING_DIM,
                     max_len=MAX_LEN, 
                     num_label=NUM_LABEL)
    model.to(device)
    
    certieria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=EPOCH_NUM)
    running_loss = 0.0
    for i in range(EPOCH_NUM):
        for idx, batch in enumerate(data_loader):
            model.train()

            texts = batch[0]
            texts = texts.to(device)
            label = batch[1]
            label = label.to(device)

            rst = model(texts)
            
            loss = certieria(rst, label)
            running_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.1)

            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if idx % 1000 == 0:
                # print(loss.data)
                writer.add_scalar('training loss',
                            running_loss / 1000,
                            i * len(data_loader) + idx)
                print("train", loss.item())
                running_loss = 0.0

        evals, prf1 = val(model, val_data_loader, certieria)
        p,r,f1,_ = prf1

        writer.add_scalar('val loss', evals, (i+1)*len(data_loader))
        # writer.add_scalar('val f1', f1, (i+1)*len(data_loader))
        print("eval", evals, p, r)





