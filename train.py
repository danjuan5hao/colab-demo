# -*- coding: utf-8 -*-
import torch.optim as optim 
import torch.nn as nn 

EPOCH_NUM = 60


class Trainer:
    def __init__(self, model, loss_func,
                 epoch_num,  optimier, schedule, 
                 warm_up, grad_clip):
        self.model = model
        self.epoch_num = epoch_num
        self.loss_func = loss_func

        self.optimier = optimier
        self.schedule = schedule(self.model.parameters())

    def fit(self, dataloader):
        for _ in range(self.epoch_num):
            for batch in dataloader:
                x, y = batch
                y_hat = self.model(x)
                loss = self.loss_func(y, y_hat) 


def train(model, ):
    pass 


def main():
    pass 

if __name__ == "__main__":
    main()






