# -*- coding: utf-8 -*-

with open("./data/train.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        rst = line.strip().split(" ")
        label = rst[0]
        text = " ".join(rst[2: -1])


