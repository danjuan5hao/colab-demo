# -*- coding: utf-8 -*-
def load_texts_and_labels(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        all_texts = []
        all_labels = []
        for line in f.readlines():
   
            rst = line.strip().split("\t")
            label = rst[0]
            text = rst[-1]
            all_texts.append(text)
            all_labels.append(label)
    return all_texts, all_labels







