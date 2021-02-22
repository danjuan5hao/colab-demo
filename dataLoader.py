# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset

class MydataSet(Dataset):
    def __init__(self, all_texts, all_labels, 
                 stoi, unk_id, padding_item, max_len):
        super(MydataSet, self).__init__()
        self.all_labels = all_labels
        self.all_texts = self._prepare_all_texts(all_texts, stoi, unk_id, padding_item, max_len)

    def __getitem__(self, idx):
        return torch.tensor(self.all_texts[idx], dtype=torch.long), \
               torch.tensor( int(self.all_labels[idx]) , dtype=torch.long) 

    def __len__(self):
        return len(self.all_labels) 

    def _prepare_text(self, text, stoi, unk_id,  padding_item, max_len):
        return self._pad_and_trunc([stoi.get(i, unk_id) for i in text], padding_item, max_len)

    def _pad_and_trunc(self, text, padding_item, max_len):
        text = text[:max_len]
        return text[:max_len] + [padding_item]*(max_len-len(text))

    def _prepare_all_texts(self, all_texts, stoi, unk_id, padding_item, max_len):
        return [self._prepare_text(text, stoi, unk_id, padding_item, max_len) for text in all_texts]