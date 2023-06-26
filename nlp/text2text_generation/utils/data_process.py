#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2023/06/03 11:20:45
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
from functools import partial
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer




class AdvertiseGen(Dataset):
    def __init__(self, data_path, tokenizer):
        # 读取数据
        self.data = self._read_data(data_path)
        self.tokenizer = tokenizer


    def __len__(self):
        '''
        返回数据量大小
        '''
        return len(self.data)
    
    def __getitem__(self, index):
        raw_content = self.data[index]["content"]
        raw_target = self.data[index]["summary"]
        input_token = self.encode_data(raw_content)
        output_token = self.encode_data(raw_target)
        return input_token, output_token, raw_target
    
    def encode_data(self, data):
        '''
        对数据进行编码
        '''
        return self.tokenizer(data, return_tensors="pt")

    def _read_data(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                data.append(line)
        return data


# 重写collate_fn函数，其输入为一个batch的sample数据
def batchify_fn(batch_data, args):
    # 找到输入数据中最长的长度
    # input_max_len = max([example[0]["input_ids"].numel() for example in batch_data])
    target_max_len  = max([example[1]["input_ids"].numel() for example in batch_data])
    
    # if input_max_len > args.model_max_length:
    #     input_max_len = args.model_max_length
    input_max_len = args.max_seq_length
    input_ids = []
    attention_mask = []
    target_ids = []    
    # target_attention_mask = []
    raw_targets = []

    for example in batch_data:
        input, target, raw_target = example
        # 对input进行padding操作或者截断
        input_pad = torch.nn.functional.pad(input["input_ids"], (0, input_max_len - input["input_ids"].numel()), value=0)
        input_ids.append(input_pad)
        attn_mask_pad = torch.nn.functional.pad(input["attention_mask"], (0, input_max_len - input["attention_mask"].numel()), value=0)
        attention_mask.append(attn_mask_pad)    
        
        # 对target进行padding操作
        target_pad = torch.nn.functional.pad(target["input_ids"], (0, target_max_len - target["input_ids"].numel()), value=0)
        target_ids.append(target_pad)
        # target_attn_pad = torch.nn.functional.pad(target["attention_mask"], (0, target_max_len - target["attention_mask"].numel()), value=0)
        # target_attention_mask.append(target_attn_pad)
        raw_targets.append(raw_target)
    
    input_ids = torch.stack(input_ids, dim=0).squeeze()
    attention_mask = torch.stack(attention_mask, dim=0).squeeze()
    target_ids = torch.stack(target_ids, dim=0).squeeze()
    # target_attention_mask = torch.stack(target_attention_mask, dim=0).squeeze()
    
    return input_ids, attention_mask, target_ids, raw_targets
    
    

if __name__ == '__main__':
    # 测试
    import sys
    import os
    print(sys.path[0])
    path = r"E:\Code\ljj_person_project\AI_Tamer\pretrained_models\nlp\mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(path)
    class args:
        train_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), r"data\AdvertiseGen\train.json")  # win
        test_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), r"data\AdvertiseGen\train.json")
        model_max_length = 512

    train_data = AdvertiseGen(args.train_path, tokenizer)
    # print(len(train_data))
    # print(train_data[0])
    
    collate_fn = partial(batchify_fn, args=args)
    train_dataloader = DataLoader(train_data, batch_size=3, shuffle=True, collate_fn=collate_fn)
    for batch in train_dataloader:
        print(type(batch))
        print(batch)
        break
