import csv 
import pandas as pd 
import torch
from transformers import BertTokenizer
import sys
import os
import numpy as np
import random 

def TensorToCsv(path,tensor):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        for i in range(tensor.shape[0]):
            temp = []
            for j in range(tensor.shape[1]):
                for m in range(tensor.shape[2]):
                    temp.append(tensor.numpy()[i][j][m])
            writer.writerow(temp)

def main():
    x = torch.tensor([[[1],[2]],[[3],[4]]])
    TensorToCsv('test.csv', x)

main()

    