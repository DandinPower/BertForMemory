from __future__ import division
import psutil
import csv
import pandas as pd
import sys, time
import os
from progressbar import ProgressBar

class MemoryCounter:
    def __init__(self):
        self.data = []
        self.process = psutil.Process(os.getpid())
    
    def Add(self):
        self.data.append(self.process.memory_info()[0])

    def GetData(self,a,b):
        value = self.data[b] - self.data[a]
        self.ShowBytes(value)
        
    def ShowBytes(self,byte):
        print(f'{byte}bytes, {byte/1024}kbs, {byte/(1024*1024)}mbs')
    
    def Clear(self):
        self.data.clear()

def CheckFileSize(path):
    p = psutil.Process()
    #print(p.io_counters())
    prev_read = p.io_counters()[2]
    with open(path,'r',encoding='utf-8') as file:
        data = file.readlines()
    fileSize = (p.io_counters()[2] - prev_read)
    print(fileSize/(1024 * 1024),'mbs')
    prev_read = p.io_counters()[2]
    return fileSize

def TextToCsv(path,texts):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        for item in texts:
            if type(item) == list:
                writer.writerow(item)
            else:
                writer.writerow(item.numpy())

def TensorToCsv(path,tensor):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        pBar = ProgressBar().start()
        writer = csv.writer(csvfile)
        z = 0
        total = len(tensor)
        for item in tensor:
            temp = []
            for i in range(item.shape[0]):
                for j in range(item.shape[1]):
                    for m in range(item.shape[2]):
                        temp.append(item.detach().numpy()[i][j][m])
            writer.writerow(temp)
            pBar.update(int((z / (total - 1)) * 100))
            z+=1
        pBar.finish()

def TextVersionTextToCsv(path,texts):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(texts)

def TextVersionTensorToCsv(path,item):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        for i in range(item.shape[0]):
            temp = []
            for j in range(item.shape[1]):
                for m in range(item.shape[2]):
                    temp.append(item.detach().numpy()[i][j][m])
            writer.writerow(temp)

def getTensorSize(tensor):
    return tensor.element_size() * tensor.nelement()

def ReadDataset(path):
    df = pd.read_csv(path)
    labels = []
    texts = []
    for i in range(len(df.Text.values)):
        text = df.Text.values[i]
        if (type(text) != str): continue
        texts.append(text.strip().lower().split(' '))
    return texts