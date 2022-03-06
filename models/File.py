import psutil
import csv
import pandas as pd

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
        writer = csv.writer(csvfile)
        for item in tensor:
            temp = []
            for i in range(item.shape[0]):
                for j in range(item.shape[1]):
                    for m in range(item.shape[2]):
                        temp.append(item.detach().numpy()[i][j][m])
            writer.writerow(temp)

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