import json
import multiprocessing
from memory_profiler import profile
import os
import torch
import csv
from torch import nn
import pandas as pd # 引用套件並縮寫為 pd  

vocab_size = 60005
max_len = 512
num_hiddens = 256

token_embedding = nn.Embedding(vocab_size, num_hiddens)
segment_embedding = nn.Embedding(2, num_hiddens)
pos_embedding = nn.Parameter(torch.randn(1, max_len,num_hiddens))

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
            writer.writerow(item)

def TensorToCsv(path,tensor):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        for item in tensor:
            for i in range(item.shape[0]):
                temp = []
                for j in range(item.shape[1]):
                    for m in range(item.shape[2]):
                        temp.append(item.detach().numpy()[i][j][m])
                writer.writerow(temp)

def getTensorSize(tensor):
    return tensor.element_size() * tensor.nelement()

#給<CLS>,<SEP>,<SEP>保留位置
def TruncatePairOfTokens(tokens,max_len):   
    while len(tokens) > max_len - 3:
        tokens.pop()
    return tokens

def ReadDataset(path):
    df = pd.read_csv(path)
    labels = []
    texts = []
    for i in range(len(df.Text.values)):
        text = df.Text.values[i]
        if (type(text) != str): continue
        texts.append(text.strip().lower().split(' '))
    return texts

def GetTokensAndSegments(texts):
    newTexts,newSegments = [],[]
    for text in texts:
        tokens = ['<cls>'] + text + ['<sep>']
        segments = [0] * (len(text) + 2)
        newTexts.append(tokens)
        newSegments.append(segments)
    return newTexts, newSegments

#進行padding
def PadBertInput(texts,segments,vocab, max_len):
    all_tokens_ids,all_segments,valid_lens = [],[],[]
    texts = vocab[texts]
    TextToCsv('csv/token.csv', texts)
    print(f'padding前的結果: {getTensorSize(torch.tensor(texts[0], dtype=torch.long))}bytes')
    print(f'該tensor的shape: {torch.tensor(texts[0], dtype=torch.long).shape}')
    print(torch.tensor(texts[0], dtype=torch.long))
    for (text,segment) in zip(texts,segments):
        paddingText = torch.tensor(text + [vocab['<pad>']] * (max_len - len(text)), dtype=torch.long)   
        all_tokens_ids.append(paddingText)
        all_segments.append(torch.tensor(segment + [0] * (max_len - len(segment)), dtype=torch.long))
        #valid_lens不包括<pad>
        valid_lens.append(torch.tensor(len(text), dtype=torch.float32))
    return all_tokens_ids,all_segments,valid_lens

def Embeddings(tokens,segments,valid_len,vocab_size):
    X = token_embedding(tokens) + segment_embedding(segments)
    X = X + pos_embedding.data[:, :X.shape[1], :]
    return X

def BatchEmbedding(batch_size, all_tokens_ids,all_segments,valid_lens,vocab_size):
    embeddingX = []
    i = 0
    for tokens,segments,valid_len in zip(all_tokens_ids,all_segments,valid_lens):
        if (i % batch_size == 0):
            if (i != 0):
                newTokens = torch.reshape(newTokens,(batch_size,tokens.shape[0]))
                newSegments = torch.reshape(newSegments,(batch_size,segments.shape[0]))
                newValidLen = torch.Tensor(newValidLen)
                newValidLen = torch.reshape(newValidLen,(batch_size,1))
                embeddingX.append(Embeddings(newTokens,newSegments,newValidLen,vocab_size))
            newTokens = tokens 
            newSegments = segments 
            newValidLen = [valid_len] 
        else:
            newTokens = torch.cat((newTokens,tokens),0)  
            newSegments = torch.cat((newSegments,segments),0) 
            newValidLen.append(valid_len)
        i += 1
    return embeddingX

def TextEmbedding(all_tokens_ids,all_segments,valid_lens,vocab_size):
    tokens,segments,valid_len = all_tokens_ids[0],all_segments[0],valid_lens[0]
    valid_len = torch.Tensor([valid_len])
    tokens = torch.reshape(tokens,(1,tokens.shape[0]))
    segments = torch.reshape(segments,(1,segments.shape[0]))
    valid_len = torch.reshape(valid_len,(1,1))
    embeddingX = Embeddings(tokens,segments,valid_len,vocab_size)
    return embeddingX

@profile(precision=10)
def DatasetVersion():
    max_len = 512
    dataset_path = "dataset/reviews_medium.csv"
    print("Read Dataset...\n")
    vocab = LoadVocab()
    texts= ReadDataset(dataset_path)
    TextToCsv('csv/original.csv', texts)
    print("TruncatePairOfTokens...\n")
    textsFormatList = [TruncatePairOfTokens(text, max_len)for text in texts]
    print("GetTokenAndSegments...")
    textsToken,segments = GetTokensAndSegments(textsFormatList)
    print("PadBertInput...")
    all_tokens_ids,all_segments,valid_lens = PadBertInput(textsToken, segments, vocab, max_len)
    print("BatchEmbedding...")
    embeddingX = BatchEmbedding(1, all_tokens_ids, all_segments, valid_lens, len(vocab))
    TensorToCsv('csv/embedding.csv', embeddingX)

def TextVersion():
    max_len = 256
    vocab = LoadVocab()
    textsString = "ortunately for me, I was just making a number of large throw pillows. Now I don't have the number I wanted, but I can be flexible. What if this had been an upholstery job and my piece was partially completed? Obviously, unacceptable. My advise is to feel free and shop here, but figure out your measurements on your own."
    textsStringList = [textsString.strip().lower().split(' ')]
    textsFormatList = [TruncatePairOfTokens(text, max_len)for text in textsStringList]
    textsToken,segments = GetTokensAndSegments(textsFormatList)
    all_tokens_ids,all_segments,valid_lens = PadBertInput(textsToken, segments, vocab, max_len)
    print(f'padding完的結果: {getTensorSize(all_tokens_ids[0])}bytes')
    print(f'該tensor的shape: {all_tokens_ids[0].shape}')
    print(all_tokens_ids[0])
    embeddingX = TextEmbedding(all_tokens_ids, all_segments, valid_lens, len(vocab))
    print(f'embedding完的結果: {getTensorSize(embeddingX)}bytes')
    print(f'該tensor的shape: {embeddingX.shape}')
    print(embeddingX)
    TensorToCsv('csv/embedding.csv', embeddingX)
    #print(embeddingX)

def CompareSize():
    originalSize = CheckFileSize('csv/original.csv')
    tokenSize = CheckFileSize('csv/token.csv')
    paddingSize = CheckFileSize('csv/padding.csv')
    embeddingSize = CheckFileSize('csv/embedding.csv')

if __name__ == "__main__":
    #TextVersion()
    #CompareSize()
    DatasetVersion()