from __future__ import division
import torch
from torch import nn
import sys, time
from progressbar import ProgressBar

vocab_size = 60005
max_len = 512
num_hiddens = 256
token_embedding = nn.Embedding(vocab_size, num_hiddens)
segment_embedding = nn.Embedding(2, num_hiddens)
pos_embedding = nn.Parameter(torch.randn(1, max_len,num_hiddens))

def Embeddings(tokens,segments,valid_len,vocab_size):
    X = token_embedding(tokens) + segment_embedding(segments)
    X = X + pos_embedding.data[:, :X.shape[1], :]
    return X

def BatchEmbedding(batch_size, all_tokens_ids,all_segments,valid_lens,vocab_size):
    embeddingX = []
    i = 0
    total = len(valid_lens)
    pBar = ProgressBar().start()
    for tokens,segments,valid_len in zip(all_tokens_ids,all_segments,valid_lens):
        if (i % batch_size == 0):
            if (i != 0):
                newTokens = torch.reshape(newTokens,(batch_size,tokens.shape[0]))
                newSegments = torch.reshape(newSegments,(batch_size,segments.shape[0]))
                newValidLen = torch.Tensor(newValidLen)
                newValidLen = torch.reshape(newValidLen,(batch_size,1))
                temp = Embeddings(newTokens,newSegments,newValidLen,vocab_size)
                embeddingX.append(temp)
            newTokens = tokens 
            newSegments = segments 
            newValidLen = [valid_len] 
        else:
            newTokens = torch.cat((newTokens,tokens),0)  
            newSegments = torch.cat((newSegments,segments),0) 
            newValidLen.append(valid_len)
        pBar.update(int((i / (total - 1)) * 100))
        i += 1        
    pBar.finish()
    return embeddingX

def TextEmbedding(all_tokens_ids,all_segments,valid_lens,vocab_size):
    tokens,segments,valid_len = all_tokens_ids[0],all_segments[0],valid_lens[0]
    valid_len = torch.Tensor([valid_len])
    tokens = torch.reshape(tokens,(1,tokens.shape[0]))
    segments = torch.reshape(segments,(1,segments.shape[0]))
    valid_len = torch.reshape(valid_len,(1,1))
    embeddingX = Embeddings(tokens,segments,valid_len,vocab_size)
    return embeddingX