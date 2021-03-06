from models.Vocab import LoadVocab
from models.Embedding import Embeddings,BatchEmbedding,TextEmbedding
from models.Token import TruncatePairOfTokens,GetTokensAndSegments,PadBertInput
from models.File import CheckFileSize,TextToCsv,TensorToCsv,TextVersionTextToCsv,TextVersionTensorToCsv,getTensorSize,ReadDataset
from memory_profiler import profile

def DatasetVersion():
    max_len = 256
    dataset_path = "dataset/reviews_20000.csv"
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
    TextToCsv('csv/padding.csv', all_tokens_ids)
    print("BatchEmbedding...")
    embeddingX = BatchEmbedding(32, all_tokens_ids, all_segments, valid_lens, len(vocab))
    print(embeddingX[0])
    #print(embeddingX[1])
    print(getTensorSize(embeddingX[0])*len(embeddingX))
    print("Output csv...")
    TensorToCsv('csv/embedding.csv', embeddingX)

def TextVersion():
    max_len = 256
    vocab = LoadVocab()
    textsString = "ortunately for me, I was just making a number of large throw pillows. Now I don't have the number I wanted, but I can be flexible. What if this had been an upholstery job and my piece was partially completed? Obviously, unacceptable. My advise is to feel free and shop here, but figure out your measurements on your own."
    textsStringList = [textsString.strip().lower().split(' ')]
    textsFormatList = [TruncatePairOfTokens(text, max_len)for text in textsStringList]
    textsToken,segments = GetTokensAndSegments(textsFormatList)
    TextVersionTextToCsv('csv/original.csv', textsToken)
    all_tokens_ids,all_segments,valid_lens = PadBertInput(textsToken, segments, vocab, max_len)
    print(f'padding????????????: {getTensorSize(all_tokens_ids[0])}bytes')
    print(f'???tensor???shape: {all_tokens_ids[0].shape}')
    print(all_tokens_ids[0])
    embeddingX = TextEmbedding(all_tokens_ids, all_segments, valid_lens, len(vocab))
    print(f'embedding????????????: {getTensorSize(embeddingX)}bytes')
    print(f'???tensor???shape: {embeddingX.shape}')
    print(embeddingX)
    TextVersionTensorToCsv('csv/embedding.csv', embeddingX)
    #print(embeddingX)

def CompareSize():
    originalSize = CheckFileSize('csv/original.csv')
    tokenSize = CheckFileSize('csv/token.csv')
    paddingSize = CheckFileSize('csv/padding.csv')
    embeddingSize = CheckFileSize('csv/embedding.csv')

@profile(precision=10)
def Memory():
    import torch
    from torch import nn
    vocab_size = 60005
    max_len = 512
    num_hiddens = 256
    token_embedding = nn.Embedding(vocab_size, num_hiddens)
    segment_embedding = nn.Embedding(2, num_hiddens)
    pos_embedding = nn.Parameter(torch.randn(1, max_len,num_hiddens))    

if __name__ == "__main__":
    #TextVersion()
    DatasetVersion()
    #CompareSize()
    #Memory()
    