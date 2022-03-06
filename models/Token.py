from .File import TextToCsv,getTensorSize,TextVersionTextToCsv
import torch

#給<CLS>,<SEP>,<SEP>保留位置
def TruncatePairOfTokens(tokens,max_len):   
    while len(tokens) > max_len - 3:
        tokens.pop()
    return tokens

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