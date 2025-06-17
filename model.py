import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Attention(nn.Module):
    def __init__(self,heads,masked=None):
        super().__init__()
        self.key = nn.Linear(heads,heads,bias=False)
        self.query = nn.Linear(heads,heads,bias=False)
        self.value = nn.Linear(heads,heads,bias=False)
        self.masked = masked
        self.heads = heads

    def forward(self,X):
        B,T,C = X.shape
        q = self.query(X)
        k = self.key(X)
        W = q @ k.transpose(-2,-1)*self.heads**(-0.5)
        if self.masked!=None and self.masked==True:
            tril = torch.tril(torch.ones(T,T,device=X.device))
            W = W.masked_fill(tril==0,float('-inf'))
        W = F.softmax(W,dim=2)
        value = self.value(X)
        # how is variance maintained after self.value(x) step?
        return W @ value
    
class EncoderBlock(nn.Module):
    def __init__(self,heads,no_of_heads):
        super().__init__()
        self.atten = MultiHeadAttention(heads//no_of_heads,no_of_heads)
        self.layer1 = nn.LayerNorm(heads)
        self.layer2 = nn.LayerNorm(heads)
        self.feedForward1 = nn.Linear(heads,4*heads)
        self.relu = nn.ReLU()
        self.feedForward2 = nn.Linear(4*heads,heads)
        self.drop = nn.Dropout(p=0.1)

    def forward(self,X):
        X = self.layer1(X + self.drop(self.atten(X)))
        output = self.drop(self.feedForward2(self.relu(self.feedForward1(X))))
        X = self.layer2(X + output)
        return X

class CrossAttention(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.query = nn.Linear(heads,heads,bias=False)
        self.key = nn.Linear(heads,heads,bias=False)
        self.value = nn.Linear(heads,heads,bias=False)
        self.heads = heads
    def forward(self,context,X):
        q = self.query(X)
        k = self.key(context)
        w = q @ k.transpose(-2,-1)*self.heads**(-0.5)
        value = self.value(context)
        output = w @ value
        return output

class MultiHeadAttention(nn.Module):
  def __init__(self,heads,no_of_heads,masked=None):
    super().__init__()
    self.attention = nn.ModuleList([Attention(heads,masked) for _ in range(no_of_heads)])
    self.heads = heads
  def forward(self,X):
    t = torch.arange(len(self.attention))
    output = torch.concat([h(X[:,:,i*self.heads:(i+1)*self.heads]) for i,h in zip(t,self.attention)],dim=2)
    return output

class MultiHeadCrossAttention(nn.Module):
  def __init__(self,heads,no_of_heads):
    super().__init__()
    self.attention = nn.ModuleList([CrossAttention(heads) for _ in range(no_of_heads)])
    self.heads = heads
  def forward(self,context,X):
    t = torch.arange(len(self.attention))
    output = torch.concat([h(context[:,:,i*self.heads:(i+1)*self.heads],X[:,:,i*self.heads:(i+1)*self.heads]) for i,h in zip(t,self.attention)],dim=2)
    return output
  

class TransformerBlock(nn.Module):
    def __init__(self,heads,no_of_heads=4):
        super().__init__()
        self.atten = MultiHeadAttention(heads//no_of_heads,no_of_heads,True)
        self.layer1 = nn.LayerNorm(heads)
        self.layer2 = nn.LayerNorm(heads)
        self.layer3 = nn.LayerNorm(heads)
        self.crossAtten = MultiHeadCrossAttention(heads//no_of_heads,no_of_heads)
        self.feedForward1 = nn.Linear(heads,4*heads)
        self.relu = nn.ReLU()
        self.feedForward2 = nn.Linear(4*heads,heads)
        self.encoderBlock = EncoderBlock(heads,no_of_heads)
        self.drop = nn.Dropout(p=0.1)
    def forward(self,context,X):
        X = self.layer1(X + self.drop(self.atten(X)))
        context = self.encoderBlock(context)
        X = self.layer2(X + self.drop(self.crossAtten(context,X)))
        output = self.drop(self.feedForward2(self.relu(self.feedForward1(X))))
        output = self.layer3(X + output)
        return context, output

class TranslationModel(nn.Module):
    def __init__(self,heads,tokens_size,T1,T2,no_of_blocks):
        super().__init__()
        self.embedding = nn.Embedding(tokens_size,heads)
        self.blocks = nn.ModuleList([TransformerBlock(heads) for _ in range(no_of_blocks)])
        self.feedForward = nn.Linear(heads,tokens_size)
        self.embedding.weight = self.feedForward.weight
        self.posEmbedding1 = nn.Embedding(T1,heads)
        self.posEmbedding2 = nn.Embedding(T2,heads)
        self.no_of_blocks = no_of_blocks
        self.drop = nn.Dropout(p=0.1)
    def forward(self,context,X,target=None):
        B,T1 = context.shape
        B,T2 = X.shape
        # print(B,T1,T2)
        X =  self.drop(self.posEmbedding2(torch.arange(T2, device=X.device)) + self.embedding(X))
        context = self.drop(self.posEmbedding1(torch.arange(T1, device=context.device)) + self.embedding(context))
        for block in self.blocks:
            context, X = block(context,X)
        logits = self.feedForward(X)
        # logits = F.softmax(logits,dim=2)
        # it seems the above line is unnecssary since cross_entropy expects logits not probs
        # in original paper layer norm's were diff not same for every layer
        if target!=None:
            loss = F.cross_entropy(logits.view(B*T2,-1),target.view(B*T2))
            return logits, loss
        return logits

from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_enc = self.tokenizer(src, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        tgt_enc = self.tokenizer(tgt, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        X = tgt_enc["input_ids"].squeeze(0).clone()
        target = tgt_enc["input_ids"].squeeze(0)
        X[1:] = tgt_enc["input_ids"].squeeze(0)[:-1]
        X[0] = self.tokenizer.pad_token_id

        return {
            "context": src_enc["input_ids"].squeeze(0),
            "decoder_input": X,
            "target": target
        }
