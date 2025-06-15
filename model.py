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
            tril = torch.tril(torch.ones(T,T))
            W = W.masked_fill(tril==0,float('-inf'))
            W = F.softmax(W,dim=2)
        value = self.value(X)
        # how is variance maintained after self.value(x) step?
        return W @ value
    
class EncoderBlock(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.atten = Attention(heads)
        self.layer = nn.LayerNorm(heads)
        self.feedForward1 = nn.Linear(heads,4*heads)
        self.relu = nn.ReLU()
        self.feedForward2 = nn.Linear(4*heads,heads)

    def forward(self,X):
        X = self.layer(X + self.atten(X))
        output = self.feedForward2(self.relu(self.feedForward1(X)))
        X = self.layer(X + output)
        return X

class CrossAttention(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.query = nn.Linear(heads,heads)
        self.key = nn.Linear(heads,heads)
        self.value = nn.Linear(heads,heads)
        self.heads = heads
    def forward(self,context,X):
        q = self.query(X)
        k = self.key(context)
        w = q @ k.transpose(-2,-1)*self.heads**(-0.5)
        value = self.value(context)
        output = w @ value
        return output

class TransformerBlock(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.atten = Attention(heads,True)        
        self.layer = nn.LayerNorm(heads)
        self.crossAtten = CrossAttention(heads)
        self.feedForward1 = nn.Linear(heads,4*heads)
        self.relu = nn.ReLU()
        self.feedForward2 = nn.Linear(4*heads,heads)
        self.encoderBlock = EncoderBlock(heads)
    def forward(self,context,X):
        X = self.layer(X + self.atten(X))
        context = self.encoderBlock(context)
        X = self.layer(X + self.crossAtten(context,X))
        output = self.feedForward1(X)
        output = self.relu(output)
        output = self.feedForward2(output)
        output = self.layer(X + output)
        return context, output

class TranslationModel(nn.Module):
    def __init__(self,heads,tokens_size,T1,T2):
        super().__init__()
        self.embedding = nn.Embedding(tokens_size,heads)
        self.tranform = TransformerBlock(heads)
        self.feedForward = nn.Linear(heads,tokens_size)
        self.embedding.weight = self.feedForward.weight
        self.posEmbedding1 = nn.Embedding(T1,heads)
        self.posEmbedding2 = nn.Embedding(T2,heads)
    def forward(self,context,X,target=None):
        B,T1 = context.shape
        B,T2 = X.shape
        X =  self.posEmbedding2(torch.arange(T2)) + self.embedding(X)
        context = self.posEmbedding1(torch.arange(T1)) + self.embedding(context) 
        for i in range(1):
            context, X = self.tranform(context,X)
        logits = self.feedForward(X)
        logits = F.softmax(logits,dim=2)
        if target!=None:
            loss = F.cross_entropy(logits.view(B,-1,T2),target)
            return logits, loss
        return logits
