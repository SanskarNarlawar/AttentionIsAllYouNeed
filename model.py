import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,C,heads,masked=None):
        super().__init__()
        self.key = nn.Linear(C,heads,bias=False)
        self.query = nn.Linear(C,heads,bias=False)
        self.value = nn.Linear(C,heads,bias=False)
        self.masked = masked
    
    def forward(self,X):
        B,T,C = X.shape
        q = self.query(X)
        k = self.key(X)
        W = q @ k.transpose(-2,-1)*heads**(-0.5)
        if self.masked!=None and self.masked==True:
            tril = torch.tril(torch.ones(T,T))
            W = W.masked_fill(tril==0,float('-inf'))
            W = F.softmax(W,dim=2)
        value = self.value(X)
        return W @ value
    
class EncoderBlock(nn.Module):
    def __init__(self,C,heads):
        super().__init__()
        self.atten = Attention(C,heads)
        self.layer = nn.LayerNorm(heads)
        self.value = nn.Linear(heads,heads)
    def forward(self,X):
        X = self.layer(X + self.atten(X))
        X = self.layer(X + self.value(X))
        return X

class CrossAttention(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.query = nn.Linear(heads,heads)
        self.key = nn.Linear(heads,heads)
        self.value = nn.Linear(heads,heads)
    def forward(self,context,X):
        q = self.query(X)
        k = self.key(context)
        w = q @ k.transpose(-2,-1)*heads**(-0.5)
        value = self.value(context)
        output = w @ value
        return output

class TransformerBlock(nn.Module):
    def __init__(self,C,heads):
        super().__init__()
        self.atten = Attention(C,heads,True)        
        self.layer = nn.LayerNorm(heads)
        self.crossAtten = CrossAttention(heads)
        self.feedForward = nn.Linear(heads,heads)
        self.encoderBlock = EncoderBlock(C,heads)
    def forward(self,context,X):
        X = self.layer(X + self.atten(X))
        context = self.encoderBlock(context)
        output = self.crossAtten(context,X)
        output = self.layer(output + self.feedForward(output))
        return output

class TranslationModel(nn.Module):
    def __init__(self,C,heads,tokens_size):
        super().__init__()
        self.tranform = TransformerBlock(C,heads)
        self.feedForward = nn.Linear(heads,tokens_size)
    def forward(self,context,X):
        output = self.tranform(context,X)
        logits = self.feedForward(output)
        logits = F.softmax(logits,dim=2)
        return logits

