import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Calculates Scaled Dot Product Attention 
class Attention(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, d_k, mask=None):

        attention = torch.matmul(q, k.transpose(2, 3) / np.power(d_k, 0.5) )
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)   
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        return attention, torch.matmul(attention, v)

#Calculates Multi-Head Attention for the first sublayer of the Encoder and the Decoder 
class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_mod, d_k, d_v, dropout=0.1):
        super().__init__()

        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        #Define weight matrices 
        self.Q_Linear = nn.Linear(d_mod, heads * d_k, bias=False)
        self.K_Linear = nn.Linear(d_mod, heads * d_k, bias=False)
        self.V_Linear = nn.Linear(d_mod, heads * d_v, bias=False)
        self.O_Linear = nn.Linear(heads * d_v, d_mod, bias=False)

        self.attention = Attention()

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_mod, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        
        res = q
        bs = q.size(0)
        qs = q.size(1)
        
        #Linear operations, split into h heads, and transpose for attention dot product to get dimension : bs * n * seq_len * d_model
        q = self.Q_Linear(q).view(bs, qs, self.heads, self.d_k).transpose(1, 2)
        k = self.K_Linear(k).view(bs, k.size(1), self.heads, self.d_k).transpose(1, 2)
        v = self.V_Linear(v).view(bs, v.size(1), self.heads, self.d_v).transpose(1, 2)


        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        attention, q = self.attention(q, k, v, self.d_k, mask=mask)

        # Transpose to get dimension: b * lq * n * dv
        # Concatenate all the heads together: bs * lq * n * dv
        q = q.transpose(1, 2)
        q = q.contiguous()
        q = q.view(bs, qs, -1)
        q = self.O_Linear(q)
        q = self.dropout(q) + res
        q = self.LayerNorm(q)

        return attention, q


#Second sublayer of the encoder and Third sublayer of the decoder
#Fully connected feed-forward network
class PosFF(nn.Module):
   def __init__(self, d_mod, d_ff, dropout=0.1):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(d_mod, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_mod, d_ff) 
        self.linear2 = nn.Linear(d_ff, d_mod) 

   def forward(self, inputs):

        res = inputs
        FFN = F.relu(self.linear1(inputs))
        FFN = self.linear2(FFN)
        FFN = self.dropout(FFN)
        FFN = self.LayerNorm(self.dropout(FFN) +  res)

        return FFN


    
