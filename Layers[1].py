''' Define the Layers '''
import torch.nn as nn
import torch
from Sublayers import *

#Encoder Layer has two sublayers :  multi-head self-attention mechanism, and positionwise fully connected feed-forward network.
class EncoderLayer(nn.Module):

    def __init__(self, d_mod, d_ff, heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(heads, d_mod, d_k, d_v, dropout=dropout)

        self.ff = PosFF(d_mod, d_ff, dropout=dropout)

    def forward(self, x, mask=None):
        attention, out = self.attention(x, x, x, mask=mask)
        FFN = self.ff(out)
        return FFN, attention

#Decoder Layer has three sublayers :  multi-head self-attention mechanism, encoder-decoder attention,  positionwise fully connected feed-forward network.
class DecoderLayer(nn.Module):

    def __init__(self, d_mod, d_ff, heads, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(heads, d_mod, d_k, d_v, dropout=dropout)
        self.enc_dec_attention = MultiHeadAttention(heads, d_mod, d_k, d_v, dropout=dropout)
        self.ff = PosFF(d_mod, d_ff, dropout=dropout)

    def forward(self, x, e_out, slf_attn_mask=None, dec_enc_attn_mask=None):
        attention, out = self.attention(x, x, x, mask=slf_attn_mask)
        enc_dec_attention, out = self.enc_dec_attention(out, e_out, e_out, mask=dec_enc_attn_mask)
        out = self.ff(out)
        return out, attention, enc_dec_attention
