''' Define the Transformer model '''
import torch
import math
import torch.nn as nn
import numpy as np
from Layers import *


#Calculates Positional Encoding Matrix
class PosEncoder(nn.Module):

    def __init__(self, d, n=200):
        super(PosEncoder, self).__init__()
        pe = torch.zeros(n, d)
        for pos in range(n):
            for i in range(0, d, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, inputs):
        return inputs + self.pe[:, :inputs.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(
            self, src_voc_size, d_embed, N_layers, heads, d_k, d_v,
            d_mod, d_ff, pad, dropout=0.1, n=200):

        super().__init__()

        #Init layers stack of the encoder
        self.enc_stack = nn.ModuleList([
            EncoderLayer(d_mod, d_ff, heads, d_k, d_v, dropout=dropout)
            for i in range(N_layers)])

        #Calculates embedding for each word
        self.src_embedding = nn.Embedding(src_voc_size, d_embed, padding_idx=pad)
        self.pos = PosEncoder(d_embed, n=n)
        self.dropout = nn.Dropout(p=dropout)
        self.layerNorm = nn.LayerNorm(d_mod, eps=1e-6)
        

    def forward(self, sequence, mask1):

        attentions = []

        FF = self.src_embedding(sequence)
        FF = self.pos(FF)
        FF = self.dropout(FF)
        FF = self.layerNorm(FF)

        for layer in self.enc_stack:
            FF, attention = layer(FF, mask=mask1)
            attentions += [attention]

        return FF, attentions


class Decoder(nn.Module):

    def __init__(
            self, trg_voc_size, d_embed, N_layers, heads, d_k, d_v,
            d_mod, d_ff, pad, n=200, dropout=0.1):

        super().__init__()

        #Init layers stack of the decoder
        self.dec_stack = nn.ModuleList([DecoderLayer(d_mod, d_ff, heads, d_k, d_v, dropout=dropout)
            for i in range(N_layers)])
        self.trg_embeding = nn.Embedding(trg_voc_size, d_embed, padding_idx=pad)
        self.pos = PosEncoder(d_embed, n=n)
        self.dropout = nn.Dropout(p=dropout)
        self.layerNorm = nn.LayerNorm(d_mod, eps=1e-6)

    def forward(self, sequence, mask1, enc_out, mask2):

        attentions1, attentions2 = [], []

        FF = self.trg_embeding(sequence)
        FF = self.pos(FF)
        FF = self.dropout(FF)
        FF = self.layerNorm(FF)

        for dec_layer in self.dec_stack:
            FF, attention1, attention2 = dec_layer(
                FF, enc_out, mask1, mask2)
            attentions1 += [attention1]
            attentions2 += [attention2]

        return FF, attentions1, attentions2


class Transformer(nn.Module):

    def __init__(
            self, src_voc_size, trg_voc_size, pad1, pad2,
            d_embed=512, d_mod=512, d_ff=2048,
            N_layers=6, heads=8, d_k=64, d_v=64, dropout=0.1, n=200,
            is_trg_sharing=True, is_src_sharing=True):

        super().__init__()

        self.pad1, self.pad2 = pad1, pad2

        self.encoder = Encoder(
            src_voc_size=src_voc_size, n=n,
            d_embed=d_embed, d_mod=d_mod, d_ff=d_ff,
            N_layers=N_layers, heads=heads, d_k=d_k, d_v=d_v,
            pad=pad1, dropout=dropout)

        self.decoder = Decoder(
            trg_voc_size=trg_voc_size, n=n,
            d_embed=d_embed, d_mod=d_mod, d_ff=d_ff,
            N_layers=N_layers, heads=heads, d_k=d_k, d_v=d_v,
            pad=pad2, dropout=dropout)

        #Final Linear Layer of the transformer
        self.out = nn.Linear(d_mod, trg_voc_size, bias=False)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param) 

        self.x_logit_scale = 1.

        if is_src_sharing:
            self.encoder.src_embedding.weight = self.decoder.trg_embeding.weight

        if is_trg_sharing:
            self.out.weight = self.decoder.trg_embeding.weight
            self.x_logit_scale = (d_mod ** -0.5)


    def forward(self, src, trg):
        mask1, mask2 = (src != self.pad1).unsqueeze(-2), (trg != self.pad2).unsqueeze(-2)
        subseq = (1 - torch.triu(torch.ones((1, trg.size()[1], trg.size()[1]), device=trg.device), diagonal=1)).bool()
        mask2 = mask2 & subseq
        dec_out = self.decoder(trg, mask2, self.encoder(src, mask1)[0], mask1)[0]
        seq_logit = self.out(dec_out) * self.x_logit_scale
        FF = seq_logit.view(-1, seq_logit.size(2))
        return FF
