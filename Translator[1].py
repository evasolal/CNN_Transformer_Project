import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import Transformer

#Translate a load trained model

class Translator(nn.Module):

    def __init__(self, model, bs, max_len,
            pad1, pad2, start_idx, end_idx):
      
        super(Translator, self).__init__()

        self.max_len = max_len
        self.pad1 = pad1
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.a = 0.7
        self.bs = bs
        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[start_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((bs, max_len), pad2, dtype=torch.long))
        self.blank_seqs[:, 0] = self.start_idx
        self.register_buffer(
            'len_map', torch.arange(1, max_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg, enc_out, mask1):
        mask2 = (1 - torch.triu(
        torch.ones((1, trg.size()[1], trg.size()[1]), device=trg.device), diagonal=1)).bool()
        out = self.model.decoder(trg, mask2, enc_out, mask1)[0]
        return F.softmax(self.model.out(out), dim=-1)
        
    def get_scores_idx(self, gen, out, scores, i):
        assert len(scores.size()) == 1
        bs = self.bs
        best_sk, idxs = out[:, -1, :].topk(bs)
        scores = torch.log(best_sk).view(bs, -1) + scores.view(bs, 1)
        scores, best_idxs = scores.view(-1).topk(bs)
        best_idxs_q  = best_idxs // bs
        best_idxs_r = best_idxs % bs
        k_idxs = idxs[best_idxs_q, best_idxs_r]
        gen[:, :i] = gen[best_idxs_q, :i]
        gen[:, i] = k_idxs

        return gen, scores
        
#Translate a given sentence
    def trans_sent(self, src):
        
        max_len = self.max_len
        bs, a =  self.bs, self.a
        pad1, end_idx = self.pad1, self.end_idx 

        with torch.no_grad():
            mask = (src != pad1).unsqueeze(-2)
            enc_out = self.model.encoder(src, mask)[0]
            out = self._model_decode(self.init_seq, enc_out, mask)
        
            best_k, idxs = out[:, -1, :].topk(bs)

            scores = torch.log(best_k).view(bs)
            gen = self.blank_seqs.clone().detach()
            gen[:, 1] = idxs[0]
            enc_out = enc_out.repeat(bs, 1, 1)
            

            idx = 0   
            for i in range(2, max_len):   
                out = self._model_decode(gen[:, :i], enc_out, mask)
                gen, scores = self.get_scores_idx(gen, out, scores, i)

                eos_locs = (gen == end_idx)  
                seq_lens = self.len_map.masked_fill(~eos_locs, max_len).min(1)[0]
                eos_bool = eos_locs.sum(1) > 0
                if eos_bool.sum(0).item() == bs:
                    idx = scores.div(seq_lens.float() ** a).max(0)[1]
                    idx = idx.item()
                    break
        return gen[idx][:seq_lens[idx]].tolist()
