''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm
from torchtext.data import Dataset
from Model import Transformer
from Translator import Translator

PAD_TOK = '<blank>'
UNK_TOK = '<unk>'
START = '<s>'
END = '</s>'

#Load the trained model
def load_model(opt, device):

    cp = torch.load(opt.model, map_location=device)
    mod_opt = cp['settings']

    model = Transformer(
        mod_opt.src_voc_size,
        mod_opt.trg_voc_size,

        mod_opt.pad1,
        mod_opt.pad2,
        is_trg_sharing=mod_opt.src_sharing,
        is_src_sharing=mod_opt.trg_sharing,
        d_k=mod_opt.d_k,
        d_v=mod_opt.d_v,
        d_mod=mod_opt.d_mod,
        d_embed=mod_opt.d_embed,
        d_ff=mod_opt.d_ff,
        N_layers=mod_opt.N_layers,
        heads=mod_opt.heads,
        dropout=mod_opt.dropout).to(device)

    model.load_state_dict(cp['model'])
    return model 


def main():

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-bs', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.pad1 = SRC.vocab.stoi[PAD_TOK]
    opt.pad2 = TRG.vocab.stoi[PAD_TOK]
    opt.start_idx = TRG.vocab.stoi[START]
    opt.end_idx = TRG.vocab.stoi[END]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device), bs=opt.bs, max_len=opt.max_len,
        pad1=opt.pad1, pad2=opt.pad2,
        start_idx=opt.start_idx, end_idx=opt.end_idx).to(device)

    u_i = SRC.vocab.stoi[SRC.unk_token]
    #Write the translated text to a prediction file
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            src = [SRC.vocab.stoi.get(word, u_i) for word in example.src]
            seq = translator.trans_sent(torch.LongTensor([src]).to(device))
            pred = ' '.join(TRG.vocab.itos[idx] for idx in seq)
            pred = pred.replace(START, '').replace(END, '')
            f.write(pred.strip() + '\n')

    print('Done.')

if __name__ == "__main__":
    main()
