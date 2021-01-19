import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

from Model import Transformer
from Optimizer import ScheduledOptim

PAD_TOK = '<blank>'

def get_loss(y_pred, trg, pad):
    trg = trg.contiguous().view(-1)
    loss = F.cross_entropy(y_pred, trg, ignore_index=pad, reduction='sum')
    #loss = F.nll_loss(y_pred, trg, ignore_index=pad, reduction='sum')
    return loss

def get_perform(y_pred, trg, pad):

    loss = get_loss(y_pred, trg, pad)

    trg = trg.contiguous().view(-1)
    ne = (trg!=pad)
    num_correct = (y_pred.max(1)[1]==trg)
    num_correct = num_correct.masked_select(ne).sum().item()
    non_pad = ne.sum().item()

    return loss, num_correct, non_pad


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, data, optim, opt, device):

    model.train()
    train_loss, num_words, num_correct = 0, 0, 0
    desc = '  - (Training)   '
    for batch in tqdm(data, mininterval=2, desc=desc, leave=False):

        src = (batch.src).transpose(0, 1).to(device)
        to, trg = map(lambda x: x.to(device), patch_trg(batch.trg, opt.pad2))

        optim.zero_grad()
        y_pred = model(src, to)

        loss, correct, word = get_perform(y_pred, trg, opt.pad2) 
        loss.backward()
        optim.update()

        num_correct += correct
        num_words += word
        train_loss += loss.item()

    acc = num_correct/num_words
    return train_loss/num_words, acc


def eval_epoch(model, data, device, opt):

    model.eval()
    val_loss, num_words, num_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(data, mininterval=2, desc=desc, leave=False):

            src = (batch.src).transpose(0, 1).to(device)
            to, trg = map(lambda x: x.to(device), patch_trg(batch.trg, opt.pad2))
            y_pred = model(src, to)
            loss, n_correct, n_word = get_perform(y_pred, trg, opt.pad2)

            num_words += n_word
            num_correct += n_correct
            val_loss += loss.item()

    acc = num_correct/num_words
    return val_loss/num_words, acc


def train(model, training_data, validation_data, optimizer, device, opt):

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '_train.log'
        log_valid_file = opt.log + '_valid.log'

        print('Training will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, acc, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  acc=100*acc, elapse=(time.time()-start_time)/60))

    val_loss_list = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_acc = train_epoch(model, training_data, optimizer, opt, device)
        print_performances('Training - ', train_loss, train_acc, start)

        start = time.time()
        val_loss, v_acc = eval_epoch(model, validation_data, device, opt)
        print_performances('Validation - ', val_loss, v_acc, start)

        val_loss_list += [val_loss]

        cp = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{acc:3.3f}.chkpt'.format(acc=100*v_acc)
                torch.save(cp, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if val_loss <= min(val_loss_list):
                    torch.save(cp, model_name)
                    print('Checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n'.format(
                    epoch=epoch_i, loss=val_loss,
                    ppl=math.exp(min(val_loss, 100)), acc=100*v_acc))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)    

    parser.add_argument('-train_path', default=None)   
    parser.add_argument('-val_path', default=None)     

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_mod', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-N_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-src_sharing', action='store_true')
    parser.add_argument('-trg_sharing', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_embed = opt.d_mod


    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    print(opt)

    transformer = Transformer(
        opt.src_voc_size,
        opt.trg_voc_size,
        pad1=opt.pad1,
        pad2=opt.pad2,
        is_trg_sharing=opt.trg_sharing,
        is_src_sharing=opt.src_sharing,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_mod=opt.d_mod,
        d_embed=opt.d_embed,
        d_ff=opt.d_ff,
        N_layers=opt.N_layers,
        heads=opt.heads,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_mod, opt.n_warmup_steps)
        #optim.SGD(transformer.parameters(), lr = 1e-03, momentum = 0.99), 2.0, opt.d_mod, opt.n_warmup_steps)
        #torch.optim.AdamW(transformer.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-09, weight_decay=0.01, amsgrad=False),
        #2.0, opt.d_mod, opt.n_warmup_steps)
    train(transformer, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.pad1 = data['vocab']['src'].vocab.stoi[PAD_TOK]
    opt.pad2 = data['vocab']['trg'].vocab.stoi[PAD_TOK]

    opt.src_voc_size = len(data['vocab']['src'].vocab)
    opt.trg_voc_size = len(data['vocab']['trg'].vocab)

    if opt.src_sharing:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
