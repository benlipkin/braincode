import os
import argparse
import logging
import pickle as pkl

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import code_seq2seq.seq2seq as seq2seq
from code_seq2seq.seq2seq.trainer import SupervisedTrainer
from code_seq2seq.seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from code_seq2seq.seq2seq.loss import Perplexity
from code_seq2seq.seq2seq.optim import Optimizer
from code_seq2seq.seq2seq.dataset import SourceField, TargetField, FnameField
from code_seq2seq.seq2seq.evaluator import Representation
from code_seq2seq.seq2seq.util.checkpoint import Checkpoint
from code_seq2seq.seq2seq.util.concat import torch_concat

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to tokenized train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to tokenized dev data')
parser.add_argument('--obf_path', action='store', dest='obf_path',
                    help='Path to obf data')
parser.add_argument('--reps_store_path', action='store', dest='reps_path',
                    help='Path to store generated reps')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Hyperparams
params = {
    'max_len': 150,
    'n_layers': 1,
    'hidden_size': 128, 
    'src_vocab_size': 15000, 
    'tgt_vocab_size': 5000, 
    'max_len': 50, 
    'rnn_cell':'gru',
    'batch_sz': 16, 
    'epochs': 30,
    'use_attention': True,
    'bidirectional': True,
    'dropout_p': 0.2,
    'print_every':50,
    'checkpoint_every':100,
    'teacher_ratio':0.3,
}

def prepare_dataset(train_path, dev_path, max_len):
    print('preparing dataset')
    src = SourceField()
    tgt = TargetField()
    fname = FnameField()
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=[('fname',fname),('src',src),('tgt',tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=dev_path, format='tsv',
        fields=[('fname',fname),('src',src),('tgt',tgt)],
        filter_pred=len_filter
    )

    return src, tgt, fname, train, dev

def prepare_vocab(src, tgt, fname, train, dev):
    src.build_vocab(train, max_size=params['src_vocab_size'])
    tgt.build_vocab(train, max_size=params['tgt_vocab_size'])
    fname.build_vocab(train, dev)
    input_vocab = src.vocab
    print(input_vocab)
    output_vocab = tgt.vocab
    fname_vocab = fname.vocab
    return src, tgt, fname, input_vocab, output_vocab, fname_vocab

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# Prepare dataset
src, tgt, fname, train, dev = prepare_dataset(opt.train_path, opt.dev_path, params['max_len'])

# Prepare vocab
src, tgt, fname, input_vocab, output_vocab, fname_vocab = prepare_vocab(src, tgt, fname, train, dev)

qwe
if opt.load_checkpoint is not None:
    print('inside CHECKPOINT')
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        print('init model')
        encoder = EncoderRNN(
                len(src.vocab), 
                params['max_len'], 
                params['hidden_size'],
                bidirectional=params['bidirectional'], 
                variable_lengths=True,
                rnn_cell=params['rnn_cell'],
                n_layers=params['n_layers']
                )
        decoder = DecoderRNN(
                    len(tgt.vocab), 
                    params['max_len'], 
                    params['hidden_size']* 2 if params['bidirectional'] else params['hidden_size'],
                    dropout_p=params['dropout_p'], 
                    use_attention=params['use_attention'], 
                    bidirectional=params['bidirectional'],
                    rnn_cell=params['rnn_cell'],
                    n_layers=params['n_layers'],
                    eos_id=tgt.eos_id, sos_id=tgt.sos_id
                )
        seq2seq = Seq2seq(encoder, decoder)
        print("# of model params:\nEncoder: {}\nDecoder: {}".format(
            sum(p.numel() for p in seq2seq.encoder.parameters() if p.requires_grad),
            sum(p.numel() for p in seq2seq.decoder.parameters() if p.requires_grad)
            )
        )
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        '''
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)
        '''
    # train
    print('start training')
    t = SupervisedTrainer(expt_dir=opt.expt_dir,
                          loss=loss, 
                          batch_size=params['batch_sz'],
                          checkpoint_every=params['checkpoint_every'],
                          print_every=params['print_every'])

    seq2seq = t.train(seq2seq, train,
                    num_epochs=params['epochs'], dev_data=dev,
                      optimizer=optimizer, 
                      teacher_forcing_ratio=params['teacher_ratio'])

def denumericalize(all_fnames):
    with torch.cuda.device_of(all_fnames):
        all_fnames = all_fnames.tolist()
    all_fnames = [fname.vocab.itos[ex] for ex in all_fnames]
    return all_fnames

rep = Representation(tgt_vocab=output_vocab)

all_reps_dev, all_fnames_dev = rep.get_representation(seq2seq, dev)
all_reps_train, all_fnames_train = rep.get_representation(seq2seq, train)

all_fnames_train = denumericalize(all_fnames_train)
all_fnames_dev = denumericalize(all_fnames_dev)

print('train data shape {}'.format(all_reps_train.shape))
print('train fn names shape {}'.format(len(all_fnames_train)))
print(all_fnames_train[:5])

# Dump train, dev, obf reps
def dump_data(pth, fname, ds):
    with open(pth+fname+'.torch', 'wb') as fp:
        torch.save(ds, fp)

reps_pth = opt.reps_path
dump_data(reps_pth, 'train', [all_reps_train, all_fnames_train])
dump_data(reps_pth, 'dev', [all_reps_dev, all_fnames_dev])

print('Done dumping to {}'.format(reps_pth))

