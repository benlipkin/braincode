import sys
import os
import torch
import pickle as pkl
from code_seq2seq.seq2seq.evaluator import Representation
from code_seq2seq.train import prepare_dataset, params


def denumericalize(all_fnames, fnames_vocab):
    with torch.cuda.device_of(all_fnames):
        all_fnames = all_fnames.tolist()
    all_fnames = [fname_vocab.itos[ex] for ex in all_fnames]
    return all_fnames


def dump_data(pth, fname, ds):
    with open(os.path.join(pth, fname), 'wb') as fp:
        torch.save(ds, fp)


def get_representations(seq2seq_model, dataset, input_vocab, fname_vocab):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    rep = Representation(tgt_vocab=input_vocab)
    all_reps, all_fnames = rep.get_representation(seq2seq_model, dataset)
    all_fnames = denumericalize(all_fnames, fname_vocab)

    print('train data shape {}'.format(all_reps.shape))
    print('train fn names shape {}'.format(len(all_fnames)))
    print(all_fnames[:5])

    return all_reps, all_fnames


saved_model_dataset_path, rep_dump_path = sys.argv[1], sys.argv[2]
train_files_path, dev_files_path = sys.argv[3], sys.argv[4]

with open(saved_model_dataset_path, 'rb') as fp:
    saved = pkl.load(fp)

if __name__ == '__main__':
    print('here1')
    seq2seq_model, input_vocab, fname_vocab = saved['model'], saved['vocab'], saved['fname_vocab']
    print('here2')
    _, _, _, train_dataset, dev_dataset = prepare_dataset(train_files_path, dev_files_path, params['max_len'])
    print('here3')
    r, f = get_representations(seq2seq_model, train_dataset, input_vocab, fname_vocab)
    dump_data(rep_dump_path, 'train_reps.torch', [r, f])
    print('here4')
    r, f = get_representations(seq2seq_model, dev_dataset, input_vocab, fname_vocab)
    dump_data(rep_dump_path, 'test_reps.torch', [r, f])

    print('Done dumping to {}'.format(rep_dump_path))