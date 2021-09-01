import sys
import os
import torch
from torchtext.data import Dataset, Example
import pickle as pkl
from code_seq2seq.seq2seq.evaluator import Representation
from code_seq2seq.tokenize import transform_data
from code_seq2seq.train import params
from code_seq2seq.seq2seq.dataset import SourceField

def dump_data(pth, fname, ds):
    with open(os.path.join(pth, fname), 'wb') as fp:
        torch.save(ds, fp)

class TabularDataset_From_List(Dataset):
    def __init__(self, input_list, fields, **kwargs):
        fields_dict = {}
        for (n, f) in fields:
            fields_dict[n] = (n, f)
        examples = [Example.fromdict(item, fields_dict) for item in input_list]
        super(TabularDataset_From_List, self).__init__(examples, fields, **kwargs)

def get_representations(seq2seq_model, tokenized_program, max_len, input_vocab):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    tokenized_program = tokenized_program[:max_len]

    src = SourceField()
    dataset = TabularDataset_From_List(
        [{'src': p}],
        [('src', src)]
    )
    rep = Representation(tgt_vocab=input_vocab)
    all_reps, all_fnames = rep.get_representation(seq2seq_model, dataset)

    return all_reps, all_fnames


saved_model_dataset_path, rep_dump_path = sys.argv[1], sys.argv[2]
data_files_path = sys.argv[3]

with open(saved_model_dataset_path, 'rb') as fp:
    saved = pkl.load(fp)

if __name__ == '__main__':
    seq2seq_model, input_vocab = saved['model'], saved['vocab']
    max_len = params['max_len']
    train_dataset = transform_data(data_files_path)

    for p in train_dataset[:5]:
        r, f = get_representations(seq2seq_model, p, max_len, input_vocab)
        dump_data(rep_dump_path, 'data_reps.torch', [r, f])
    

    print('Done dumping to {}'.format(rep_dump_path))