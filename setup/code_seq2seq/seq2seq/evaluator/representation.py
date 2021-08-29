from __future__ import print_function, division

import torch
import torchtext

import code_seq2seq.seq2seq as seq2seq
from code_seq2seq.seq2seq.util.concat import torch_concat

class Representation(object):
    """ Class to gather intermediate representation produced by the model for the given dataset.

    Args:
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, batch_size=64, tgt_vocab=None):
        self.batch_size = batch_size
        self.tgt_vocab = tgt_vocab

    def get_representation(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
            
        model.eval()

        device = 'cuda:1' if torch.cuda.is_available() else -1
        # device = -1
        batch_iterator = torchtext.legacy.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = self.tgt_vocab 
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
        hidden_rep = []
        with torch.no_grad():
            all_hidden, all_fnames = None, None
            for batch in batch_iterator:
                input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                fnames = getattr(batch, seq2seq.fname_field_name)
                _, (encoder_output, encoder_hidden) = model(input_variables, input_lengths.tolist(), target_variables)
                encoder_hidden = torch.sum(encoder_hidden, 0)
                all_hidden = torch_concat(all_hidden, encoder_hidden)
                all_fnames = torch_concat(all_fnames, fnames)

        return all_hidden, all_fnames
