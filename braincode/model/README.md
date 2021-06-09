## Saved seq2seq model + dictionary

To load the model and test it on unseen programs, we need two things -
- `./gru_seq2seq.pt` : a saved model
- `./vocab.pkl` : a saved vocabulary, which is essentially a map of the tokens seen during training

The util code which loads a saved model and vocabulary can be found here -- 
[https://github.mit.edu/ALFA-PL-ML/jaeyoon_brain_decoding/blob/master/decoder/util.py#L27-L80]

Sorry - this is definitely not the best way for me to providing a minimum working example.

The only dependency this has is on `Python 3.7` and `Pytorch 1.6`.

Let me know if you think setting up a seq2seq/GRU encoder-decoder would be quicker? I can share the data we trained it on?
