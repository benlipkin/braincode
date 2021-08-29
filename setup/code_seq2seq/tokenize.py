import sys
import json
import builtins
import io
import keyword
import token
from tokenize import tokenize
import pickle as pkl


def _tokenize_programs(programs):
    sequences = []
    tokens = keyword.kwlist + dir(builtins)
    for program in programs:
        sequence = []
        for type, text, _, _, _ in tokenize(io.BytesIO(program.encode('utf-8')).readline):
            if type is token.STRING:
                sequence.append(1)
            elif type is token.NUMBER:
                sequence.append(2)
            elif text in tokens:
                sequence.append(3 + tokens.index(text))
            else:
                continue
        sequences.append(sequence)
    return sequences


def transform_data(src_path_train, dest_path):
  with open(src_path_train, 'r') as fp:
    files = fp.readlines()
  
  files = [fi[:-1] for fi in files][:3]
  print('Files loaded..\n {}'.format(json.dumps(files[:3], indent=2)))
  
  all_src, all_src_names = [], []
  for f in files:
    with open(f, 'r') as fp:
      src = fp.read()
      all_src.append(src)
      name = f.split("/")[-2]+"_"+f.split("/")[-1]
      all_src_names.append(name)

  tokenized_programs = _tokenize_programs(all_src)
  '''
  for p in all_src:
    print(len(p))
  for t in tokenized_programs:
    print(len(t))
  '''

  with open(dest_path, 'w') as fp:
    for n, p in zip(all_src_names, tokenized_programs):
      fp.write("{}\t{}\t{}\n".format(n, p, p))
  
  print('Done dumping tokenized programs to {}'.format(dest_path))


if __name__ == '__main__':
  train_file_path = sys.argv[1]
  test_file_path  = sys.argv[2]
  train_dest_path = sys.argv[3]
  test_dest_path  = sys.argv[4]

  dataset = transform_data(train_file_path, train_dest_path)
  dataset = transform_data(test_file_path, test_dest_path)
