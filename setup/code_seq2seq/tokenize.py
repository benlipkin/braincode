import re
import os
import gzip
import json
import tqdm
import os.path
import multiprocessing
import random
import csv 
from src import data_dump_path_src, data_dump_path_seq2seq

def camel_case_split(identifier):
  matches = re.finditer(
    '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
    identifier
  )
  return [m.group(0) for m in matches]


def subtokens(in_list):
  good_list = []
  for tok in in_list:
    for subtok in tok.replace('_', ' ').split(' '):
      if subtok.strip() != '':
        good_list.extend(camel_case_split(subtok))
  
  return good_list


def clean_name(in_list):
  return subtokens(in_list)


def normalize_subtoken(subtoken):
  normalized = re.sub(
    r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
    re.sub(
      r'["\',`]', r'',     # Get rid of quotes and comma 
      re.sub(
        r'\s+', r'',       # Get rid of spaces
        subtoken.lower()
          .replace('\\\n', '')
          .replace('\\\t', '')
          .replace('\\\r', '')
      )
    )
  )

  return normalized.strip()


def process(item):
  src = list(filter(None, [
    normalize_subtoken(subtok) for subtok in subtokens(item.split(' '))
  ]))

  return ' '.join(src)

def transform_data(data_dir, source, dest, data_type):
  vocab = set()
  total = 0
  max_len = 0
  avg_len = 0
  min_len = None
  with open(source, 'r') as f:
    files = f.readlines()
    files = [fi[:-1] for fi in files]

  with open(dest+data_type+'.txt', 'w') as f:
      for fi in files:
          pth = os.path.join(data_dir, fi)
          text = open(pth).read()
          src = process(text)
          if len(src) == 0:
            continue
          src = src.replace('\0','')
          f.write(fi + '\t' + src + '\t' + src + '\n')
          total += 1
          res = src.split(' ')
          vocab |= set(res)
          avg_len += len(res)
          if len(res) > max_len:
            max_len = len(res)
          if min_len == None or len(res) < min_len:
            min_len = len(res)


  print('vocab: ', len(vocab))
  print('total examples: ', total)
  print('max length: ', max_len)
  print('avg_len: ', avg_len//total)
  print('min len: ', min_len)
  print('done writing')

if __name__ == '__main__':
  data_dir = sys.argv[1] # os.path.join(data_dump_path_seq2seq, 'input/src/')
  dest_dir = sys.argv[2] # os.path.join(data_dump_path_seq2seq, 'input/tokenized/')
  source_dir = sys.argv[3] # os.path.join(data_dump_path_src, 'src/')
  # obf_dir = os.path.join(data_dump_path_src, 'src/obf/')
  transform_data(source_dir, os.path.join(data_dir, 'train_files.txt'), dest_dir, 'train')
  transform_data(source_dir, os.path.join(data_dir, 'test_files.txt'), dest_dir, 'test')
  # transform_data(obf_dir, os.path.join(data_dir, 'obf_files.txt'), dest_dir, 'obf')

