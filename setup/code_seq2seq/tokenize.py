import sys
sys.path.append('..')
from braincode.encoding import CountVectorizer


class TokenizePrograms(CountVectorizer):
  @property
  def _mode(self):
      return None

  def transform_data(self, src_path, dest_path):
    with open(src_path, 'r') as fp:
      files = fp.readlines()
      files = [fi[:-1] for fi in files]


if __name__ == '__main__':
  train_file_path = sys.argv[1]
  test_file_path = sys.argv[2]
  dest_dir_train = sys.argv[3]
  dest_dir_test = sys.argv[4]

  tp = TokenizePrograms()
  train_set = tp.transform_data(train_file_path, dest_dir_train)
  test_set  = tp.transform_data(test_file_path, dest_dir_test)

