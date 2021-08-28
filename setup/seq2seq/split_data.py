import os
import sys
import json
from sklearn.model_selection import train_test_split

def save_data(split, filename):
    with open(os.path.join(dest_dir, filename), 'w+') as f:
        for file in split:
            f.write(file + '\n')
    print("wrote " + filename)

data_dir, dest_dir, extension = sys.argv[1], sys.argv[2], sys.argv[3]

data_files = []
data_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.splitext(f)[1] == extension]
print(json.dumps(data_files[:10], indent=2))
train, test = train_test_split(data_files, test_size = 0.2)
save_data(test, os.path.join(dest_dir, "test_files.txt"))
save_data(train, os.path.join(dest_dir, "train_files.txt"))

'''
obf_dir = os.path.join(data_dump_path_src, "src/obf")
obf_files = os.listdir(obf_dir)
obf_files = [f for f in obf_files if os.path.isfile(os.path.join(obf_dir, f))]
save_data(obf_files, "obf_files.txt")
'''