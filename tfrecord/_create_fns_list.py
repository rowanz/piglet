"""
For simplicity early on, let's only handle the interaction part
"""
import sys

sys.path.append('../')
import argparse
import glob
import random
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-data_path',
    dest='data_path',
    default='/home/rowan/datasets3/ipk-v1/',
    type=str,
    help='Where data is located',
)
parser.add_argument(
    '-train_perc',
    dest='train_perc',
    default=0.9,
    type=float,
    help='Training fraction',
)
parser.add_argument(
    '-dry_run',
    dest='dry_run',
    action='store_true',
    default=False,
    help='If we dont want to write at all',
)

args = parser.parse_args()
random.seed(123456)

all_fns = glob.glob(os.path.join(args.data_path, '*/*.h5'))
category_to_fns = defaultdict(list)
for fn in all_fns:
    category_to_fns[fn.split('/')[-2]].append(fn)

c2l = {c: len(v) for c, v in category_to_fns.items()}
for c, n in sorted(c2l.items(), key=lambda x: -x[1]):
    print(f"{c} -> {n} items", flush=True)


min_len = min(c2l.values())
num_train = round(args.train_perc * min_len)
num_val = min_len-num_train

print(f"Using len={min_len} ({num_train} train, {num_val} val) x {len(category_to_fns)} items -> {min_len*len(category_to_fns)}", flush=True)
if args.dry_run:
    assert False
train_fns = []
val_fns = []
for k, fns_k in category_to_fns.items():
    random.shuffle(fns_k)
    train_fns.extend(fns_k[:num_train])
    val_fns.extend(fns_k[num_train:(num_train+num_val)])

random.shuffle(train_fns)
random.shuffle(val_fns)

with open('train_fns.txt', 'w') as f:
    for l in train_fns:
        f.write(l + '\n')

with open('val_fns.txt', 'w') as f:
    for l in val_fns:
        f.write(l + '\n')