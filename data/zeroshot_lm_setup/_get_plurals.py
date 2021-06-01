import pandas as pd
import inflect
import re

inflecter = inflect.engine()
df = pd.read_csv('../categories_and_banned_words.csv')

def _get_inflections(word):
    word = word.lower()
    wset = {word}

    sn = inflecter.singular_noun(word)
    if not sn:
        pn = inflecter.plural_noun(word)
        if pn:
            wset.add(pn)
    else:
        wset.add(sn)

    # sn = inflecter.singular_noun(word)
    # if sn:
    #     wset.add(sn)

    for x in sorted(wset):
        wset.add(x.replace(' ', ''))

    return sorted(wset)

def _get_word_map():
    word_to_idx = {}
    for i, (_, row) in enumerate(df.iterrows()):
        for col in ['thor_name', 'vg_name', 'oi_name', 'coco_name']:
            if not pd.isnull(row[col]):
                for word in row[col].strip().split(','):

                    if col == 'oi_name':
                        word = word.split('(')[0].strip() # Remove everything in parens and after

                    if col == 'thor_name':
                        word = ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', word))

                    for winf in _get_inflections(word):
                        if word_to_idx.get(winf, i) != i:
                            print(f"Oh no! Word {word} occurrs twice as {winf} {word_to_idx[winf]} and {i}", flush=True)
                        else:
                            word_to_idx[winf] = i
    return word_to_idx

word_to_idx = _get_word_map()
idx_to_word = [[] for i in range(max(word_to_idx.values())+1)]
for w, i in word_to_idx.items():
    idx_to_word[i].append(w)

start_idx_to_word = df['vg_name'].fillna('').str.strip().str.split(',').apply(lambda x: [y for y in x if len(y.strip()) > 0]).tolist()
for i, (sidx, eidx, (_, row)) in enumerate(zip(start_idx_to_word, idx_to_word, df.iterrows())):
    if any([x.lower() not in eidx for x in sidx]):
        print("Index {} row {}.\n{} vs {}\n~~\n".format(i, row, sidx, eidx), flush=True)

# Should be none

df['vg_name'] = [','.join(x) for x in idx_to_word]

# NOW get counts.
################################################################
################################################################
################################################################
################################################################
################################################################
# Get the object counts of all concepts
import json
from collections import defaultdict
from functools import lru_cache

# from nltk.metrics.distance import edit_distance
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

all_counts = defaultdict(int)
################ VISUAL GENOME
# vg_fn_to_split = {}
# for vg_split in ('train', 'val', 'test'):
#     with open(f'/home/rowan/code/backbones/vgpretrain_data/genome/{vg_split}.txt', 'r') as f:
#         for l in f:
#             vg_fn_to_split[l.strip().split(' ')[0]] = vg_split
# vg_splits = {'train': [], 'val': [], 'test': []}
# with open('/home/rowan/datasets3/visual_genome/butd_out/vg_annots_butd.jsonl', 'r') as f:
#     for l in tqdm(f, total=107072):
#         item = json.loads(l)
#         vg_splits[vg_fn_to_split[item['file_name']]].append(item)
#
@lru_cache(maxsize=2048)
def sanitize_vg(x):
    spacy_doc = nlp(x)
    if len(spacy_doc) == 1:
        return spacy_doc[0].lemma_.lower()
    return x.lower().replace('-', ' ')
#
# for split in ['train', 'val', 'test']:
#     for annot in vg_splits[split]:
#         for x in annot['bbox_annotations']:
#             for xa in x.get('attribute_names', []):
#                 all_counts[(sanitize_vg(xa), 'vgatt')] += 1
#             all_counts[(sanitize_vg(x['object_name']), 'vg')] += 1

# VG option B.
common_attributes = set(['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow',
                         'small', 'large', 'silver', 'wooden', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall',
                         'long', 'dark'])
def clean_string(string):
    string = string.lower().strip()
    if len(string) >= 1 and string[-1] == '.':
        return string[:-1].strip()
    return string
def clean_objects(string, common_attributes):
    ''' Return object and attribute lists '''
    string = clean_string(string)
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if not att in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:], words[:-1]
        else:
            return [string], []
    else:
        return [string], []


def clean_attributes(string):
    ''' Return attribute list '''
    string = clean_string(string)
    if string == "black and white":
        return [string]
    else:
        return [word.lower().strip() for word in string.split(" and ")]

with open('/home/rowan/datasets3/visual_genome/scene_graphs_with_attrs.json', 'r') as f:
    data = json.load(f)
for sg in data:
    for obj in sg['objects']:
        o, a = clean_objects(obj['names'][0], common_attributes)
        all_counts[(sanitize_vg(' '.join(o)), 'vg')] += 1

print("COCO", flush=True)
################# COCO

coco_splits = {'train': [], 'val': []}
coco_files_by_split = {'train': ['/home/rowan/datasets2/mscoco/annotations/instances_train2014.json',
                                 '/home/rowan/datasets2/mscoco/annotations/instances_valminusminival2014.json'],
                       'val': ['/home/rowan/datasets2/mscoco/annotations/instances_minival2014.json']}

for fn in coco_files_by_split['train']:
    # for fn in coco_files:
    with open(fn, 'r') as f:
        obj_annotations = json.load(f)
    cat2coconame = {x['id']: x['name'] for x in obj_annotations['categories']}
    for annot in tqdm(obj_annotations['annotations']):
        coco_name = cat2coconame[annot['category_id']]
        all_counts[(coco_name, 'coco')] += 1

print("OI", flush=True)
with open('/home/rowan/code/backbones/vgpretrain_data/oi-train.jsonl', 'r') as f:
    for l in tqdm(f):
        item = json.loads(l)
        for annot in item['bbox_annotations']:
            name = annot['oid_name']
            all_counts[(name, 'oi')] += 1

for dataset in ['vg', 'oi', 'coco']:
    counts = np.zeros(df.shape[0], dtype=np.int64)
    for i, ns in enumerate(df[f'{dataset}_name']):
        if pd.isnull(ns):
            continue
        for n in ns.split(','):
            counts[i] += all_counts[(n, dataset)]
    df[f'{dataset}_counts'] = counts

# Min counts = 10
total_conts = (~pd.isnull(df['thor_name'])).astype(np.float32)*10000 + df['vg_counts'] + df['oi_counts'] * 0.2 + df['coco_counts']
df = df[total_conts > 10]


df.to_csv('../categories_and_banned_words_counts.tsv', sep='\t', index=False)