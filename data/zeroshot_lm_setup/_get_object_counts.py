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

vg_categories = []
with open('/home/rowan/code/backbones/vgpretrain_data/genome/1600-400-20/objects_vocab.txt', 'r') as f:
    for i, l in enumerate(f):
        vg_categories.append(l.strip())

with open('/home/rowan/datasets3/mscoco/annotations/instances_val2014.json', 'r') as f:
    coco_instances_val = json.load(f)
    coco_categories = [x['name'] for x in coco_instances_val['categories']]

# Open images categories
openimages_categories = []
with open('/home/rowan/datasets3/openimages/class-descriptions-boxable.csv', 'r') as f:
    for l in f:
        openimages_categories.append(l.strip().split(',')[1])

all_counts = defaultdict(int)
################ VISUAL GENOME
vg_fn_to_split = {}
for vg_split in ('train', 'val', 'test'):
    with open(f'/home/rowan/code/backbones/vgpretrain_data/genome/{vg_split}.txt', 'r') as f:
        for l in f:
            vg_fn_to_split[l.strip().split(' ')[0]] = vg_split
vg_splits = {'train': [], 'val': [], 'test': []}
with open('/home/rowan/datasets3/visual_genome/butd_out/vg_annots_butd.jsonl', 'r') as f:
    for l in tqdm(f, total=107072):
        item = json.loads(l)
        vg_splits[vg_fn_to_split[item['file_name']]].append(item)

@lru_cache(maxsize=2048)
def sanitize_vg(x):
    spacy_doc = nlp(x)
    if len(spacy_doc) == 1:
        return spacy_doc[0].lemma_.lower()
    return x.lower().replace('-', ' ')

for split in ['train', 'val', 'test']:
    for annot in vg_splits[split]:
        for x in annot['bbox_annotations']:
            for xa in x.get('attribute_names', []):
                all_counts[(sanitize_vg(xa), 'vgatt')] += 1
            all_counts[(sanitize_vg(x['object_name']), 'vg')] += 1




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

df = pd.read_csv('../categories_and_banned_words.tsv', delimiter='\t')
for dataset in ['vg', 'oi', 'coco']:
    counts = np.zeros(df.shape[0], dtype=np.int64)
    for i, ns in enumerate(df[f'{dataset}_name']):
        if pd.isnull(ns):
            continue
        for n in ns.split(','):
            counts[i] += all_counts[(n, dataset)]
    df[f'{dataset}_counts'] = counts
df.to_csv('../categories_and_banned_words_counts.tsv', sep='\t', index=False)