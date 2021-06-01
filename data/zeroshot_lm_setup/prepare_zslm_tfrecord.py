"""
create tfrecords
"""
import sys
sys.path.append('../../')
import argparse
import random
from tfrecord.tfrecord_utils import S3TFRecordWriter, int64_list_feature
from data.zeroshot_lm_setup.wikipedia_tbooks_iterator import TOTAL_LEN, buffer_shuffler_iterator, encoder
##############################################
import pandas as pd
import spacy
import tensorflow as tf
import numpy as np
import os
from data.concept_utils import mapping_dict, concept_df
import pandas as pd

spacymodel = spacy.load("en_core_web_lg", disable=['vectors', 'textcat', 'parser', 'ner'])

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=1337,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '-out_path',
    dest='out_path',
    default='tmp',
    type=str,
    help='Where data is located',
)
parser.add_argument(
    '-seq_len',
    dest='seq_len',
    default=512,
    type=int,
    help='sl',
)
parser.add_argument(
    '-noskip',
    dest='noskip',
    action='store_true',
    help='DONT skip banned',
)
parser.add_argument(
    '-max_ex',
    dest='max_ex',
    default=1000000000000,
    type=int,
    help='at most this many examples',
)
parser.add_argument(
    '-onlythor',
    dest='onlythor',
    action='store_true',
    help='ONLY INCLUDE THOR SENTENCES',
)
args = parser.parse_args()

file_name = os.path.join(args.out_path, '{:04d}of{:04d}.tfrecord'.format(args.fold, args.num_folds))

random.seed(args.seed)
inds = [i for i in range(TOTAL_LEN) if i % args.num_folds == args.fold]
random.shuffle(inds)


concept_count = np.zeros(concept_df.shape[0], dtype=np.int64)

n_concepts_in_each = []
total_written = 0
with S3TFRecordWriter(file_name) as writer:
    for x in buffer_shuffler_iterator(inds=inds, seq_len=args.seq_len, min_doc_character_len=64,
                                      buffer_size=10000, noskip=args.noskip):
        x_dec_b = encoder.decode_list(x)

        # Identify concepts
        concepts = []
        x_dec_b_st = ''.join(x_dec_b)

        spacy_doc = spacymodel(x_dec_b_st)
        spacy_toks = [{'lemma': x.orth_.lower(), 'start': x.idx, 'pos': x.pos_} for x in spacy_doc]
        spacy_toks.append({'lemma': '', 'start': len(x_dec_b_st), 'pos': ''}) # Handle the [-1] case gracefully
        # Single lookahead
        for t0, t1 in zip(spacy_toks[:-1], spacy_toks[1:]):
            l0 = t0['lemma']
            l1 = '{}{}'.format(l0, t1['lemma'])

            # First try bigram
            if l1 in mapping_dict:
                concepts.append({
                    'lemma': l1,
                    'id': mapping_dict[l1],
                    'start_idx': t0['start'],
                })
            elif (l0 in mapping_dict) and (t0['pos'] == 'NOUN'):
                concepts.append({
                    'lemma': l0,
                    'id': mapping_dict[l0],
                    'start_idx': t0['start'],
                })

        # escape anything weird
        if any([concept_df.iloc[c['id']]['is_zeroshot'] for c in concepts]):
            # print(f"Skipping a ZS one! {x_dec_b_st} -> {concepts}", flush=True)
            continue

        # If we should only include thor sentences
        thor_concepts = [c for c in concepts if not pd.isnull(concept_df.loc[c['id'], 'thor_name'])]
        if args.onlythor:
            if len(thor_concepts) == 0:
                continue
            if len(thor_concepts) == 1 and random.random() < 0.5:
                continue
        n_concepts_in_each.append(len(thor_concepts))

        # NOTE THAT THESE WILL CONTAIN FREQUENT MISTAKES BUT MAYBE IT'S NOT SO BAD IF I FILTER BASED ON NOUNS
        end_idxs = np.cumsum([len(x) for x in x_dec_b])
        query_idxs = [x['start_idx'] for x in concepts]
        bpe_idx = np.searchsorted(end_idxs, query_idxs)

        # These are now the positions of the tags
        concept_tags = np.zeros_like(x, dtype=np.int64) - 1
        concept_tags[bpe_idx] = [c['id'] for c in concepts]

        for c in concepts:
            concept_count[c['id']] += 1


        # Verbose
        if total_written < 10:
            print(f"Example {total_written}: {x_dec_b_st}\n", flush=True)
            for i in np.where(concept_tags != -1)[0]:
                print("{}) {} -> {}".format(i, '~'.join(x_dec_b[i:(i+3)]), concept_df.iloc[concept_tags[i]]['vg_name']), flush=True)
            print("~~~~~\n", flush=True)

        features = {
            'input_ids': int64_list_feature(x),
            'concept_ids': int64_list_feature(concept_tags),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        total_written += 1
        if total_written % 1000 == 0:
            print("Have written {} articles. Mean # THOR Concepts: {}".format(total_written, np.mean(n_concepts_in_each)), flush=True)
        if total_written >= args.max_ex:
            break

    if file_name.startswith('gs://'):
        concept_temp_fn = os.path.join(writer.storage_dir.name, 'temp.npy')
        np.save(concept_temp_fn, concept_count)
        bucket = writer.gclient.get_bucket(writer.bucket_name)
        blob = bucket.blob(os.path.join(os.path.dirname(writer.file_name), 'counts',
                                                        '{:04d}of{:04d}.npy'.format(args.fold, args.num_folds)))
        blob.upload_from_filename(concept_temp_fn)

print(f"UPLOADING {total_written} articles", flush=True)
