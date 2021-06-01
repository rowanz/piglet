import sys

sys.path.append('../../')

import os
import json
from collections import defaultdict
from data.zeroshot_lm_setup.encoder import Encoder, get_encoder
from tfrecord.tfrecord_utils import S3TFRecordWriter, int64_list_feature, int64_feature
from data.thor_constants import instance_to_tfrecord

import numpy as np
import tensorflow as tf

from copy import deepcopy

from model.interact.dataloader import input_fn_builder
from data.thor_constants import numpy_to_instance_states
import pandas as pd
from google.cloud import storage
from tempfile import TemporaryDirectory
import regex as re

slim_example_decoder = tf.contrib.slim.tfexample_decoder


########################################

def get_statechange_dataset(seed=123456, zs_in_train=False, daxify=False):
    """
    Retrieve datasets from the ZSLM test corpora file
    :param seed: random seed, sets it globally but that's probably ok for now
    :return:
    """
    all_qs = []
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'annotations.jsonl'), 'r') as f:
        for i, l in enumerate(f):
            all_qs.append(json.loads(l))
            all_qs[-1]['id'] = i

    sets = {k: [v for v in all_qs if v['split'] == k] for k in ['train', 'val', 'test']}

    if not zs_in_train:
        sets['test'] += [x for x in sets['train'] if x['is_zs']]
        sets['train'] = [x for x in sets['train'] if not x['is_zs']]

    np.random.seed(seed)
    for k in ['train', ]:
        inds_perm = np.random.permutation(len(sets[k])).tolist()
        sets[k] = [sets[k][i] for i in inds_perm]

    tf.logging.info(" ".join(['{}: {}'.format(k, len(v)) for k, v in sets.items()]))

    if daxify:
        from data.concept_utils import concept_df
        from data.concept_utils import get_concepts_in_text

        zs_objs = concept_df[concept_df['is_zeroshot']]['thor_name'].unique().tolist()
        ids = np.random.permutation(len(zs_objs))
        obj_to_name = {obj: f'blicket{id}' for obj, id in zip(zs_objs, ids)}

        for split in ['train', 'val', 'test']:
            for item in sets[split]:
                for sent in ['precondition', 'action', 'postcondition']:
                    k = f'{sent}_language'
                    sent_k = item['annot'][k]
                    print(item['annot'][k])

                    all_concepts = get_concepts_in_text(sent_k)
                    replace_dict = {}
                    for v in all_concepts:
                        concept_item = concept_df.iloc[v['id']]
                        if concept_item['is_zeroshot']:
                            replace_dict[sent_k[v['start_idx']:v['end_idx']]] = obj_to_name[concept_item['thor_name']]
                    print(replace_dict)
                    for k, v in replace_dict.items():
                        sent_k = sent_k.replace(k, v)
                    item['annot'][k] = sent_k
                    print(item['annot'][k])
                    print('---')
    return sets


def create_tfrecord_statechange(encoder: Encoder, dataset_items, out_fn, pad_interval=1,
                                include_precondition=True,
                                include_action=True,
                                include_postcondition=False):
    """
    Creates a tfrecord file out of dataset_df
    :param encoder: Encoder to use
    :param dataset_items: DF to use
    :param out_fn: Where to save it
    :param pad_interval: Pad dummy tfrecords to this interval
    :param include_precondition: Include the precondition sentence.
    :param include_action: Include the action sentence.
    :param include_postcondition: Include the postcondition sentence.

    :return: all lenghs
    """
    total_written = 0
    lens = []
    prepost_lens = []
    extrasignal_lens = []
    tf.logging.info(f"Creating tfrecord for {out_fn}")
    with S3TFRecordWriter(out_fn) as writer:
        for j, row in enumerate(dataset_items):

            # Encode all sentences
            for k in ['precondition', 'action', 'postcondition']:
                row[f'{k}_bpe'] = encoder.encode(row['annot'][f'{k}_language'])

            # Encode precondition and action
            # 1. High level precondition -> Low level precondition
            # 2. Low level precondition -> high level precondition
            # 3. High level precondition / high level action -> High level postcondition
            # 4. High level precondition / high level action -> Low level postcondition
            # 5. High level postcondition -> low level postcondition
            # 6. low level postcondition -> high level postcondition

            sep = [encoder.begin_summary, encoder.end_summary]
            sents = {}
            sents['pre_act2post'] = [encoder.begin_article] + encoder.encode("predict post") + [sep[0]] + row[
                'precondition_bpe'] + [sep[1]] + row['action_bpe'] + [encoder.end_article]
            sents['post_act2pre'] = [encoder.begin_article] + encoder.encode("predict pre") + [sep[0]] + row[
                'postcondition_bpe'] + [sep[1]] + row['action_bpe'] + [encoder.end_article]

            sents['pre'] = [encoder.begin_summary] + row['precondition_bpe'] + [encoder.end_article]
            sents['post'] = [encoder.begin_article] + row['postcondition_bpe'] + [encoder.end_article]

            lens.append(max([len(v) for v in sents.values()]))
            prepost_lens.append(max(len(sents['pre']), len(sents['post'])))
            tf_record = instance_to_tfrecord(row)

            #
            # def _de_camelcase(txt):
            #     return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', txt)).lower().strip()

            # Represent objects
            objects = []
            for obj_id in range(2):
                object_name_fmt = [row['pre'][f'ObjectName{obj_id}'].lower()]
                for k in ['receptacleObjectIds', 'parentReceptacles']:
                    name_short = {'receptacleObjectIds': 'has: ', 'parentReceptacles': 'in: '}[k]
                    roi_fmt = [row['pre'][f'{k}{obj_id}'].lower(), row['post'][f'{k}{obj_id}'].lower()]

                    if roi_fmt[0] == roi_fmt[1]:
                        if roi_fmt[0] != 'none':
                            object_name_fmt.append(f'{name_short} {roi_fmt[0]}')
                    else:
                        object_name_fmt.append(f'{name_short} {roi_fmt[0]} to {roi_fmt[1]}')

                object_name_fmt = ', '.join(object_name_fmt)
                objects.append(f'( {object_name_fmt})')
            objects = ' '.join(objects).strip()
            objects = re.sub(r'\s+', ' ', objects)
            if j < 10:
                print(objects)

            sents['_extrasignal'] = encoder.encode(objects)
            extrasignal_lens.append(len(sents['_extrasignal']))

            for k, v in sents.items():
                tf_record[f'ids/{k}'] = int64_list_feature(v)
            # names = [(row['pre']['ObjectName0'], 1),
            #          (row['pre']['ObjectName1'],1),
            #          (row['pre']['ObjectName0'], 1),
            #          (row['pre']['ObjectName1'], 1),
            #          (row['pre']['Action0'], 1)]
            # names_bpe_short = []
            # for k, budget in names:
            #     enc_k = encoder.encode(k.lower().strip())
            #     names_bpe_short.extend(enc_k[:budget])
            #     for i in range(budget - len(enc_k)):
            #         names_bpe_short.append(12) #it's bpe for comma

            # names_bpe_short = [encoder.encode(n.lower().strip())[0] for n in names]
            # names_bpe_short.extend(encoder.encode(row['pre']['Action0'].lower().strip())[:2])
            # print(encoder.decode(names_bpe_short))
            # tf_record['ids/_extrasignal'] = int64_list_feature(names_bpe_short)
            ##################
            tf_record["is_real_example"] = int64_feature(1)
            tf_ex = tf.train.Example(features=tf.train.Features(feature=tf_record))
            writer.write(tf_ex.SerializeToString())
            total_written += 1

            if (j % 100 == 0) and (j > 0):
                tf.logging.info(f"Created {j}/{len(dataset_items)} ex.")
                for k, v in sorted(sents.items()):
                    tf.logging.info(f"{k:>20s}: {encoder.decode(v)}")

        extra_to_add = (pad_interval - total_written % pad_interval) % pad_interval
        for i in range(extra_to_add):
            tf_record2 = deepcopy(tf_record)
            for k, v in sents.items():
                tf_record[f'ids_{k}'] = int64_list_feature([encoder.begin_article])
            tf_record2["is_real_example"] = int64_feature(0)
            tf_ex = tf.train.Example(features=tf.train.Features(feature=tf_record2))
            writer.write(tf_ex.SerializeToString())
    tf.logging.info(f"Wrote {total_written} real ex {out_fn} and {extra_to_add} padding")
    tf.logging.info("Max prepost lens {}".format(max(prepost_lens)))
    tf.logging.info("Max ES lens {}".format(max(extrasignal_lens)))
    return lens


def evaluate_statechange(estimator, config):
    results = {}
    results_agg = []
    for split in ['val', 'test', 'train']:
        config.data['val_file'] = os.path.join(config.device['output_dir'], f'langdset-{split}.tfrecord')
        input_fn = input_fn_builder(config, is_training=False)
        result = [x for x in estimator.predict(input_fn=input_fn) if (x['is_real_example'] == 1)]

        # Eval the model
        pred_states = []
        gt_states = []
        for res in result:
            pred_state = numpy_to_instance_states(object_types=res['objects/object_types'][[0, 2]],
                                                  object_states=res['tokens'][-2:])
            gt_state = numpy_to_instance_states(object_types=res['objects/object_types'][[0, 2]],
                                                object_states=res['objects/object_states'][[1, 3]])

            for obj_id in range(2):
                ps_obj = {k[:-1]: v for k, v in pred_state.items() if k.endswith(str(obj_id))}
                gs_obj = {k[:-1]: v for k, v in gt_state.items() if k.endswith(str(obj_id))}
                ps_obj['object_ind'] = obj_id
                gs_obj['object_ind'] = obj_id
                if gs_obj['ObjectName'] != 'None':
                    pred_states.append(ps_obj)
                    gt_states.append(gs_obj)

        results[f'{split}_pred'] = pd.DataFrame(pred_states)
        results[f'{split}_gt'] = pd.DataFrame(gt_states)
        agg_res = (results[f'{split}_pred'] == results[f'{split}_gt']).mean(0)

        agg_res['all'] = (results[f'{split}_pred'] == results[f'{split}_gt']).all(1).mean()

        # Perplexity
        if 'per_example_neglogprob' in result[0]:
            neg_logprob = defaultdict(list)
            for res in result:
                for i, (ids_name, h_src) in enumerate([('pre', 'gt'), ('post', 'pred')]):
                    # Add an extra 0 for padding
                    ids_gt = np.append(res[f'ids/{ids_name}'], 0)
                    n_tok = np.where(ids_gt == 0)[0][0] - 1 # Subtract 1 bc we never predict the first one
                    neg_logprob[f'{ids_name}_{h_src}'].extend(res['per_example_neglogprob'][i, :n_tok].tolist())

            for k, v in neg_logprob.items():
                agg_res[f'{k}_ppl'] = np.exp(np.mean(v))
        if 'gen_tokens' in result[0]:
            encoder = get_encoder()

            def _decode(tok_np):
                end_tokens = [encoder.end_article, encoder.padding]
                tok_np = np.concatenate((tok_np, end_tokens), 0)
                n_tok = np.where((tok_np[:, None] == np.array(end_tokens)[None]).any(1))[0][0]
                tok_l = [x for x in tok_np[:n_tok].tolist() if x not in (encoder.begin_article,)]
                return encoder.decode(tok_l).strip()

            out_items = [{'pre_gt': _decode(res['gen_tokens'][0]),
                          'post_pred': _decode(res['gen_tokens'][1]),
                          } for res in result]

            results[f'{split}_gens'] = pd.DataFrame(out_items)

        # Add fn at beginning
        agg_res = pd.Series({'fn': config.device['output_dir']}).append(agg_res)
        # agg_res['fn'] = config.device['output_dir']
        agg_res.name = split
        results_agg.append(agg_res)
        print("Results {} {}".format(split, agg_res), flush=True)

    results['all'] = pd.DataFrame(results_agg)
    # Upload
    gclient = storage.Client()
    storage_dir = TemporaryDirectory()
    bucket_name, file_prefix = config.device['output_dir'].split('gs://', 1)[1].split('/', 1)
    bucket = gclient.get_bucket(bucket_name)

    # Save locally
    results['all'].to_csv(file_prefix.replace('/', '_') + '_all.csv')
    for k, df in results.items():
        df.to_csv(os.path.join(storage_dir.name, k + '.csv'))
        blob = bucket.blob(os.path.join(file_prefix, k + '.csv'))
        blob.upload_from_filename(os.path.join(storage_dir.name, k + '.csv'))
