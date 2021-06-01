"""
For simplicity early on, let's only handle the interaction part
"""
import sys

sys.path.append('../')
import argparse
import tensorflow as tf
import glob
import random
import os
import h5py
import numpy as np
import json
from tqdm import tqdm
from tfrecord.tfrecord_utils import _convert_image_seq_to_jpgstring, bytes_feature, float_feature, float_list_feature, \
    int64_list_feature, int64_feature, S3TFRecordWriter, _print_padding_tradeoff
from data.thor_constants import THOR_AFFORDANCES, THOR_OBJECT_TYPES, THOR_ACTIONS, _action_to_type_ind, _object_to_type_ind, _object_to_statechange_df
import hashlib
import pandas as pd
from typing import List
from collections import defaultdict

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
    '-fns_list',
    dest='fns_list',
    default='train_fns.txt',
    type=str,
    help='Where data is located',
)

args = parser.parse_args()
random.seed(args.fold)

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 640


def _dumb_hash(fn_tag):
    """
    Given a filename like j5ShHWd1Q9qv determine if it's in our fold or not

    :param fn_tag:
    :return:
    """
    fn_tag = fn_tag.split('/')[-1].split('.')[0]
    assert len(fn_tag) == 12
    pool = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    c2i = {c: i for i, c in enumerate(pool)}
    res = 0
    for c in fn_tag:
        res = (res * len(pool)) + c2i[c]
    return res


def traj_dataloader(fns_list=None):
    # This is slightly slow but not too bad
    if fns_list is None:
        print("Using all fns in data path", flush=True)
        fns_list = glob.glob(os.path.join(args.data_path, '*/*.h5'))
        fns_list = sorted(fns_list, key=_dumb_hash)

    all_fns = [fn for i, fn in enumerate(fns_list) if i % args.num_folds == args.fold]
    random.shuffle(all_fns)
    for fn in tqdm(all_fns):
        try:
            h5reader = h5py.File(fn, 'r')

            # Process it
            item = {}
            for k in ['meta_info', 'alias_object_id_to_old_object_id', 'object_id_to_states', 'output_action_results',
                      'output_actions']:
                item[k] = json.loads(h5reader[k][()].decode('utf-8'))

            item['object_ids'] = [x.decode('utf-8') for x in h5reader['object_ids'][()].tolist()]

            for k, k_v in h5reader['pos3d'].items():
                for t, v in k_v.items():
                    item['object_id_to_states'][k][t]['pos3d'] = v[()]

            # bboxes
            bbox_keys = sorted([int(k) for k in h5reader['bboxes'].keys()])
            item['bboxes'] = [h5reader['bboxes'][(str(k))][()] for k in bbox_keys]
            if not all([x.dtype == np.uint16 for x in item['bboxes']]):  # Previously I had a clipping bug
                raise ValueError("dtype")
            # item['frames'] = h5reader['frames'][()]
            # assert item['frames'].shape[1] == IMAGE_HEIGHT
            # assert item['frames'].shape[2] == IMAGE_WIDTH

            item['agent_states'] = h5reader['agent_states'][()]
            item['meta_info']['fn'] = fn
            yield item
        except Exception as e:
            print("Error with {}: {}".format(fn, str(e)), flush=True)


def _convert_bboxes(bboxes_t, t, object_ids):
    """
    Converts bboxes into tensorflow format
    :param bboxes_t: [N boxes, [obj_id, x1, y1, x2, y2]]
    :param t:  Int
    :param object_ids: Mapping obj_id -> string
    :param image_width:
    :param image_height:
    :return:
    """
    # Convert to tf format
    bbox_info_float = bboxes_t.astype(np.float32)[:, 1:5] / np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float32)[None]
    sizes = np.sqrt((bbox_info_float[:,2] - bbox_info_float[:,0]) * (bbox_info_float[:,3] - bbox_info_float[:,1]))

    # Get rid of really small objects
    big_enough = sizes > np.sqrt(4.0/(IMAGE_HEIGHT*IMAGE_WIDTH))
    bbox_info_float = bbox_info_float[big_enough]
    bboxes_t = bboxes_t[big_enough]
    sizes = sizes[big_enough]
    df = pd.DataFrame(np.column_stack([bbox_info_float, sizes]), columns=['xmin', 'ymin', 'xmax', 'ymax', 'size'])
    df['frame'] = t
    df['object_ids'] = [object_ids[i] for i in bboxes_t[:, 0]]
    df['category_ids'] = df['object_ids'].apply(_object_to_type_ind)
    return df


def _convert_action(action, main_object_ids: List[str], alias_object_id_to_old_object_id):
    """
    Convert an action into something referring to the main obj ids (dealing with aliases and stuff)
    :param action:
    :param main_object_ids:
    :param alias_object_id_to_old_object_id:
    :return:
    """
    results = {'action_id': _action_to_type_ind(action)}

    oid_to_ind = {oid: i for i, oid in enumerate(main_object_ids)}
    for alias_object_id, old_object_id in alias_object_id_to_old_object_id.items():
        oid_to_ind[alias_object_id] = oid_to_ind[old_object_id]

    if 'objectId' in action:
        results['object_id'] = oid_to_ind[action['objectId']]
    else:
        results['object_id'] = -1

    if 'receptacleObjectId' in action:
        results['receptacle_object_id'] = oid_to_ind[action['receptacleObjectId']]
    else:
        results['receptacle_object_id'] = -1
    return results


def _get_main_object_id_mappings(main_object_ids, all_object_ids,
                                 output_actions, alias_object_id_to_old_object_id):
    """
    Return a list of main object IDs, and a mapping from all object Ids to the main ones
    :param main_object_ids: Main ids identified by the sampler
    :param all_object_ids: All object IDs ever seen
    :param output_actions: All output actions -- we might need to add more main object IDs if needed
    :param alias_object_id_to_old_object_id: Aliases - e.g. if we chop somethign it changes ID. ugh
    :return: new list of main object IDs, and a mapping of objectId to main ind (or 0 otherwise). Starts at 1.
    """
    # Create a mapping of objectId -> mainObjectId ind (or nothing!)
    # Tack on enough things to main object ids if they're referenced
    if isinstance(main_object_ids, str):  # Not sure what's going on here
        main_object_ids = [main_object_ids]

    ref_oids = set([v for a in output_actions for k, v in a.items() if k.endswith('bjectId')])
    for roid in sorted(ref_oids):
        if roid not in sorted(alias_object_id_to_old_object_id.keys()) + main_object_ids:
            main_object_ids.append(roid)
    # print("{} objects: {}".format(len(main_object_ids), main_object_ids), flush=True)

    object_id_to_main_ind = {oid: -1 for oid in all_object_ids}
    for i, mi in enumerate(main_object_ids):
        object_id_to_main_ind[mi] = i
        for k, v in alias_object_id_to_old_object_id.items():
            if v == mi:
                object_id_to_main_ind[k] = i

    return main_object_ids, object_id_to_main_ind


# masses = []
# actions = set()
# arguments = set() ,_ MAIN ARGUMENTS ARE action, objectId, receptacleObjectId
fns_list = []
with open(args.fns_list, 'r') as f:
    for l in f:
        fns_list.append(l.strip('\n'))

sequence_level_stats = defaultdict(int)
instance_level_stats = defaultdict(int)


for i, item in enumerate(traj_dataloader(fns_list=fns_list)):

    main_object_ids, object_id_to_main_ind = _get_main_object_id_mappings(item['meta_info']['main_object_ids'],
                                                                          all_object_ids=item['object_ids'],
                                                                          output_actions=item['output_actions'],
                                                                          alias_object_id_to_old_object_id=item[
                                                                              'alias_object_id_to_old_object_id'])
    num_frames = len(item['bboxes'])
    df_mapping = {}

    # Compute object -> size and also get a dynamic mapping of the states over time
    object_to_size = {}
    for k, sz in item['object_id_to_states'].items():
        for s in sz.values():
            size = np.prod(s['pos3d'][-1] + 1e-8)
            object_to_size[k] = max(size, object_to_size.get(k, 0.0))

    for oid in main_object_ids:
        oid_list = [oid] + [aid for aid, oid2 in item['alias_object_id_to_old_object_id'].items() if oid2 == oid]
        df_mapping[oid] = _object_to_statechange_df([item['object_id_to_states'][k] for k in oid_list],
                                                    num_frames=num_frames,
                                                    object_to_size=object_to_size)
        column_order = df_mapping[oid].columns.tolist()

    for k, v in df_mapping.items():
        for col, col_vals in v.iteritems():
            object_type = k.split('|')[0]
            is_the_same = col_vals.values[1:] == col_vals.values[:-1]
            instance_level_stats[f'{col}~{object_type}~CHANGED'] += np.sum(~is_the_same)
            instance_level_stats[f'{col}~{object_type}~SAME'] += np.sum(is_the_same)

            if np.all(is_the_same):
                sequence_level_stats[f'{col}~{object_type}~SAME'] += 1
            else:
                sequence_level_stats[f'{col}~{object_type}~CHANGED'] += 1

    if i % 10000 == 0:
        ####
        smoothing = 0.01
        # Create two DFs, one for sequence and one for instance
        for name in ['instance', 'sequence']:
            stats = globals()[f'{name}_level_stats']
            rows = sorted(set([x.split('~')[1] for x in stats.keys()]))
            # cols = sorted(set([x.split('~')[0] for x in stats.keys()]))
            v = np.zeros((len(rows), len(column_order) + 1), dtype=np.float64)
            for i, r in enumerate(rows):
                total = 0.0
                for j, c in enumerate(column_order):
                    num = stats[f'{c}~{r}~CHANGED'] + smoothing
                    denom = stats[f'{c}~{r}~SAME'] + num + smoothing
                    v[i, j] = num / denom
                    total = stats[f'{c}~{r}~CHANGED'] + stats[f'{c}~{r}~SAME']
                v[i, -1] = total
            v_df = pd.DataFrame(v, columns=column_order + ['_total'], index=rows)
            globals()[f'{name}_df'] = v_df
            v_df.to_csv(f'{name}_df_{args.fold}of{args.num_folds}.csv')
        ##########

smoothing = 0.01
# Create two DFs, one for sequence and one for instance
for name in ['instance', 'sequence']:
    stats = globals()[f'{name}_level_stats']
    rows = sorted(set([x.split('~')[1] for x in stats.keys()]))
    # cols = sorted(set([x.split('~')[0] for x in stats.keys()]))
    v = np.zeros((len(rows), len(column_order)+1), dtype=np.float64)
    for i, r in enumerate(rows):
        total = 0.0
        for j, c in enumerate(column_order):
            num = stats[f'{c}~{r}~CHANGED'] + smoothing
            denom = stats[f'{c}~{r}~SAME'] + num + smoothing
            v[i,j] = num/denom
            total = stats[f'{c}~{r}~CHANGED'] + stats[f'{c}~{r}~SAME']
        v[i, -1] = total
    v_df = pd.DataFrame(v, columns=column_order + ['_total'], index=rows)
    globals()[f'{name}_df'] = v_df
    v_df.to_csv(f'{name}_df_{args.fold}of{args.num_folds}.csv')