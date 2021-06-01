"""
This is a script for a model that involves TRAJECTORIES -- timewise.
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
    int64_list_feature, int64_feature, S3TFRecordWriter, _print_padding_tradeoff, traj_dataloader_v3, _get_main_object_id_mappings, _convert_bboxes, _convert_action
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
parser.add_argument(
    '-out_path',
    dest='out_path',
    default='gs://ipk-europe-west4/trajectories-v2-oct30/train',
    type=str,
    help='Where data is located',
)
parser.add_argument(
    '-no_objchange',
    dest='no_objchange',
    action='store_true',
    help='No object change',
)
parser.add_argument(
    '-no_1ary',
    dest='no_1ary',
    action='store_true',
    help='No 1ary obj',
)
parser.add_argument(
    '-no_2ary',
    dest='no_2ary',
    action='store_true',
    help='No 2ary obj',
)


args = parser.parse_args()
random.seed(args.fold)

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 640


# masses = []
# actions = set()
# arguments = set() ,_ MAIN ARGUMENTS ARE action, objectId, receptacleObjectId
fns_list = []
with open(args.fns_list, 'r') as f:
    for l in f:
        fns_list.append(l.strip('\n'))
all_fns = [fn for i, fn in enumerate(fns_list) if i % args.num_folds == args.fold]

count_log = []

object_to_state = defaultdict(list)
obj_change = []

for item, ntm in traj_dataloader_v3(all_fns, include_frames=True):
    # del ntm['parentReceptacles']
    ntm.reset_index(inplace=True)
    for _, row in ntm.iterrows():
        name = row['index'].split('|')[0]
        staticchange = {
            'meta_info': item['meta_info'],
            'agent_state': None,
            'action': {'action_id': 0, 'object_id': 0, 'receptacle_object_id': -1, 'succeeds': True,
                       'err_msg': '', 'action_name': 'Static', 'object_name': row['index'], 'receptacle_name': None},
            'pre': [dict(row)],
            'post': [dict(row)],
            'heavier': None,
            'bigger': None,
            'frames': None,
        }
        object_to_state[name].append(staticchange)

    for t, (_, action_i) in enumerate(item['actions'].iloc[:-1].iterrows()):

        # Skip some
        if action_i['action_name'] in ('MoveRight', 'MoveAhead', 'MoveLeft', 'MoveBack', 'LookDown',
                                       'LookUp', 'JumpAhead', 'RotateLeft', 'RotateRight', 'RotateHand'):
            continue
        if action_i['action_name'] == 'PickupObject':
            if random.random() > 0.25:
                continue
        if action_i['action_name'] in ('CloseObject', 'OpenObject'):
            if random.random() > 0.25:
                continue
        if action_i['action_name'] == 'PutObject':
            if action_i['succeeds'] and random.random() > 0.25:
                continue

        pre = pd.concat([v.iloc[[t]].copy().rename(index={t: k}) for k, v in item['df_mapping'].items()], 0)
        post = pd.concat([v.iloc[[t+1]].copy().rename(index={t+1: k}) for k, v in item['df_mapping'].items()], 0)
        pre['index'] = pre.index
        post['index'] = post.index

        cols = [c for c in pre.columns if c not in ['pos3d', 'distance', 'distanceraw', 'parentReceptacles', 'receptacleObjectIds',
                                                    'massraw', 'sizeraw']]

        objs_changing = (pre[cols] != post[cols]).any(1)
        attrs_changing = (pre[cols] != post[cols]).any(0)

        if (action_i['action_name'] == 'HeatUpPan') and not objs_changing.any():
            print("WTF! Pan", flush=True)

        if not objs_changing.any():
            continue

        # Get the rows
        key_objects = []
        if action_i['object_name'] is not None:
            key_objects.append(action_i['object_name'])

        if action_i['receptacle_name'] is not None:
            key_objects.append(action_i['receptacle_name'])

        for name, didchange in objs_changing.items():
            if didchange and (name not in key_objects):
                key_objects.append(name)

        # double check
        if len(key_objects) >= 3 and action_i['action_name'] == 'PutObject':
            key_objects = [action_i['object_name'], action_i['receptacle_name']]

        if len(key_objects) > 2:
            print("OH no! action={} Key objects too long: {}".format(action_i, ','.join(key_objects)), flush=True)
            key_objects = key_objects[:2]
        elif (len(key_objects) == 1):
            if random.random() > 0.1:
                # Randomly add a nonchanging object
                extra_objs = [k for k, v in objs_changing.items() if k not in key_objects]
                if len(extra_objs) > 0:
                    random.shuffle(extra_objs)
                    key_objects.append(extra_objs[0])
        elif len(key_objects) == 0:
            continue

        random.shuffle(key_objects)

        if len(key_objects) == 2:
            heavier = pre.loc[key_objects[0], 'massraw'] > pre.loc[key_objects[1], 'massraw']
            bigger = pre.loc[key_objects[0], 'sizeraw'] > pre.loc[key_objects[1], 'sizeraw']
        else:
            heavier = None
            bigger = None

        obj_change.append({
            'meta_info': item['meta_info'],
            'agent_state': item['agent_states'][t],
            'action': dict(action_i),
            'pre': [dict(pre.loc[ko]) for ko in key_objects],
            'post': [dict(post.loc[ko]) for ko in key_objects],
            'heavier': heavier,
            'bigger': bigger,
        })

        if 'frames' in item:
            obj_change[-1]['frames'] = item['frames'][t:t+2]

        if len(obj_change) % 1000 == 0:
            print("OBJ CHANGE {}, CONSTANT {}".format(len(obj_change), sum([len(v) for v in object_to_state.values()])))

            ko_and_change = []
            for ko in key_objects:
                ko_and_change.append((ko, tuple([k for k, v in (pre.loc[ko, cols] != post.loc[ko, cols]).items() if v])))

            print("I performed <{},{},{}>. Key objects\n {}\n\n".format(
                action_i['action_name'],
                action_i['object_name'],
                action_i['receptacle_name'],
                ' \n'.join(['{} -> {}'.format(z[0], ','.join(z[1])) for z in ko_and_change])), flush=True)

action_counts = defaultdict(int)
for x in obj_change:
    action_counts[x['action']['action_name']] += 1

print(f"Action counts! {action_counts}", flush=True)
print("State counts:")
initial_budget = sorted([(k,len(v)) for k, v in object_to_state.items()], key=lambda x: -x[1])
initial_budget = pd.DataFrame(initial_budget, columns=['name', 'budget']).set_index('name').squeeze()
thresh = initial_budget.values.max()
while initial_budget.values.sum() > len(obj_change):
    thresh -= 1
    initial_budget = np.minimum(initial_budget, thresh)
print(initial_budget, flush=True)

# ############## Get distribution of sizes
# sizes = []
# for k, vs in object_to_state.items():
#     for v in vs:
#         sizes.append(v['pre'][0]['distance'])
# pcts = []
# for pct in np.linspace(0.0, 100, 9):
#     print("{:.3f}: Size {}".format(pct, np.percentile(sizes, pct)), flush=True)
#     pcts.append(np.percentile(sizes, pct))
# pcts[0] = 0.0
# pcts[-1] = float('inf')
# pcts = [(x, y) for x, y in zip(pcts[:-1], pcts[1:])]
# assert False


o2s_flat_1ary = []
o2s_flat_2ary_raw = []

for k, v in object_to_state.items():
    random.shuffle(v)
    o2s_flat_1ary.extend(v[:initial_budget.loc[k]])
    random.shuffle(v)
    o2s_flat_2ary_raw.extend(v[:initial_budget.loc[k]])

o2s_flat_2ary = []
v2_inds = [i for i in range(len(o2s_flat_2ary_raw))]
random.shuffle(v2_inds)
for v1, v2i in zip(o2s_flat_2ary_raw, v2_inds):
    v2 = o2s_flat_2ary_raw[v2i]
    o2s_flat_2ary.append({
        'meta_info': {'scene_name': 'JOINT'},
        'agent_state': None,
        'action': v1['action'],
        # 'key_objects': v1['key_objects'] + v2['key_objects'],
        'pre': v1['pre'] + v2['pre'],
        'post': v1['post'] + v2['post'],
        'heavier': v1['pre'][0]['massraw'] > v2['pre'][0]['massraw'],
        'bigger': v1['pre'][0]['sizeraw'] > v2['pre'][0]['sizeraw'],
        'frames': None,
    })


everything = []
if not args.no_objchange:
    print("Obj change {}".format(len(obj_change)), flush=True)
    everything.extend(obj_change)
if not args.no_1ary:
    print("o2s_flat_1ary {}".format(len(o2s_flat_1ary)), flush=True)
    everything.extend(o2s_flat_1ary)
if not args.no_2ary:
    print("o2s_flat_2ary {}".format(len(o2s_flat_2ary)), flush=True)
    everything.extend(o2s_flat_2ary)
random.shuffle(everything)

file_name = '{}-{:04d}of{:04d}.tfrecord'.format(args.out_path, args.fold, args.num_folds)
with S3TFRecordWriter(file_name) as writer:
    for item in everything:

        # RANDOMLY SHUFFLE OBJECTS
        if (random.random() > 0.5) and len(item['pre']) == 2:
            item['pre'] = item['pre'][::-1]
            item['post'] = item['post'][::-1]
            item['heavier'] = {None: None, False: True, True: False}[item['heavier']]
            item['bigger'] = {None: None, False: True, True: False}[item['bigger']]

        action_args = []
        for k in ['object_name', 'receptacle_name']:
            ok = item['action'][k]

            if ok is None:
                action_args.append(0)

            elif ok == item['pre'][0]['index']:
                action_args.append(1)
            elif ok == item['pre'][1]['index']:
                action_args.append(2)
            else:
                import ipdb
                ipdb.set_trace()

        comparison_labels = []
        for k in ['heavier', 'bigger']:
            if item[k] is None:
                comparison_labels.append(0)
            elif item[k]:
                comparison_labels.append(1)
            else:
                comparison_labels.append(2)

        tfrecord = {
            'meta': bytes_feature(json.dumps(item['meta_info']).encode('utf-8')),
            'agent_state': float_list_feature([0]*6 if item['agent_state'] is None
                                              else [1.0] + item['agent_state'].tolist()),
            'actions/action_id': int64_feature(item['action']['action_id']),
            'actions/action_args': int64_list_feature(action_args),
            'actions/action_success': int64_feature(int(item['action']['succeeds'])),
            'comparison_labels': int64_list_feature(comparison_labels),
        }
        if item['frames'] is not None:
            img_encoded = _convert_image_seq_to_jpgstring(item['frames'])
            num_frames = 2
        else:
            img_encoded = _convert_image_seq_to_jpgstring(np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
            num_frames = 0

        tfrecord['frames/encoded'] = bytes_feature(img_encoded)
        tfrecord['frames/width'] = int64_feature(IMAGE_WIDTH)
        tfrecord['frames/height'] = int64_feature(IMAGE_HEIGHT)
        tfrecord['frames/num_frames'] = int64_feature(num_frames)
        tfrecord['frames/format'] = bytes_feature('jpeg'.encode('utf-8'))

        assert len(item['pre']) == len(item['post'])
        # [ Pre, post, pre, post].

        objects = [item['pre'][0], item['post'][0]]
        if len(item['pre']) == 2:
            objects += [item['pre'][1], item['post'][1]]

        tfrecord['objects/object_types'] = int64_list_feature([_object_to_type_ind(o['index']) for o in objects])
        for affordance_name, _, _ in THOR_AFFORDANCES:
            if affordance_name in ['canChangeTempToCold', 'canChangeTempToHot', 'salientMaterials_None']:
                continue

            tfrecord[f'objects/{affordance_name}'] = int64_list_feature(
                [o[affordance_name] for o in objects])

        tfrecord['objects/distance'] = int64_list_feature([o['distance'] for o in objects])

        example = tf.train.Example(features=tf.train.Features(feature=tfrecord))
        writer.write(example.SerializeToString())

def _get_type(x):
    if len(x.bytes_list.value) >= 1:
        return "tf.io.FixedLenFeature((), tf.string, default_value='')"
    if len(x.float_list.value) > 1:
        return "tf.io.VarLenFeature(tf.float32)"
    if len(x.float_list.value) == 1:
        return "tf.io.FixedLenFeature((), tf.float32, 1)"
    if len(x.int64_list.value) > 1:
        return "tf.io.VarLenFeature(tf.int64)"
    if len(x.int64_list.value) == 1:
        return "tf.io.FixedLenFeature((), tf.int64, 1)"
    raise ValueError(f'unknown {x}')

# create keys to features
tfrecord_sorted = sorted(tfrecord.items(), key=lambda x: x[0].split('/'))

print("keys_to_features = {", flush=True)
for k, v in tfrecord_sorted:
    print(f"    '{k}': {_get_type(v)},", flush=True)
print('}', flush=True)


for k, v in tfrecord_sorted:
    len_total = len(v.float_list.value) + len(v.bytes_list.value) + len(v.int64_list.value)
    print(f"{k} -> {len_total}", flush=True)