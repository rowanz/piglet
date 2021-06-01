import os
import random
from io import BytesIO
from tempfile import TemporaryDirectory

import tensorflow as tf
from PIL import Image
from google.cloud import storage
import numpy as np
import glob
from tqdm import tqdm
import h5py
import json
from data.thor_constants import THOR_AFFORDANCES, THOR_OBJECT_TYPES, THOR_ACTIONS, _action_to_type_ind, \
    _object_to_type_ind, _object_to_statechange_df, _fixup_df, THOR_ACTION_TYPE_TO_IND
from typing import List
import pandas as pd


class S3TFRecordWriter(object):
    def __init__(self, fn, buffer_size=10000):
        """
        Upload to gcloud
        :param fn:
        :param buffer_size: Trying to space out idential things here by shuffling a buffer

        p(first lasts until the end,N) = (1-pflush) ^ (N/(p*buffer_size))
        each flush event removes buffer_size*p
        If the buffer size is big enough then we have good randomness I think
        """
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.io.TFRecordWriter(fn)
        self.buffer_size = buffer_size
        self.buffer = []
        self.num_written = 0

    def write(self, x):
        self.num_written += 1
        if self.buffer_size < 10:
            self.writer.write(x)
            return

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(x)
        else:
            random.shuffle(self.buffer)
            for i in range(self.buffer_size // 5):  # Pop 20%
                self.writer.write(self.buffer.pop())

    def close(self):
        # Flush buffer
        for x in self.buffer:
            self.writer.write(x)

        self.writer.close()

        if self.gclient is not None:
            print(f"UPLOADING {self.num_written}ex!!!!!", flush=True)
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _convert_image_to_jpgstring(image):
    """
    :param image: Numpy array of an image [H, W, 3]
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image_pil = Image.fromarray(image, mode='RGB')
        image_pil.save(output, format='JPEG', quality=95)
        return output.getvalue()


def _convert_image_seq_to_jpgstring(image):
    """
    :param image: Numpy array of an image [N, H, W, 3]
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image_pil = Image.fromarray(image.reshape((image.shape[0] * image.shape[1], image.shape[2], 3)), mode='RGB')
        image_pil.save(output, format='JPEG', quality=95)
        return output.getvalue()


def _print_padding_tradeoff(lens, ps=(80, 85, 90, 95, 99, 100,)):
    """
    Given the lengths of everything, print out how mcuh we lose by cutting it off to a shorter percentile
    :param lens: Lengths
    :param ps: Percentiles
    :return:
    """
    lens_array = np.array(lens)
    for p in ps:
        lensp = np.percentile(lens_array, p)
        lensused = np.minimum(lens_array, lensp).sum()
        lenstotal = np.sum(lens_array)
        wasted_space = np.sum(lensp - np.minimum(lens_array, lensp)) / (lensp * len(lens_array))
        print(
            "Lens {}%: {:.3f}. Using that as seqlength, we use {} frames of {} ({:.3f}), wasted space {:.3f}".format(
                p, np.percentile(lens_array, p), lensused, lenstotal, lensused / lenstotal, wasted_space),
            flush=True)


#############################
def traj_dataloader(all_fns, include_frames=False):
    """
    :param all_fns: list of all filenames to use
    :param include_frames: Whether to include the img
    :return:
    """
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

            if include_frames:
                item['frames'] = h5reader['frames'][()]

            item['agent_states'] = h5reader['agent_states'][()]
            item['meta_info']['fn'] = fn
            yield item
        except Exception as e:
            print("Error with {}: {}".format(fn, str(e)), flush=True)


def traj_dataloader_v2(all_fns, IMAGE_WIDTH=640, IMAGE_HEIGHT=384):
    for item in traj_dataloader(all_fns, include_frames=True):
        num_frames = item['frames'].shape[0]
        main_object_ids, object_id_to_main_ind = _get_main_object_id_mappings(item['meta_info']['main_object_ids'],
                                                                              all_object_ids=item['object_ids'],
                                                                              output_actions=item['output_actions'],
                                                                              alias_object_id_to_old_object_id=item[
                                                                                  'alias_object_id_to_old_object_id'])

        # boxes - use an extra ind that tells us what frame we're on
        # [img_id, obj_id, x1, y1, x2, y2].
        bboxes_list = [_convert_bboxes(v, t, object_ids=item['object_ids'],
                                       image_width=IMAGE_WIDTH,
                                       image_height=IMAGE_HEIGHT,
                                       ) for t, v in enumerate(item['bboxes']) if v.size > 0]
        bboxes_df = pd.concat([x for x in bboxes_list if x.size > 0], 0)
        bboxes_df['main_inds'] = bboxes_df['object_ids'].apply(lambda x: object_id_to_main_ind[x])

        # SORT bboxes_df by first, frame number, then, whether it's a main ind or not, and third the size
        item['bboxes_df'] = bboxes_df.sort_values(by=['frame', 'main_inds', 'size'], ascending=[True, False, False],
                                                  ignore_index=True)

        item['output_actions'].append({'action': 'Done'})
        item['output_action_results'].append({'action_success': True, 'action_err_msg': ''})
        # Predict next action maybe
        item['actions'] = pd.DataFrame([_convert_action(x, main_object_ids=main_object_ids,
                                                        alias_object_id_to_old_object_id=item[
                                                            'alias_object_id_to_old_object_id'])
                                        for x in item['output_actions']])
        del item['output_actions']

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
                                                        num_frames=item['frames'].shape[0],
                                                        object_to_size=object_to_size)
        item['df_mapping'] = df_mapping
        item['main_object_ids'] = main_object_ids
        item['object_id_to_main_ind'] = object_id_to_main_ind
        yield item


def _convert_bboxes(bboxes_t, t, object_ids, image_width, image_height):
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
    bbox_info_float = bboxes_t.astype(np.float32)[:, 1:5] / \
                      np.array([image_width, image_height, image_width, image_height], dtype=np.float32)[None]
    sizes = np.sqrt((bbox_info_float[:, 2] - bbox_info_float[:, 0]) * (bbox_info_float[:, 3] - bbox_info_float[:, 1]))

    # Get rid of really small objects
    big_enough = sizes > np.sqrt(4.0 / (image_height * image_width))
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


def traj_dataloader_v3(all_fns, IMAGE_WIDTH=640, IMAGE_HEIGHT=384, include_frames=False):
    for item in traj_dataloader(all_fns, include_frames=include_frames):

        main_object_ids, object_id_to_main_ind = _get_main_object_id_mappings(item['meta_info']['main_object_ids'],
                                                                              all_object_ids=item['object_ids'],
                                                                              output_actions=item['output_actions'],
                                                                              alias_object_id_to_old_object_id=item[
                                                                                  'alias_object_id_to_old_object_id'])

        # boxes - use an extra ind that tells us what frame we're on
        # [img_id, obj_id, x1, y1, x2, y2].
        bboxes_list = [_convert_bboxes(v, t, object_ids=item['object_ids'],
                                       image_width=IMAGE_WIDTH,
                                       image_height=IMAGE_HEIGHT,
                                       ) for t, v in enumerate(item['bboxes']) if v.size > 0]
        bboxes_df = pd.concat([x for x in bboxes_list if x.size > 0], 0)

        del item['bboxes']

        bboxes_df['main_inds'] = bboxes_df['object_ids'].apply(lambda x: object_id_to_main_ind[x])

        # SORT bboxes_df by first, frame number, then, whether it's a main ind or not, and third the size
        item['bboxes_df'] = bboxes_df.sort_values(by=['frame', 'main_inds', 'size'], ascending=[True, False, False],
                                                  ignore_index=True)

        item['output_actions'].append({'action': 'Done'})
        item['output_action_results'].append({'action_success': True, 'action_err_msg': ''})
        item['num_frames'] = len(item['output_actions'])

        # Predict next action maybe
        item['actions'] = pd.DataFrame([_convert_action(x, main_object_ids=main_object_ids,
                                                        alias_object_id_to_old_object_id=item[
                                                            'alias_object_id_to_old_object_id'])
                                        for x in item['output_actions']])
        item['actions']['succeeds'] = [x['action_success'] for x in item['output_action_results']]
        item['actions']['err_msg'] = [x['action_err_msg'] for x in item['output_action_results']]

        del item['output_action_results']

        item['actions']['action_name'] = item['actions']['action_id'].apply(lambda x: THOR_ACTIONS[x - 1])
        item['actions']['object_name'] = item['actions']['object_id'].apply(
            lambda x: main_object_ids[x] if x >= 0 else None)
        item['actions']['receptacle_name'] = item['actions']['receptacle_object_id'].apply(
            lambda x: main_object_ids[x] if x >= 0 else None)

        bad_cols = ['canChangeTempToHot', 'canChangeTempToCold', 'salientMaterials_None']

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
                                                        num_frames=item['num_frames'],
                                                        object_to_size=object_to_size,
                                                        include_pos=True,
                                                        agent_states=item['agent_states'])

            # FIX BUGS
            object_name = oid.split('|')[0]
            # Coffee machines
            if object_name in ('CoffeeMachine', 'StoveBurner'):
                assert not df_mapping[oid]['isBroken'].any()
                df_mapping[oid]['breakable'] = False

            # These things haven't been changing state to 'slicd'
            if object_name.startswith(('Potato', 'Tomato', 'Apple', 'Lettuce', 'Bread', 'Egg')):
                df_mapping[oid]['isSliced'] |= (~df_mapping[oid]['sliceable'])
                df_mapping[oid].loc[:, 'cookable'] = True
            elif df_mapping[oid]['salientMaterials_Food'].any():
                import ipdb
                ipdb.set_trace()

            elif len(df_mapping[oid]['sliceable'].value_counts()) > 1:
                import ipdb
                ipdb.set_trace()

            # some objects should ALWAYS be at room temperature
            if object_name in ('CounterTop',):
                df_mapping[oid]['ObjectTemperature'] = 1

            # Objects never cool down UNLESS they are put in a fridge
            # temp_lowered = [False] + (df_mapping[oid]['ObjectTemperature'].values[1:] < df_mapping[oid]['ObjectTemperature'].values[:-1]).tolist()
            # if any(temp_lowered):
            #     import ipdb
            #     ipdb.set_trace()

            # Don't change these

            # Some bugs with things not getting cooked in the microwave (or even getting placed there?)
            microwave_rows = item['actions'].apply(
                lambda row: (row['action_name'] == 'PutObject') and row['receptacle_name'].startswith('Microwave') and (
                            row['object_name'] == oid) and row['succeeds'], axis=1)
            if microwave_rows.any():
                inds = df_mapping[oid].index > np.min(np.where(microwave_rows)[0])
                if df_mapping[oid]['cookable'].any():
                    df_mapping[oid].loc[inds, 'isCooked'] = True
                df_mapping[oid].loc[inds, 'ObjectTemperature'] = 2

            # Kill these columns
            df_mapping[oid] = df_mapping[oid][[c for c in df_mapping[oid].columns
                                               if c not in bad_cols]]

        ############################
        # fix stoveburner nonsense
        if item['actions'].apply(lambda row: (row['action_name'] == 'ToggleObjectOn') and (
                row['object_name'].split('|')[0] == 'StoveKnob'), axis=1).any():

            # First FAIL FAST if not enough important objects
            needed = ['StoveBurner', 'StoveKnob']
            if not all([n in ' '.join(main_object_ids) for n in needed]) or len(main_object_ids) < 4:
                continue

            # Create a new actions df
            actions2 = []
            for t_, row in item['actions'].iterrows():
                actions2.append(row)
                if (row['action_name'] == 'ToggleObjectOn') and (row['object_name'].split('|')[0] == 'StoveKnob'):
                    # Find a stoveburner
                    stoveburner_id = [k for k, v in df_mapping.items() if
                                      k.startswith('StoveBurner') and v.shape[0] > (t_ + 1) and (
                                                  v.iloc[t_ + 1:]['ObjectTemperature'] == 2).any()]
                    if len(stoveburner_id) == 0:
                        import ipdb
                        ipdb.set_trace()
                    stoveburner_id = stoveburner_id[0]

                    pan_id = []
                    for t__, v__ in item['object_id_to_states'][stoveburner_id].items():
                        if int(t__) > t_:
                            for r_o_id in v__['receptacleObjectIds']:
                                if r_o_id in df_mapping and df_mapping[r_o_id].iloc[0]['receptacle']:
                                    pan_id.append(r_o_id)
                    #
                    #
                    # pan_id = [k for k in item['object_id_to_states'][stoveburner_id][str(t_+1)]['receptacleObjectIds']
                    #           if item['object_id_to_states'][stoveburner_id][str(t_+1)]['receptacle']]
                    if len(pan_id) == 0:
                        import ipdb
                        ipdb.set_trace()
                    pan_id = pan_id[0]

                    # OK now we can add the dummy action
                    actions2.append(pd.Series({
                        'action_id': THOR_ACTION_TYPE_TO_IND['HeatUpPan'],
                        'object_id': object_id_to_main_ind[pan_id],
                        'receptacle_object_id': -1,
                        'succeeds': True,
                        'err_msg': '',
                        'action_name': 'HeatUpPan',
                        'object_name': pan_id,
                        'receptacle_name': None,
                    }, name=t_))

            # NOTE THAT THIS RENDERS object_id_to_states OBSOLETE
            actions2 = pd.DataFrame(actions2)
            idx = actions2.index.values

            # Create an alternate index where instead of like
            # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            #        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30])
            # see that 28?
            # we have
            # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            #        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30])
            # Since at 28 we have "ToggleObjectOn" and then the new action "HeatUpPan"
            idx2 = np.copy(idx)
            for i in np.where(idx[1:] == idx[:-1])[0]:
                idx2[i + 1] = idx[i + 1] + 1

            item['agent_states'] = item['agent_states'][idx]

            bboxes_df_grouped = item['bboxes_df'].groupby('frame')
            item['bboxes_df'] = pd.concat(
                [bboxes_df_grouped.get_group(t) for t in idx if t in bboxes_df_grouped.groups], 0).reset_index(
                drop=True)

            if ('frames' in item) and (len(idx) != item['num_frames']):
                item['frames'] = item['frames'][idx]
            item['num_frames'] = len(idx)

            item['actions'] = actions2.reset_index(drop=True)
            for k in sorted(df_mapping.keys()):
                if k.startswith(('StoveBurner', 'StoveKnob')):
                    df_mapping[k] = df_mapping[k].iloc[idx2].reset_index(drop=True)
                else:
                    df_mapping[k] = df_mapping[k].iloc[idx].reset_index(drop=True)

            # This is stupid but whatever
            for t, row in item['actions'].iterrows():
                if (row['action_name'] == 'ToggleObjectOn') and row['object_name'].startswith('StoveKnob'):
                    for k in sorted(df_mapping.keys()):
                        # Heat up all stove burners
                        if k.startswith('StoveBurner'):
                            df_mapping[k].loc[t + 1:, 'ObjectTemperature'] = 2
                if row['action_name'] == 'HeatUpPan':
                    for k in sorted(df_mapping.keys()):
                        if (df_mapping[k].loc[t:(t + 5), 'ObjectTemperature'] == 2).any():
                            # print("FIXING heatuppan{}".format(k), flush=True)
                            df_mapping[k].loc[t + 1:, 'ObjectTemperature'] = 2
                            if df_mapping[k].iloc[0]['cookable']:
                                df_mapping[k].loc[t + 1:, 'isCooked'] = True

        # Sinks should fill with water
        #####
        for k, v in df_mapping.items():
            if k.startswith('Sink'):
                v['canFillWithLiquid'] = True

                filled_now = False
                filled_list = [False]
                for _, row in item['actions'].iterrows():
                    if row['object_name'] is not None and row['object_name'].startswith('Faucet'):
                        if row['action_name'] == 'ToggleObjectOn':
                            filled_now = True
                        elif row['action_name'] == 'ToggleObjectOff':
                            filled_now = False
                    filled_list.append(filled_now)
                v['isFilledWithLiquid'] = filled_list[:-1]
            if k.startswith('Faucet'):
                # Weird stuff with size!
                v['sizeraw'] = v.iloc[0]['sizeraw']
                v['size'] = v.iloc[0]['size']


        # If there's a pan fail then skip
        if (item['actions']['action_name'] == 'HeatUpPan').any():
            becomes_cooked = False
            for v in df_mapping.values():
                if v['cookable'].any() and len(v['isCooked'].value_counts()) == 2:
                    becomes_cooked = True
            if not becomes_cooked:
                # print("SKIPPING BECAUSE NOTHING BECAME COOKED", flush=True)
                continue

        item['df_mapping'] = df_mapping
        item['main_object_ids'] = main_object_ids
        item['object_id_to_main_ind'] = object_id_to_main_ind

        ########### Separate into "Temporal" and "non-temporal"
        keys = []
        df_rows = []
        for k, v in item['object_id_to_states'].items():
            if len(v) > 0:
                keys.append(k)
                all_vals = list(v.values())
                df_rows.append(random.choice(all_vals))
            # if '0' in v:
            #     keys.append(k)
            #     df_rows.append(v['0'])
        nontemporal_mapping = _fixup_df(pd.DataFrame(df_rows), object_to_size, include_pos=True)
        nontemporal_mapping['index'] = keys
        nontemporal_mapping = nontemporal_mapping[[c for c in nontemporal_mapping.columns if c not in bad_cols]]
        nontemporal_mapping.set_index('index', drop=True, inplace=True)
        yield item, nontemporal_mapping
