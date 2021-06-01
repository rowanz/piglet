import os

import numpy as np
import pandas as pd
import json
from difflib import SequenceMatcher

THOR_OBJECT_TYPES = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'object_counts.csv'),
                                skiprows=[0])['ObjectType'].tolist()

THOR_MATERIALS = [x.split('=')[1] for x in
                  pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'object_counts.csv'),
                              skiprows=[0]).columns.tolist() if x.startswith('salientMaterials=')]

THOR_ACTIONS = ['CloseObject',
                'DirtyObject',
                'EmptyLiquidFromObject',
                'Done',
                'JumpAhead',
                'LookDown',
                'LookUp',
                'MoveAhead',
                'MoveBack',
                'MoveLeft',
                'MoveRight',
                'OpenObject',
                'PickupObject',
                'PutObject',
                'RotateHand',
                'RotateLeft',
                'RotateRight',
                'SliceObject',
                'ThrowObject10',  # Several different magnitudes
                'ThrowObject100',
                'ThrowObject1000',
                'ToggleObjectOff',
                'ToggleObjectOn',
                'HeatUpPan', # hack
                ]

THOR_MATERIALS_TO_IND = {t: i for i, t in enumerate(['__background__'] + THOR_MATERIALS)}
THOR_OBJECT_TYPE_TO_IND = {t: i for i, t in enumerate(['__background__'] + THOR_OBJECT_TYPES)}
THOR_ACTION_TYPE_TO_IND = {t: i for i, t in enumerate(['__background__'] + THOR_ACTIONS)}


def _action_to_type_ind(action):
    # Slight change here....
    if action['action'] == 'ThrowObject':
        return THOR_ACTION_TYPE_TO_IND['ThrowObject{}'.format(int(action['moveMagnitude']))]
    else:
        return THOR_ACTION_TYPE_TO_IND[action['action']]


def _object_to_type_ind(obj_id):
    if obj_id in ('__background__', 'None'):
        return 0
    obj_type = obj_id.split('|')[0]
    return THOR_OBJECT_TYPE_TO_IND[obj_type]


# minval, maxval
MASS_CATEGORIES = [
    (-1e8, 1e-8),  # Massless -- 46.5 %
    (1e-8, 0.04500),
    (0.04500, 0.11000),
    (0.11000, 0.24000),
    (0.24000, 0.62000),
    (0.62000, 1.00000),
    (1.00000, 5.00000),
    (5.00000, float('inf')),
]

# MASS_NAMES = ['[{:.1f}kg,{:.1f}kg]'.format(x,y) for x, y in MASS_CATEGORIES]
# MASS_NAMES[0] = 'massless'
# MASS_NAMES[-1] = 'heaviest'

MASS_NAMES = ['Massless', 'below 0.1 lb', '.1 to .2lb', '.2 to .5lb', '.5 to 1lb', '1 to 2lb', '2 to 10lb', 'over 10lb']
MASS_NAMES_TO_ID = {mn:i for i, mn in enumerate(MASS_NAMES)}



# Size percentiles
# 0.525: Size 0.000005646
# 0.573: Size 0.000056751
# 0.621: Size 0.000057008
# 0.669: Size 0.000069993
# 0.716: Size 0.000072791
# 0.764: Size 0.000075708
# 0.812: Size 0.000079935
# 0.860: Size 0.000089174

SIZE_CATEGORIES = [
 (-1e-08, 3.7942343624308705e-05),
 (3.7942343624308705e-05, 0.0006833796505816281),
 (0.0006833796505816281, 0.0028182819951325655),
 (0.0028182819951325655, 0.010956133715808392),
 (0.010956133715808392, 0.038569377816362015),
 (0.038569377816362015, 0.08704059571027756),
 (0.08704059571027756, 0.18317937850952148),
 (0.18317937850952148, float('inf')),
]
# SIZE_NAMES = ['[{:.4f}m3,{:.4f}m3]'.format(x,y) for x, y in SIZE_CATEGORIES]
# SIZE_NAMES[0] = 'sizeless'
# SIZE_NAMES[-1] = 'biggest'
SIZE_NAMES = ['sizeless', 'tiny', 'small', 'medium', 'medium-plus', 'large', 'extra large', 'extra extra large']
SIZE_NAMES_TO_ID = {sn:i for i, sn in enumerate(SIZE_NAMES)}

TEMP_NAMES = ['Cold', 'RoomTemp', 'Hot']
TEMP_NAMES_TO_ID = {tn:i for i, tn in enumerate(TEMP_NAMES)}

DISTANCE_CATEGORIES = [(-1.0, 0.25),
 (0.25, 0.5),
 (0.5, 0.75),
 (0.75, 1.0),
 (1.0, 1.5),
 (1.5, 2.0),
 (2.0, 2.5),
 (2.5, float('inf')),
]

# DISTANCE_NAMES = ['[{:.1f}m,{:.1f}m]'.format(x,y) for x, y in DISTANCE_CATEGORIES]
# DISTANCE_NAMES[0] = 'closest'
# DISTANCE_NAMES[-1] = 'farthest'
DISTANCE_NAMES = ['below 1ft', '1 to 2ft', '2 to 3ft', '3 to 4ft', '4 to 6ft', '6 to 8 ft', '8 to 10ft', 'over 10ft']
DISTANCE_NAMES_TO_ID = {mn:i for i, mn in enumerate(DISTANCE_NAMES)}

# to generate categories
# N = 7
# bottom = 46
# for i in range(N):
#     lower_perc = bottom + (100.0 - bottom) / N * i
#     upper_perc = bottom + (100.0 - bottom) / N * (i+1)
#     print("({:.5f}, {:.5f}),".format(np.percentile(masses, lower_perc),
#                                     np.percentile(masses, upper_perc),))
def _mass_to_categorical(mass):
    for i, (lval, uval) in enumerate(MASS_CATEGORIES):
        if (mass >= lval) and (mass <= uval):
            return i
    import ipdb
    ipdb.set_trace()
    return 0

def _size_to_categorical(size):
    for i, (lval, uval) in enumerate(SIZE_CATEGORIES):
        if (size >= lval) and (size <= uval):
            return i
    import ipdb
    ipdb.set_trace()
    return 0

def _distance_to_categorical(distance):
    for i, (lval, uval) in enumerate(DISTANCE_CATEGORIES):
        if (distance >= lval) and (distance < uval):
            return i
    import ipdb
    ipdb.set_trace()
    return 0

# [affordance_name, arity, is_object]
THOR_AFFORDANCES = [('ObjectTemperature', 3, False),
                    ('breakable', 1, False),
                    ('canBeUsedUp', 1, False),
                    ('canChangeTempToCold', 1, False),
                    ('canChangeTempToHot', 1, False),
                    ('canFillWithLiquid', 1, False),
                    ('cookable', 1, False),
                    ('dirtyable', 1, False),
                    ('isBroken', 1, False),
                    ('isCooked', 1, False),
                    ('isDirty', 1, False),
                    ('isFilledWithLiquid', 1, False),
                    ('isOpen', 1, False),
                    ('isPickedUp', 1, False),
                    ('isSliced', 1, False),
                    ('isToggled', 1, False),
                    ('isUsedUp', 1, False),
                    ('mass', len(MASS_CATEGORIES), False),
                    ('size', len(SIZE_CATEGORIES), False),
                    ('distance', len(DISTANCE_CATEGORIES), False),
                    ('moveable', 1, False),
                    ('openable', 1, False),
                    ('parentReceptacles', len(THOR_OBJECT_TYPE_TO_IND), True),
                    ('pickupable', 1, False),
                    ('receptacle', 1, False),
                    ('receptacleObjectIds', len(THOR_OBJECT_TYPE_TO_IND), True),
                    ('salientMaterials_Ceramic', 1, False),
                    ('salientMaterials_Fabric', 1, False),
                    ('salientMaterials_Food', 1, False),
                    ('salientMaterials_Glass', 1, False),
                    ('salientMaterials_Leather', 1, False),
                    ('salientMaterials_Metal', 1, False),
                    ('salientMaterials_None', 1, False),
                    ('salientMaterials_Organic', 1, False),
                    ('salientMaterials_Paper', 1, False),
                    ('salientMaterials_Plastic', 1, False),
                    ('salientMaterials_Rubber', 1, False),
                    ('salientMaterials_Soap', 1, False),
                    ('salientMaterials_Sponge', 1, False),
                    ('salientMaterials_Stone', 1, False),
                    ('salientMaterials_Wax', 1, False),
                    ('salientMaterials_Wood', 1, False),
                    ('sliceable', 1, False),
                    ('toggleable', 1, False)]

def _fixup_df(df, object_to_size, include_pos=False, agent_states=None):
    """
    helper fn
    :param df:
    :param object_to_size:
    :param include_pos:
    :return:
    """

    # These things don't change or are not interesting
    for k in ['objectType', 'objectId', 'visible', 'isMoving']:
        del df[k]
    if not include_pos:
        for k in ['pos3d']:
            del df[k]

    df['ObjectTemperature'] = df['ObjectTemperature'].apply(lambda x: {'Cold': 0, 'RoomTemp': 1, 'Hot': 2}[x])

    # ONEHOT
    def _make_onehot(vs, n):
        oh = np.zeros(n, dtype=np.bool)
        oh[vs] = True
        return oh

    salient_materials = np.stack(df['salientMaterials'].apply(
        lambda x: _make_onehot([THOR_MATERIALS_TO_IND[y] for y in x], n=len(THOR_MATERIALS_TO_IND))).values, 0)
    del df['salientMaterials']
    for name, i in sorted(THOR_MATERIALS_TO_IND.items(), key=lambda x: x[1]):
        if name != '__background__':
            df[f'salientMaterials_{name}'] = salient_materials[:, i]

    def _get_smallest_item_type(itemlist, largest=False):
        if len(itemlist) == 0:
            return 0
        item_and_size = sorted([(item_id, object_to_size[item_id]) for item_id in itemlist], key=lambda x: x[1])
        if largest:
            item_and_size = item_and_size[::-1]
        return _object_to_type_ind(item_and_size[0][0])

    ml = df['parentReceptacles'].apply(len).max()
    df['parentReceptacles'] = df['parentReceptacles'].apply(_get_smallest_item_type)
    # df['parentReceptacles'] = df['parentReceptacles'].apply(lambda x:
    #                                                         set([_object_to_type_ind(z) for z in x]))
    # df['receptacleObjectIds'] = df['receptacleObjectIds'].apply(lambda x:
    #                                                         set([_object_to_type_ind(z) for z in x]))

    # If the names are the same -- report things that change. Otherwise, pick something at random
    same_col_names = len(df['name'].value_counts()) == 1
    for col in ['receptacleObjectIds']:
        vs = []
        if same_col_names:
            last_vals = []
            for new_vals in df[col].tolist():

                # Make sure last_vals is a subset of new_vals
                last_vals = [z for z in last_vals if z in new_vals]

                if len(new_vals) == 0:
                    vs.append(0)
                    last_vals = []
                elif len(new_vals) == 1:
                    vs.append(_object_to_type_ind(new_vals[0]))
                    last_vals = [new_vals[0]]
                else:
                    # Multiple things.
                    # 1. see if anything new added
                    newly_added = [z for z in new_vals if z not in last_vals]

                    if len(newly_added) == 0:
                        vs.append(_object_to_type_ind(last_vals[0]))
                    else:
                        # Completely redo last_vals
                        newly_added = sorted(newly_added, key=lambda x: object_to_size[x])
                        if col == 'receptacleObjectIds':
                            # 'largest' mode
                            last_vals = last_vals[::-1]
                        for z in new_vals:
                            if z not in newly_added:
                                newly_added.append(z)
                        last_vals = newly_added
                        vs.append(_object_to_type_ind(last_vals[0]))

        else:
            for new_vals in df[col].tolist():
                if len(new_vals) == 0:
                    vs.append(0)
                else:
                    idx = int(np.random.choice(len(new_vals)))
                    vs.append(_object_to_type_ind(new_vals[idx]))
        df[col] = vs

    del df['name']

    df['massraw'] = df['mass']
    df['mass'] = df['massraw'].apply(_mass_to_categorical)

    df['sizeraw'] = df['pos3d'].apply(lambda x: np.prod(x[3]+0.01))
    df['size'] = df['sizeraw'].apply(_size_to_categorical)

    if agent_states is not None:
        # dr = df['distance'].apply(_distance_to_categorical)
        # if (0 in dr.unique().tolist()) and (7 in dr.unique().tolist()):
        #     import ipdb
        #     ipdb.set_trace()
        df['distanceraw'] = [compute_distance_raw(as_x, dx) for as_x, dx in zip(agent_states, df['pos3d'])]
    else:
        # This will probably not give accurate results because I messed up updating the distances...
        # BUT we wont use this i think
        df['distanceraw'] = df['distance']

    df['distance'] = df['distanceraw'].apply(_distance_to_categorical)

    # Sanity check affordances
    affordances_used = {n: max(v, 2) for n, v, _ in THOR_AFFORDANCES}
    for col in df.columns:
        if col in ['pos3d', 'sizeraw', 'massraw', 'distanceraw']:
            continue
        col_v = df[col].values.astype(np.int64)
        if col not in affordances_used:
            raise ValueError(f"{col} not in THOR_AFFORDANCES")

        items_not_in_vocab = np.setdiff1d(col_v, np.arange(affordances_used.pop(col)))
        if items_not_in_vocab.size > 0:
            raise ValueError(f"These values are not in the vocab {items_not_in_vocab}")

    if len(affordances_used) != 0:
        raise ValueError(f"These affordances were never used: {affordances_used}")

    return df

def _object_to_statechange_df(states_by_t_list, num_frames, object_to_size, include_pos=False, agent_states=None):
    """
    :param object_id_to_stateses: A list, in ascending order of priority, of mappings {t: states}.

    e.g. we can get it for something like


    obj_list = ['Desk|-00.50|+00.01|-01.47'] + [aid for aid, oid in
                    item['alias_object_id_to_old_object_id'].items() if oid == 'Desk|-00.50|+00.01|-01.47']
    states_by_t_list = [item['object_id_to_states'][k] for k in obj_list]

    :param num_frames: How many frames
    :param object_to_size: Global object to size
    :return: a DF
    """
    df = []
    for t in range(num_frames):
        # Add a dummy None which will get overwritten
        df.append(None)
        for states_by_t in states_by_t_list:
            if str(t) in states_by_t:
                df[t] = states_by_t[str(t)]
        if df[t] is None:
            assert t > 0
            df[t] = df[t - 1]
    df = pd.DataFrame(df)
    return _fixup_df(df, object_to_size, include_pos=include_pos, agent_states=agent_states)


def load_instance_attribute_weights(max_weight=1000.0):
    """
    :param max_weight:
    :return:
    """
    # with open(os.path.join(os.path.dirname(__file__), 'traj_change_counts.json'), 'r') as f:
    #     change_counts = json.load(f)['oa_change']

    # Count change vs. no change for each attr
    counts_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'instance_df_counts_v2.csv'))

    counts_all = {aname: np.ones((len(THOR_OBJECT_TYPE_TO_IND), 2), dtype=np.int64)
                   for aname, _, _ in THOR_AFFORDANCES}
    for _, row in counts_df.iterrows():
        counts_all[row['attr_name']][_object_to_type_ind(row['object_name']), 0] += row['nochange_count']
        counts_all[row['attr_name']][_object_to_type_ind(row['object_name']), 1] += row['change_count']

    weights_map = {}
    for attr_name, cts in counts_all.items():
        # cts[:,0] is nochange and cts[:,1] is didchange
        change_prob = cts[:,1].astype(np.float32) / cts.sum(1).astype(np.float32)

        weights = np.minimum(1.0 / (1e-8 + change_prob), max_weight)
        weights_map[attr_name] = weights
    return weights_map

#################### DEMO UTILS
def instance_to_np_demo(instance):
    """
    This is an incomplete inverse of numpy_to_instance_states because it doesn't handle both pre and post conditions.
    :param instance:
    :return:
    """
    from model.interact.dataloader import names_and_arities
    # PREPARE THE INSTANCE
    obj_states = []
    for name_i, arity_i in names_and_arities:
        vals = [instance[f'{name_i}0'], instance[f'{name_i}1']]
        if arity_i == 1:
            pass
            # obj_states.append(np.array(vals).astype(np.int32))
        elif arity_i == 126:
            vals = [THOR_OBJECT_TYPE_TO_IND.get(v, 0) for v in vals]
            # obj_states.append(np.array(vs, dtype=np.int32))
        elif name_i == 'ObjectTemperature':
            vals = [TEMP_NAMES_TO_ID.get(v, 1) for v in vals]
        elif name_i == 'mass':
            vals = [MASS_NAMES_TO_ID.get(v, 0) for v in vals]
        elif name_i == 'size':
            vals = [SIZE_NAMES_TO_ID.get(v, 0) for v in vals]
        vals = np.array(vals, dtype=np.int32)
        obj_states.append(vals)
    obj_states = np.stack(obj_states, 1)

    object_types = np.array([THOR_OBJECT_TYPE_TO_IND.get(instance[f'ObjectName{i}'], 0) for i in range(2)])
    is_valid = object_types != 0
    action_id = np.array(THOR_ACTION_TYPE_TO_IND.get(instance['Action0']), dtype=np.int32)

    action_args_raw = [instance['ActionObject0'], instance['ActionReceptacle0']]
    action_args = np.array([{'None': 0, 'Object1': 1, 'Object2': 2}.get(v, 0) for v in action_args_raw],
                           dtype=np.int32)
    features_np = {
        'objects/object_types': object_types[[0, 0, 1, 1]],
        'objects/is_valid': is_valid.astype(np.int32)[[0, 0, 1, 1]],
        'objects/object_states': obj_states[[0, 0, 1, 1]],
        'actions/action_args': action_args,
        'actions/action_id': action_id,
    }
    return features_np


def numpy_to_instance_states(object_types, object_states):
    """
    :param object_types: [num_objects]
    :param object_states: [num_objects, num_affordances]
    :return:
    """
    fields = {}
    from model.interact.dataloader import names_and_arities
    object_list = ['None'] + THOR_OBJECT_TYPES
    for obj_n in range(2):
        # Object states

        fields[f'ObjectName{obj_n}'] = object_list[object_types[obj_n]]
        for i, (name_i, arity_i) in enumerate(names_and_arities):
            v_i = object_states[obj_n, i]

            if arity_i == 1:
                val_i = bool(v_i)
            elif arity_i == 126:
                val_i = object_list[v_i]
            elif name_i in ('ObjectTemperature', 'mass', 'size', 'distance'):
                val_i = int(v_i)
            else:
                raise ValueError("Invalid option")
            fields[f'{name_i}{obj_n}'] = val_i
    return fields


def numpy_to_instance(features_np, add_post_conditions=False):
    """
    Get precondition and maybe postconditions.
    :param features_np:
    :return:
    """
    inds_to_use = np.array([0, 2], dtype=np.int32)

    # ACTIONS
    action_list = ['CompareObjects'] + THOR_ACTIONS

    action_args = [['None', 'Object1', 'Object2'][x] for x in features_np['actions/action_args'].tolist()]

    fields = {
        'Action0': action_list[int(features_np['actions/action_id'])],
        'ActionObject0': action_args[0],
        'ActionReceptacle0': action_args[1],
    }
    fields.update(**numpy_to_instance_states(features_np['objects/object_types'][inds_to_use],
                                             features_np['objects/object_states'][inds_to_use],
                  ))
    if not add_post_conditions:
        return fields

    post_inds_to_use = np.array([1, 3], dtype=np.int32)
    post_fields = numpy_to_instance_states(features_np['objects/object_types'][post_inds_to_use],
                                           features_np['objects/object_states'][post_inds_to_use],
                  )
    return fields, post_fields

def instance_to_tfrecord(item):
    """
    Move item back to tfrecord. this completes this really complicated circuit I have


    tfrecord -> numpy (_decode_record)

    numpy -> json (numpy_to_instance, used for getting all the data)

    json -> tfrecord (this thing)


    :param item:
    :return:
    """
    from tfrecord.tfrecord_utils import _convert_image_seq_to_jpgstring, bytes_feature, float_feature, \
        float_list_feature, int64_list_feature, int64_feature

    # Action ID
    action_type_to_ind = {k: v for k, v in THOR_ACTION_TYPE_TO_IND.items()}
    action_type_to_ind['CompareObjects'] = 0
    action_id = action_type_to_ind[item['pre']['Action0']]

    # Action Args
    action_args = [{'None': 0, 'Object1': 1, 'Object2': 2}.get(v, 0) for v in [
        item['pre']['ActionObject0'], item['pre']['ActionReceptacle0']]]

    tfrecord = {'meta': bytes_feature(json.dumps(item['meta']).encode('utf-8')),
                # 'agent_state': float_list_feature([0]*6),
                'actions/action_id': int64_feature(action_id),
                'actions/action_args': int64_list_feature(action_args),
                # 'actions/action_success': int64_feature(1),
                # 'comparison_labels': int64_list_feature([0, 0]),
                }

    objects = [{k.rstrip('0'): v for k,v  in item[k].items() if k.endswith('0')} for k in ['pre', 'post']]
    if item['pre']['ObjectName1'] != 'None':
        objects += [{k.rstrip('1'): v for k,v  in item[k].items() if k.endswith('1')} for k in ['pre', 'post']]

    #####
    tfrecord['objects/object_types'] = int64_list_feature([_object_to_type_ind(o['ObjectName']) for o in objects])
    for affordance_name, arity_i, _ in THOR_AFFORDANCES:
        if affordance_name in ['canChangeTempToCold', 'canChangeTempToHot', 'salientMaterials_None',
                               'canBeUsedUp', 'canFillWithLiquid']:
            continue

        if arity_i == 1:
            val_i = [int(o[affordance_name]) for o in objects]
        elif arity_i == 126:
            val_i = [_object_to_type_ind(o[affordance_name]) for o in objects]
        else:
            val_i = [o[affordance_name] for o in objects]
        tfrecord[f'objects/{affordance_name}'] = int64_list_feature(val_i)

    tfrecord['objects/distance'] = int64_list_feature([o['distance'] for o in objects])
    return tfrecord


def compute_distance_raw(agent_pos, object_pos):
    """
    :param agent_pos: (x, y, z) and then rotation
    :param object_pos: a (4,3) thing. we want the 0rd row (pos)
    :return:
    """
    op_center = object_pos[0]
    dist = np.sqrt(np.square(op_center - agent_pos[:3]).sum())
    return dist


AFFORDANCES_IN_ORDER = [('ObjectName', 126, True)] + sorted(THOR_AFFORDANCES, key=lambda x: (-x[1], x[0]))

def format_object_as_text(field, object_id, convert_ids=True):
    """
    Given an instance (ie from "numpy_to_instance") convert it into text
    :param field:
    :param object_id: which object
    :param convert_ids: Whether to convert numbers like "distance: 3" into the names
    :return: text representation
    """
    if field[f'ObjectName{object_id}'] == 'None':
        txt = ['None']
    else:
        txt = []
        for aname, a_arity, a_isobject in AFFORDANCES_IN_ORDER:
            v = field.get(f'{aname}{object_id}', None)
            if v is None:
                continue

            if convert_ids:
                if aname == 'ObjectTemperature':
                    v = TEMP_NAMES[v]
                elif aname == 'mass':
                    v = MASS_NAMES[v]
                elif aname == 'distance':
                    v = DISTANCE_NAMES[v]
                elif aname == 'size':
                    v = SIZE_NAMES[v]

            if aname == 'ObjectTemperature':
                aname = 'Temp'
            if aname == 'receptacleObjectIds':
                aname = 'containedObjects'
            if aname.startswith('is'):
                aname = aname[2:]
            if aname.startswith('salientMaterials_'):
                continue
                # aname = aname.split('_')[1]
            txt.append(f'{aname}: {v}')
        # Fix salient materials
        materials = [aname.split('_')[1] for aname, a_arity, a_isobject in AFFORDANCES_IN_ORDER
                     if aname.startswith('salientMaterials_') and field.get(f'{aname}{object_id}', False)]
        if len(materials) == 0:
            materials.append('None')
        txt.append('Materials: {}'.format(' '.join(materials)))
    txt_joined = ', '.join(txt)
    txt_joined = f'({txt_joined})'
    return txt_joined

def format_action_as_text(field):
    """
    Given an instance (ie from "numpy_to_instance") convert it into text
    :param field:
    :param object_id: which object
    :param convert_ids: Whether to convert numbers like "distance: 3" into the names
    :return: text representation
    """
    txt = ['Action: {}'.format(field['Action0'])]
    if field['ActionObject0'] != 'None':
        txt.append('Object: {}'.format(field['ActionObject0']))
    if field['ActionReceptacle0'] != 'None':
        txt.append('Receptacle: {}'.format(field['ActionReceptacle0']))
    txt_joined = ', '.join(txt)
    txt_joined = f'({txt_joined})'
    return txt_joined


# UNPARSE
def text_to_instance_states(txt_rep):
    """
    Invert T5 representation
    :param txt_rep:
    :return:
    """
    def _get_closest_ind(query, keys):
        """
        :param query: string
        :param keys: list of keys
        :return:
        """
        query_fixed = query.lower()
        keys_fixed = [v.lower() for v in keys]
        if query_fixed in keys_fixed:
            ind = int(np.where([q == query_fixed for q in keys_fixed])[0][0])
        else:
            diffs = np.array([SequenceMatcher(None, query_fixed, k2).ratio() for k2 in keys_fixed])
            ind = int(np.argmax(diffs))
            print("Not everything was close, doing {} {} -> {}".format(query, keys_fixed, keys_fixed[ind]), flush=True)
        return ind

    def _get_closest_option(raw_dict, k, valid_choices):
        """
        Get closest option
        :param raw_dict: where the data is located, was generated by T5
        :param k: key we are searching for
        :param valid_choices:
        :return:
        """
        all_keys = sorted([k2 for k2 in raw_dict.keys()])
        our_v = raw_dict[all_keys[_get_closest_ind(k, all_keys)]]
        ind = _get_closest_ind(our_v, valid_choices)
        return ind

    sub_reps = [a.strip(')(').split(', ') for a in txt_rep.split(') (')]
    sub_reps = [x for x in sub_reps if len(x) > 1] # Skip "none" (empty)
    fields = {}
    for obj_n, sub_rep_n in enumerate(sub_reps):
        obj_rep_n_dict_raw = {}
        for l in sub_rep_n:
            k, v = l.split(': ', 1)
            obj_rep_n_dict_raw[k] = v

        # Handle "materials"
        materials = obj_rep_n_dict_raw.get('materials', '').split(' ')
        for aname, a_arity, a_isobject in AFFORDANCES_IN_ORDER:
            if aname.startswith('salientMaterials_'):
                material = aname.split('_')[1]
                obj_rep_n_dict_raw[aname] = str(material.lower() in materials)
        # fields[f'ObjectName{obj_n}'] = object_list[object_types[obj_n]]

        object_list = ['None'] + THOR_OBJECT_TYPES
        fields[f'ObjectName{obj_n}'] = object_list[_get_closest_option(obj_rep_n_dict_raw, 'ObjectName', object_list)]

        for i, (name_i, arity_i, _) in enumerate(THOR_AFFORDANCES):
            if name_i in ['canChangeTempToCold', 'canChangeTempToHot', 'salientMaterials_None',
                                   'canBeUsedUp', 'canFillWithLiquid']:
                continue

            aname = name_i
            if aname == 'ObjectTemperature':
                aname = 'Temp'
            if aname == 'receptacleObjectIds':
                aname = 'containedObjects'
            if aname.startswith('is'):
                aname = aname[2:]

            if arity_i == 1:
                vals = ['True', 'False']
            elif arity_i == 126:
                vals = object_list
            elif name_i == 'ObjectTemperature':
                vals = ['Cold', 'RoomTemp', 'Hot']
            elif name_i == 'mass':
                vals = MASS_NAMES
            elif name_i == 'size':
                vals = SIZE_NAMES
            elif name_i == 'distance':
                vals = DISTANCE_NAMES
            else:
                raise ValueError("Invalid option")

            val_i = _get_closest_option(obj_rep_n_dict_raw, aname, vals)

            if arity_i == 1:
                val_i = bool(val_i == 0)
            elif arity_i == 126:
                val_i = object_list[val_i]

            fields[f'{name_i}{obj_n}'] = val_i
    return fields