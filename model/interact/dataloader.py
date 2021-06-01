import sys

sys.path.append('../../')
import time
from model import model_utils
from model.model_utils import pad_to_fixed_size as p2fsz
from model.model_utils import randomize_onehot_given_budget, get_shape_list
import tensorflow as tf
from model.neat_config import NeatConfig
from data.thor_constants import THOR_AFFORDANCES

slim_example_decoder = tf.contrib.slim.tfexample_decoder

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 640

_keys_to_features = {
    'actions/action_args': tf.io.VarLenFeature(tf.int64),
    'actions/action_id': tf.io.FixedLenFeature((), tf.int64, 1),
    'actions/action_success': tf.io.FixedLenFeature((), tf.int64, 1),
    'agent_state': tf.io.VarLenFeature(tf.float32),
    'comparison_labels': tf.io.VarLenFeature(tf.int64),
    'meta': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'objects/ObjectTemperature': tf.io.VarLenFeature(tf.int64),
    'objects/breakable': tf.io.VarLenFeature(tf.int64),
    'objects/canBeUsedUp': tf.io.VarLenFeature(tf.int64),
    'objects/canFillWithLiquid': tf.io.VarLenFeature(tf.int64),
    'objects/cookable': tf.io.VarLenFeature(tf.int64),
    'objects/dirtyable': tf.io.VarLenFeature(tf.int64),
    'objects/distance': tf.io.VarLenFeature(tf.int64),
    'objects/isBroken': tf.io.VarLenFeature(tf.int64),
    'objects/isCooked': tf.io.VarLenFeature(tf.int64),
    'objects/isDirty': tf.io.VarLenFeature(tf.int64),
    'objects/isFilledWithLiquid': tf.io.VarLenFeature(tf.int64),
    'objects/isOpen': tf.io.VarLenFeature(tf.int64),
    'objects/isPickedUp': tf.io.VarLenFeature(tf.int64),
    'objects/isSliced': tf.io.VarLenFeature(tf.int64),
    'objects/isToggled': tf.io.VarLenFeature(tf.int64),
    'objects/isUsedUp': tf.io.VarLenFeature(tf.int64),
    'objects/mass': tf.io.VarLenFeature(tf.int64),
    'objects/moveable': tf.io.VarLenFeature(tf.int64),
    'objects/object_types': tf.io.VarLenFeature(tf.int64),
    'objects/openable': tf.io.VarLenFeature(tf.int64),
    'objects/pickupable': tf.io.VarLenFeature(tf.int64),
    'objects/receptacle': tf.io.VarLenFeature(tf.int64),
    'objects/parentReceptacles': tf.io.VarLenFeature(tf.int64),
    'objects/receptacleObjectIds': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Ceramic': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Fabric': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Food': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Glass': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Leather': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Metal': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Organic': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Paper': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Plastic': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Rubber': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Soap': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Sponge': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Stone': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Wax': tf.io.VarLenFeature(tf.int64),
    'objects/salientMaterials_Wood': tf.io.VarLenFeature(tf.int64),
    'objects/size': tf.io.VarLenFeature(tf.int64),
    'objects/sliceable': tf.io.VarLenFeature(tf.int64),
    'objects/toggleable': tf.io.VarLenFeature(tf.int64),
    'frames/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'frames/format': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'frames/height': tf.io.FixedLenFeature((), tf.int64, 1),
    'frames/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'frames/num_frames': tf.io.FixedLenFeature((), tf.int64, 1),
    'frames/width': tf.io.FixedLenFeature((), tf.int64, 1),
    'is_real_example': tf.io.FixedLenFeature((), tf.int64, 0),    # Only needed for the turk stuff
    'ids/pre_act2post': tf.io.VarLenFeature(tf.int64),                   # Only needed for the turk stuff
    'ids/post_act2pre': tf.io.VarLenFeature(tf.int64),  # Only needed for the turk stuff
    'ids/pre': tf.io.VarLenFeature(tf.int64),  # Only needed for the turk stuff
    'ids/post': tf.io.VarLenFeature(tf.int64),  # Only needed for the turk stuff
    'ids/_extrasignal': tf.io.VarLenFeature(tf.int64),  # Only needed for the turk stuff
}

# all_salient_materials = sorted([x for x in _keys_to_features.keys() if x.startswith('objects/salientMaterials_')])

names_and_arities = []
for name_i, arity_i, _ in THOR_AFFORDANCES:
    if f'objects/{name_i}' in _keys_to_features.keys():
        if not name_i in ('canFillWithLiquid', 'canBeUsedUp'):
            names_and_arities.append((name_i, arity_i))
names_and_arities = sorted(names_and_arities, key=lambda x: (-x[1], x[0]))

# Handle the image things separately
_items_to_handlers = {k: (slim_example_decoder.Tensor(k)) for k in _keys_to_features.keys()}

_items_to_handlers['frames'] = slim_example_decoder.Image(
    image_key=f'frames/encoded',
    format_key=f'frames/format',
    channels=3)


def _decode_record(record, use_vision=False, use_language=True):
    """Decodes serialized tensorflow example and returns a tensor dictionary. See keys_to_features for arguments
    """
    serialized_example = tf.reshape(record, shape=[])

    if use_vision:
        i2h = _items_to_handlers
    else:
        i2h = {k: v for k, v in _items_to_handlers.items() if not k.startswith('frames')}

    decoder = slim_example_decoder.TFExampleDecoder(_keys_to_features, i2h)
    keys = sorted(decoder.list_items())

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))

    if use_vision:
        tensor_dict['frames'] = tf.reshape(tensor_dict['frames'], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    if not use_language:
        for k in sorted(tensor_dict.keys()):
            if k.startsiwth("ids/"):
                tf.logging.info(f"Popping {k} bc no language")
                del tensor_dict[k]

    for k in tensor_dict:
        if tensor_dict[k].dtype == tf.int64:
            tensor_dict[k] = tf.cast(tensor_dict[k], tf.int32)
    return tensor_dict


def _dataset_parser(value, is_training=True, NUM_OBJECTS=4, max_lang_seq_length=64,
                    use_vision=False, use_language=True):
    """

    :param value: TFRecord to decode
    :param is_training:
    :return:
    """
    with tf.name_scope('parser'):
        data = _decode_record(value, use_vision=use_vision, use_language=use_language)

        # new_height, new_width, inverse_scale_factor, orig_height, orig_width
        features = {
            'meta': model_utils.encode_string(data['meta'], 512),
            'actions/action_id': data['actions/action_id'],
            'actions/action_success': data['actions/action_success'],
            'labels/comparison_labels': p2fsz(data['comparison_labels'], pad_value=0, output_shape=[2], axis=0,
                                              truncate=False),
            'actions/action_args': p2fsz(data['actions/action_args'], pad_value=0, output_shape=[NUM_OBJECTS // 2],
                                         axis=0, truncate=True),
            'agent_state': p2fsz(data['agent_state'], pad_value=0, output_shape=[6], axis=0, truncate=True),
            'is_real_example': data['is_real_example'],
        }
        for k, v in data.items():
            if k.startswith('ids/'):
                features[k] = p2fsz(v, pad_value=0, output_shape=[max_lang_seq_length], axis=0,
                               truncate=True, name=f'pad_{k}')


        if use_vision:
            frames = tf.image.convert_image_dtype(data[f'frames'], dtype=tf.float32)
            frames = model_utils.normalize_image(frames)


            features['frames/frames'] = p2fsz(frames, pad_value=0.0, output_shape=[2, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                                       axis=0, truncate=True, name='pad_frames')
            features['frames/num_frames'] = data['frames/num_frames']

        # With some probability switch the action args
        # are_two_arguments = tf.cast(tf.reduce_all(tf.greater(features['actions/action_args'], 0)), dtype=tf.int32)
        # change_aargs = tf.random.categorical(tf.math.log([[0.75, 0.25]]), dtype=tf.int32, num_samples=1)[
        #                    0, 0] * are_two_arguments
        # features['actions/action_args'] = features['actions/action_args'] * (
        #             1 - change_aargs) + change_aargs * tf.gather(
        #     features['actions/action_args'], [1, 0])
        # features['labels/change_aargs'] = change_aargs + are_two_arguments

        # Object states -------------------------------------------------
        features['objects/object_types'] = p2fsz(data['objects/object_types'], pad_value=0, output_shape=[NUM_OBJECTS],
                                                 axis=0,
                                                 truncate=True)
        features['objects/is_valid'] = tf.cast(tf.greater(features['objects/object_types'], 0), dtype=tf.int32)

        # Do nothing, switch pre-post conditions, randomize the VALS
        # Create a new matrix
        # names_and_arities = []
        # for name_i, arity_i, _ in THOR_AFFORDANCES:
        #     if f'objects/{name_i}' in data:
        #         names_and_arities.append((name_i, arity_i))

        # [o1   o1    o2   o2  ]
        # [pre, post, pre, post]
        obj_states = []
        for name_i, arity_i in names_and_arities:
            tf.logging.info(f'{name_i:>40s} {arity_i}')

            with tf.name_scope(f'convert_{name_i}'):
                raw = p2fsz(data[f'objects/{name_i}'], pad_value=0, output_shape=[NUM_OBJECTS], axis=0, truncate=True)
                obj_states.append(raw)

        # [num_objs, num_obj_states]
        obj_states = tf.stack(obj_states, 1)

        features['objects/object_states'] = obj_states

        tf.logging.info("Feature summary!")
        for k, v in sorted(features.items(), key=lambda x: x[0]):
            tf.logging.info('{}: {}'.format(k, get_shape_list(v)))

        return features, {}


def input_fn_builder(config: NeatConfig, is_training=True):
    input_file = config.data['train_file'] if is_training else config.data['val_file']

    def input_fn(params):

        # this is a reserved term
        batch_size = params['batch_size']
        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1
        tf.logging.info("Current host = {} num hosts = {}".format(current_host, num_hosts))

        if num_hosts == 1:
            dataset = tf.data.Dataset.list_files(
                input_file, shuffle=is_training,
                seed=tf.compat.v1.random.set_random_seed(int(time.time() * 1e9)))
        else:
            # For multi-host training, we want each hosts to always process the same
            # subset of files.  Each host only sees a subset of the entire dataset
            assert is_training
            dataset = tf.data.Dataset.list_files(input_file, shuffle=False)
            dataset = dataset.shard(num_shards=num_hosts, index=current_host)

        if is_training:
            dataset = dataset.repeat()

        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                lambda file_name: tf.data.TFRecordDataset(file_name).prefetch(1),
                cycle_length=64,
                sloppy=is_training))

        if is_training:
            dataset = dataset.shuffle(buffer_size=config.data.get('buffer_size', 1000))

        dataset = dataset.map(lambda x: _dataset_parser(x, is_training=is_training,
                                                        max_lang_seq_length=config.data.get('max_lang_seq_length',64)),
                              num_parallel_calls=64)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    return input_fn