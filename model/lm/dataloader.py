import tensorflow as tf

from model.model_utils import pad_to_fixed_size as p2fsz
from model.model_utils import get_shape_list, binomial_sample
import time

slim_example_decoder = tf.contrib.slim.tfexample_decoder

###################################
# Data loading stuff v2
keys_to_features = {
    'input_ids': tf.io.VarLenFeature(tf.int64),
    'concept_ids': tf.io.VarLenFeature(tf.int64),
}
items_to_handlers = {k: (slim_example_decoder.Tensor(k)) for k in keys_to_features.keys()}


def _decode_record(record):
    """Decodes serialized tensorflow example and returns a tensor dictionary. See keys_to_features for arguments
    """
    serialized_example = tf.reshape(record, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    keys = sorted(decoder.list_items())

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    return tensor_dict


def _rowan_mask_v3(bidirectional_part, vocab_size=50270, bert_probs=(0.865, 0.12, 0.015), mask_token=50257,
                   dont_mask_these_tokens=(50265, 50266, 50269, 0),
                   ):
    """
    Rowan's mask function V3. It's just spanbert but I'm not masking out some special tokens.
    :param bidirectional_part: bidirectional_part: [num_tokens] int32 tensor to mask.
    :param vocab_size:  vocab size
    :param bert_probs: [no change prob, masktoken prob, randomtoken prob].
    :param mask_token: I'm using encoder.begin_domain rn
    :param dont_mask_these_tokens: [encoder.begin_article, encoder.end_article, encoder.reset_context, encoder.padding]
    :return:
    """
    bidirectional_size = get_shape_list(bidirectional_part, expected_rank=1)[0]

    do_nothing_prob, mask_prob, random_change_prob = bert_probs

    tf.logging.info(
        "Masking with probabilities:\nDONOTHING: {:.3f}\nMASK:       {:.3f}\nRANDOM:     {:.3f}".format(do_nothing_prob,
                                                                                                        mask_prob,
                                                                                                        random_change_prob))

    with tf.control_dependencies([tf.assert_greater_equal(bidirectional_size, 1)]):
        # sample from a binomial.
        max_budget = binomial_sample(bidirectional_size, p=mask_prob + random_change_prob)

    dontmask = tf.constant(dont_mask_these_tokens, dtype=tf.int32)

    def _body(masked_bidirectional_part, things_to_change):
        budget = max_budget - tf.reduce_sum(tf.cast(tf.not_equal(things_to_change, 0), dtype=tf.int32))
        budget = tf.maximum(budget, 0)

        # First sample the span's length from a geometric distribution
        # Probability of success at 1 <= i <= 8 is (1-p)^{i-1}p
        # Let's have expected length to be 3, which gives us a maxlen of 8
        maxlen = 8
        p = 0.25
        weights = [p * (1. - p) ** i for i in range(maxlen)]
        span_length = tf.reshape(tf.random.categorical(tf.math.log([weights]), dtype=tf.int32, num_samples=1) + 1,
                                 [])
        span_length = tf.minimum(span_length, budget)

        # Maxval might be out of bounds here but we'll double check it later before we actually mask
        start_ind = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=tf.maximum(bidirectional_size - span_length, 2),
                                      dtype=tf.int32)

        our_choice = tf.reshape(
            tf.random.categorical(tf.math.log([[mask_prob, random_change_prob]]), dtype=tf.int32, num_samples=1),
            [])
        possibilities = tf.stack([
            tf.fill([bidirectional_size], mask_token),
            tf.random.uniform(shape=[bidirectional_size],
                              minval=0,
                              maxval=vocab_size - 1,
                              dtype=tf.int32),
        ], 1)
        replacement = tf.gather(possibilities, our_choice, axis=1)

        # Mask out ALL THE TOKENS that overlap between start and end.
        actually_change = tf.math.logical_and(
            tf.math.greater_equal(tf.range(bidirectional_size), start_ind),
            tf.math.less(tf.range(bidirectional_size), start_ind + span_length),
        )

        actually_change = tf.math.logical_and(
            actually_change,
            tf.math.logical_not(
                tf.reduce_any(tf.math.equal(masked_bidirectional_part[:, None], dontmask[None]), axis=1)),
        )

        mbp2 = tf.where(
            actually_change,
            replacement,
            masked_bidirectional_part,
        )

        ttc2 = tf.where(
            actually_change,
            tf.fill([bidirectional_size], our_choice + 1),  # Fill 1 if changed and 0 otherwise
            things_to_change
        )
        return [mbp2, ttc2]

    def _cond(masked_bidirectional_part, things_to_change):
        budget = max_budget - tf.reduce_sum(tf.cast(tf.not_equal(things_to_change, 0), dtype=tf.int32))
        budget = tf.maximum(budget, 0)
        return tf.math.greater(budget, 0)

    things_to_change = tf.zeros(bidirectional_size, dtype=tf.int32)
    masked_bidirectional_part = tf.identity(bidirectional_part)
    masked_bidirectional_part, things_to_change = tf.while_loop(
        cond=_cond,
        body=_body,
        maximum_iterations=10,
        loop_vars=[masked_bidirectional_part, things_to_change],
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])],
        back_prop=False,
    )

    lm_weights = tf.gather(tf.constant([random_change_prob / do_nothing_prob, 1.0, 1.0]), things_to_change)
    return masked_bidirectional_part, lm_weights


def combine_bidirectional_and_unidirectional(masked_inputs, gt_inputs,
                                             left_to_right_prob=0.333,
                                             pure_bidirectional_prob=0.333,
                                             mix_prob=0.333,
                                             reset_ctx_token=50269,
                                             switch_mode_token=50258,
                                             ):
    """
    Given masked inputs, we might switch up the objectives so that we can do multiple things
    :param masked_inputs:  [T]
    :param gt_inputs       [T]
    :return: our choice (0,1,2)
             inputs
             bidirectional size
    """
    T = get_shape_list(gt_inputs, 1)[0]
    tf.logging.info("Combining with pure bidirectional {:.3f} L2R prob {:.3f} mix {:.3f}".format(
        pure_bidirectional_prob, left_to_right_prob, mix_prob
    ))
    our_choice = tf.reshape(
        tf.random.categorical(tf.math.log([[pure_bidirectional_prob, left_to_right_prob, mix_prob]]), dtype=tf.int32, num_samples=1),
        [])
    #
    # # If purely bidirectional then "masked inputs" is the context and "gt_inputs" is the target
    # # then we use this mask
    # pure_bidirectional_mask = tf.ones(T, dtype=tf.int32)
    #
    # # If left to right then
    left_to_right_inputs = tf.concat([
                tf.fill([1], switch_mode_token),
                gt_inputs[:-1],
            ], axis=0)
    # left_to_right_bidirectional_mask = tf.zeros(T, dtype=tf.int32)
    # left_to_right_weights = tf.ones(T, dtype=tf.float32)
    #
    # # If a mix, try to cutoff @ resetctx half the time
    # # mix_bidirectional_size = tf.random.uniform(shape=[], minval=1, maxval=T, dtype=tf.int32)
    is_reset_ctx = tf.cast(tf.equal(gt_inputs, reset_ctx_token), dtype=tf.float32)
    mix_bidirectional_size_prob = 1.0 + is_reset_ctx * tf.cast(T, dtype=tf.float32)/(tf.reduce_sum(is_reset_ctx) + 1e-5)
    mix_bidirectional_size = tf.reshape(tf.random.categorical(tf.math.log(mix_bidirectional_size_prob[None]), dtype=tf.int32, num_samples=1), []) + 1

    mix_inputs = tf.concat([masked_inputs[:mix_bidirectional_size],
                            tf.fill([1], switch_mode_token),
                            gt_inputs[mix_bidirectional_size:-1]
                            ], axis=0)
    # mix_bidirectional_mask = tf.cast(tf.less(tf.range(T, dtype=tf.int32), mix_bidirectional_size), dtype=tf.int32)
    # mix_weights = tf.concat([target_weights[:mix_bidirectional_size], tf.fill([T-mix_bidirectional_size], 1.0)], axis=0)

    # Python scopes are weird
    options = [
        (masked_inputs, T),
        (left_to_right_inputs, tf.constant(0, dtype=tf.int32)),
        (mix_inputs, mix_bidirectional_size),
    ]
    inputs_to_use = tf.switch_case(our_choice, branch_fns=[lambda x=x: x[0] for x in options])
    bidirectional_size_to_use = tf.switch_case(our_choice, branch_fns=[lambda x=x: x[1] for x in options])
    return our_choice, inputs_to_use, bidirectional_size_to_use

def _dataset_parser(value, max_seq_length=1025, do_l2r_also=True):
    """

    With some probability do bidirectional, with some probability do left-to-right

    :param value: TFRecord to decode
    :param config: NeatConfig > model and NeatConfig > data.
    :param is_training:
    :return:
    """
    with tf.name_scope('parser'):
        data = _decode_record(value)
        data = {k: tf.cast(v, dtype=tf.int32) for k, v in data.items()}

        features = {}
        T = get_shape_list(data['input_ids'], 1)[0]
        mbp, tweights = _rowan_mask_v3(data['input_ids'])

        features['target_ids'] = p2fsz(data['input_ids'], pad_value=0, output_shape=[max_seq_length], axis=0,
                                       truncate=True)

        if do_l2r_also:
            our_choice, mbp, bidirectional_size = combine_bidirectional_and_unidirectional(
                masked_inputs=mbp, gt_inputs=data['input_ids'])

            # Recompute weights with a default value of 0.05, this seems to work
            tweights = tf.concat([tweights[:bidirectional_size], tf.fill([T - bidirectional_size], 0.05)], axis=0)

            features['is_bidirectional'] = p2fsz(tf.cast(tf.less(tf.range(T, dtype=tf.int32), bidirectional_size), dtype=tf.int32),
                                                 pad_value=0,
                                                 output_shape=[max_seq_length], axis=0, truncate=True)
        else:
            # if 'concept_ids' in data:
            #     # Not really supported any more
            #     features['concept_ids'] = p2fsz(data['concept_ids'], pad_value=0, output_shape=[max_seq_length], axis=0,
            #                                     truncate=True)
            features['is_bidirectional'] = p2fsz(tf.ones(get_shape_list(data['input_ids'], 1), dtype=tf.int32),
                                                 pad_value=0,
                                                 output_shape=[max_seq_length], axis=0, truncate=True)

        features['target_weights'] = p2fsz(tweights, pad_value=0.0, output_shape=[max_seq_length], axis=0,
                                           truncate=True)
        features['input_ids'] = p2fsz(mbp, pad_value=0, output_shape=[max_seq_length], axis=0, truncate=True)

    return features, {}


def input_fn_builder(input_file, seq_length, is_training=True, do_l2r_also=True):
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
                cycle_length=8,
                sloppy=is_training))

        if is_training:
            dataset = dataset.shuffle(buffer_size=128)

        dataset = dataset.map(lambda x: _dataset_parser(x, max_seq_length=seq_length, do_l2r_also=do_l2r_also),
                              num_parallel_calls=8)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # # This flattens and then transposes the images, since big tensors are otherwise padded :/
        # def _transpose_images(features, labels):
        #     features['frames'] = tf.transpose(
        #         tf.reshape(features['frames'], [batch_size * config.data['max_frames'], IMAGE_HEIGHT, IMAGE_WIDTH, 3]), [1, 2, 3, 0])
        #     return features, labels
        #
        # # Batch level stuff
        # dataset = dataset.map(_transpose_images, num_parallel_calls=64)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    return input_fn


if __name__ == '__main__':
    use_train = True
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess = tf.Session()

    tfrecord_fn = 'gs://ipk-europe-west4/zslm-aug3/0000of4096.tfrecord'
    tfrecord_fn = '0000of4096.tfrecord'
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(_dataset_parser)
    it = dataset.make_one_shot_iterator()

    out = sess.run(it.get_next()[0])

    from data.zeroshot_lm_setup.encoder import get_encoder

    encoder = get_encoder()

    out_len = out['is_bidirectional'].sum()

    encoder.decode(out['input_ids'][:out_len])
