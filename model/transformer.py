"""
Transformer, mostly adapted from the old Grover code
"""
import math

import tensorflow as tf

from model.model_utils import dropout, get_shape_list, gelu, \
    layer_norm


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def _attention_projection_and_transpose(x_flat, batch_size, seq_length, num_attention_heads, size_per_head,
                                        name, initializer_range=0.02):
    """
    :param x_flat: [batch_size*seq_length, width]
    :return: A fixed up tensor of size [batch_size, num_attention_heads, seq_length, size_per_head]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    # Had to remove this bc of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    projected = tf.layers.dense(
        x_flat,
        num_attention_heads * size_per_head,
        name=name,
        kernel_initializer=create_initializer(initializer_range))

    projected = tf.reshape(
        projected, [batch_size, seq_length, num_attention_heads, size_per_head])
    output_tensor = tf.transpose(projected, [0, 2, 1, 3])
    return output_tensor


def attention_layer(x_flat, attention_mask, batch_size, seq_length, size_per_head=512, num_attention_heads=1, *,
                    cache=None,
                    initializer_range=0.02, hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1, do_cache=False):
    """

    :param x_flat: Tensor input, should be [batch_size*seq_length, dim]
    :param attention_mask: Attention mask to use of size [seq_length, seq_length+cached_length]
    :param size_per_head: dim = size_per_head * num_attention_heads
    :param num_attention_heads:  dim = size_per_head * num_attention_heads
    :param cache: Optionally some past (cached) things of size
                [batch, 2, heads, sequence, features], where 2 is [k, v]
    :param do_cache: True if we should return cache
    :return: A new tensor of shape [batch_size, seq_length, dim]
    as well as a new cache "cached_keys_and_values" that will be of size
                                   [batch_size, 2, num_attention_heads, seq_length, dim]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    # Had to remove this because of generation script
    # if (batch_size_seq_length != batch_size * seq_length):
    #     raise ValueError("passed in a tensor of shape {} when batch_size={} and seq_length={}".format(
    #         (batch_size_seq_length, dim), batch_size, seq_length
    #     ))

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    # if do_cache and past is not None:
    #     Shape will be (batch_size, 2, num_attention_heads, past_seq_length, dim)
    #     past_shape = get_shape_list(past, 5)
    #     desired_shape = (batch_size, 2, num_attention_heads, seq_length, dim)
    #     if tuple(past_shape) != desired_shape:
    #         raise ValueError(f"The shape of the cache is {past_shape} but we want {desired_shape}")

    # [ batch_size, num_attention_heads, seq_length, size_per_head]
    query = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='query_layer',
                                                initializer_range=initializer_range)
    key = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                              num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                              name='key_layer',
                                              initializer_range=initializer_range)

    value = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='value_layer',
                                                initializer_range=initializer_range)

    # Add to cache
    cached_keys_and_values = tf.stack([key, value], axis=1) if do_cache else None

    # Things that were relevant from the cache
    if cache is not None:
        pk, pv = tf.unstack(cache, axis=1)
        key = tf.concat([pk, key], axis=-2)
        value = tf.concat([pv, value], axis=-2)

    # Multiply [batch_size, num_attention_heads, seq_length, size_per_head] with
    #          [batch_size, num_attention_heads, size_per_head, seq_length+cached_length] ->
    #          [batch_size, num_attention_heads, seq_length, seq_length+cached_length]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    # Deal with the masking
    if len(get_shape_list(attention_mask)) == 2:
        tf.logging.info("Broadcasting the attention mask along the batch AND num_attention_heads dimensions")
        mask2use = attention_mask[None, None]
    elif len(get_shape_list(attention_mask)) == 3:
        tf.logging.info("Broadcasting the attention mask along the num_attention_heads dimension")
        mask2use = attention_mask[:, None]
    else:
        tf.logging.info("Not broadcasting the attention mask along ANY dimension")
        mask2use = attention_mask

    attention_scores = attention_scores * mask2use - tf.cast(1e10, attention_scores.dtype) * (1 - mask2use)
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if attention_probs_dropout_prob > 0.0:
        attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # Multiply [batch_size, num_attention_heads, seq_length, seq_length+cached_length] with
    #          [batch_size, num_attention_heads, seq_length+cached_length, size_per_head] ->
    #          [batch_size, num_attention_heads, seq_length, size_per_head] ->
    context_layer = tf.matmul(attention_probs, value)

    # `context_layer` = [batch_size, seq_length, num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    context_layer = tf.reshape(context_layer, [batch_size * seq_length, num_attention_heads * size_per_head])

    context_layer_projected = tf.layers.dense(
        context_layer,
        num_attention_heads * size_per_head,
        kernel_initializer=create_initializer(initializer_range),
        name='context_projection_layer'
    )
    context_layer_projected = dropout(context_layer_projected, hidden_dropout_prob)

    return context_layer_projected, cached_keys_and_values


def residual_mlp_layer(x_flat, intermediate_size, initializer_range=0.02, hidden_dropout_prob=0.1):
    """
    :param x_flat: The attention output. It should be [batch_size*seq_length, dim]
    :param intermediate_size: the hidden projection. By default this is the input_dim * 4.

    in the original GPT we would return layer_norm(x_norm + h1) rather than layer_norm(x + h1)

    :return:
    """
    batch_size_seq_length, hidden_size = get_shape_list(x_flat, expected_rank=2)
    x_norm = layer_norm(x_flat, name='mlp_ln0')

    intermediate_output = tf.layers.dense(
        x_norm,
        intermediate_size,
        activation=gelu,
        kernel_initializer=create_initializer(initializer_range),
        name='intermediate',
    )

    output_for_residual = tf.layers.dense(
        intermediate_output,
        hidden_size,
        name='output',
        kernel_initializer=create_initializer(initializer_range))
    output_for_residual = dropout(output_for_residual, hidden_dropout_prob)

    layer_output = layer_norm(x_flat + output_for_residual, name='mlp_ln1')
    return layer_output


def residual_mlp(input, hidden_size, final_size, num_layers=1, initializer_range=0.02, hidden_dropout_prob=0.1,
                 dropout_input=True, intermediate_size_scale=4, initial_proj=True, final_proj=True):
    """
    Multilayer residual MLP

    Initial linear -> num_layers residual MLP blocks -> final layer.

    :param input: Stuff to go in
    :param hidden_size: Hidden size to use
    :param final_size: Final size to project to
    :param num_layers: How many layers
    :param initializer_range:
    :param hidden_dropout_prob:
    :param intermediate_size_scale: Scale up hidden size by this much
    :param dropout_input: Whether to apply dropout to input
    :param initial_proj: Initial projection?
    :param final_proj: Final projection?
    :return:
    """
    x = dropout(input, hidden_dropout_prob) if dropout_input else input

    init_hidden_size = get_shape_list(input)[-1]
    if initial_proj or (init_hidden_size != hidden_size):
        tf.logging.info("Adding an initial projection")
        x = tf.layers.dense(
            x,
            hidden_size,
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range),
            name='l0_proj'
        )

    for hl in range(num_layers):
        with tf.variable_scope(f'layer{hl}'):
            x = residual_mlp_layer(x,
                                   intermediate_size=hidden_size * intermediate_size_scale,
                                   hidden_dropout_prob=hidden_dropout_prob)

    if final_proj or (final_size != hidden_size):
        x = tf.layers.dense(
            x,
            final_size,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='lfinal_proj'
        )
    return x


def embed(input_ids,
          vocab_size,
          embedding_size,
          position_offset=0,
          initializer_range=0.02,
          max_position_embeddings=512,
          use_one_hot_embeddings=True,
          hidden_state_for_conditional_ltr=None):
    """regular and position embeddings
    :param input_ids: int Tensor of shape [batch_size, seq_length].
    :param vocab_size: number of words in vocab
    :param embedding_size: dimensionality of the embedding
    :param position_offset: aka number of cached tokens.
    :param initializer_range: float. Range of the weight initialization.
    :param max_position_embeddings: int. Maximum sequence length.
    :param use_one_hot_embeddings: probably want this to be true
    :param hidden_state_for_conditional_ltr: [batch_size, seq_length, H] tack this on at the beginning.
    :return: [batch_size, seq_length, embedding_size] embedded tensor
    """
    (batch_size, seq_length) = get_shape_list(input_ids, expected_rank=2)

    embedding_table = tf.get_variable(
        name='word_embed',
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
    )

    assert_op = tf.assert_less_equal(tf.reduce_max(input_ids), vocab_size - 1)
    with tf.control_dependencies([assert_op]):
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(input_ids, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output_flat = tf.matmul(one_hot_input_ids, embedding_table)
        else:
            output_flat = tf.nn.embedding_lookup(embedding_table, input_ids)

        embedded_input = tf.reshape(output_flat, [batch_size, seq_length, embedding_size])

        if hidden_state_for_conditional_ltr is not None:
            tf.logging.info("Embedding hidden_state_for_conditional_ltr")
            seq_length_hidden = get_shape_list(hidden_state_for_conditional_ltr, expected_rank=3)[1]
            embedded_input = tf.concat([hidden_state_for_conditional_ltr, embedded_input[:, seq_length_hidden:]], 1)


    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)

    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name='pos_embed',
            shape=[max_position_embeddings, embedding_size],
            initializer=create_initializer(initializer_range),
        )
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        if position_offset == 0:
            embedded_input += tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])[None]
        else:
            # Tensorflow is too stupid to allow slicing
            flat_pos_ids = (tf.range(seq_length, dtype=tf.int32) + position_offset)
            one_hot_pos_ids = tf.one_hot(flat_pos_ids, depth=max_position_embeddings)

            # [seq_length, full_position_embeddings], [full_position_embeddings, dim]
            seq_embeds = tf.matmul(one_hot_pos_ids, full_position_embeddings)
            embedded_input += seq_embeds[None]

            # embedded_input += tf.slice(full_position_embeddings[position_offset:], [0, 0], [seq_length, -1])[None]

    return embedded_input, embedding_table


def _top_p_sample(logits, ignore_ids=None, num_samples=1, p=0.9):
    """
    Does top-p sampling. if ignore_ids is on, then we will zero out those logits.
    :param logits: [batch_size, vocab_size] tensor
    :param ignore_ids: [vocab_size] one-hot representation of the indices we'd like to ignore and never predict,
                        like padding maybe
    :param p: topp threshold to use, either a float or a [batch_size] vector
    :return: [batch_size, num_samples] samples

    # TODO FIGURE OUT HOW TO DO THIS ON TPUS. IT'S HELLA SLOW RIGHT NOW, DUE TO ARGSORT I THINK
    """
    with tf.variable_scope('top_p_sample'):
        batch_size, vocab_size = get_shape_list(logits, expected_rank=2)

        probs = tf.nn.softmax(logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10,
                              axis=-1)

        if isinstance(p, float) and p > 0.999999:
            # Don't do top-p sampling in this case
            print("Top-p sampling DISABLED", flush=True)
            return {
                'probs': probs,
                'sample': tf.random.categorical(
                    logits=logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10,
                    num_samples=num_samples, dtype=tf.int32),
            }

        # [batch_size, vocab_perm]
        indices = tf.argsort(probs, direction='DESCENDING')
        cumulative_probabilities = tf.math.cumsum(tf.batch_gather(probs, indices), axis=-1, exclusive=False)

        # find the top pth index to cut off. careful we don't want to cutoff everything!
        # result will be [batch_size, vocab_perm]
        p_expanded = p if isinstance(p, float) else p[:, None]
        exclude_mask = tf.logical_not(
            tf.logical_or(cumulative_probabilities < p_expanded, tf.range(vocab_size)[None] < 1))

        # OPTION A - sample in the sorted space, then unsort.
        logits_to_use = tf.batch_gather(logits, indices) - tf.cast(exclude_mask, tf.float32) * 1e10
        sample_perm = tf.random.categorical(logits=logits_to_use, num_samples=num_samples)
        sample = tf.batch_gather(indices, sample_perm)

        # OPTION B - unsort first - Indices need to go back to 0 -> N-1 -- then sample
        # unperm_indices = tf.argsort(indices, direction='ASCENDING')
        # include_mask_unperm = tf.batch_gather(include_mask, unperm_indices)
        # logits_to_use = logits - (1 - tf.cast(include_mask_unperm, tf.float32)) * 1e10
        # sample = tf.random.categorical(logits=logits_to_use, num_samples=num_samples, dtype=tf.int32)

    return {
        'probs': probs,
        # 'cumsum': cumulative_probabilities,
        'sample': sample,
        # 'indices_sorted': indices,
        # 'logits_masked': logits_to_use,
        # 'logits_raw': tf.batch_gather(logits_to_use, indices),
    }


def _top_k_sample(logits, ignore_ids=None, num_samples=1, k=10):
    """
    Does top-k sampling. if ignore_ids is on, then we will zero out those logits.
    :param logits: [batch_size, vocab_size] tensor
    :param ignore_ids: [vocab_size] one-hot representation of the indices we'd like to ignore and never predict,
                        like padding maybe
    :param p: topp threshold to use, either a float or a [batch_size] vector
    :return: [batch_size, num_samples] samples

    # TODO FIGURE OUT HOW TO DO THIS ON TPUS. IT'S HELLA SLOW RIGHT NOW, DUE TO ARGSORT I THINK
    """
    with tf.variable_scope('top_p_sample'):
        batch_size, vocab_size = get_shape_list(logits, expected_rank=2)

        probs = tf.nn.softmax(logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10,
                              axis=-1)
        # [batch_size, vocab_perm]
        indices = tf.argsort(probs, direction='DESCENDING')

        # find the top pth index to cut off. careful we don't want to cutoff everything!
        # result will be [batch_size, vocab_perm]
        k_expanded = k if isinstance(k, int) else k[:, None]
        exclude_mask = tf.range(vocab_size)[None] >= k_expanded

        # OPTION A - sample in the sorted space, then unsort.
        logits_to_use = tf.batch_gather(logits, indices) - tf.cast(exclude_mask, tf.float32) * 1e10
        sample_perm = tf.random.categorical(logits=logits_to_use, num_samples=num_samples)
        sample = tf.batch_gather(indices, sample_perm)

    return {
        'probs': probs,
        'sample': sample,
    }

def _argmax_sample(logits, ignore_ids=None):
    """
    Does top-k sampling. if ignore_ids is on, then we will zero out those logits.
    :param logits: [batch_size, vocab_size] tensor
    :param ignore_ids: [vocab_size] one-hot representation of the indices we'd like to ignore and never predict,
                        like padding maybe
    :param p: topp threshold to use, either a float or a [batch_size] vector
    :return: [batch_size, num_samples] samples

    """
    with tf.variable_scope('argmax_sample'):
        batch_size, vocab_size = get_shape_list(logits, expected_rank=2)

        probs = tf.nn.softmax(logits if ignore_ids is None else logits - tf.cast(ignore_ids[None], tf.float32) * 1e10,
                              axis=-1)
        # [batch_size, vocab_perm]
        sample = tf.argmax(probs, 1, output_type=tf.int32)[:, None]
    return {
        'probs': probs,
        'sample': sample,
    }
