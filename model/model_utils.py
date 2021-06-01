# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Liense is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
import re

import six
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from typing import List, Union
from functools import reduce


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    # math.sqrt needed for bfloat16 compatibility
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / math.sqrt(2.0)))
    return input_tensor * cdf


# @tf.custom_gradient
# def gelu(input_tensor):
#     """
#     This version with the grad might be needed for memory? IDK. but I checked the grads are good
#     :param input_tensor:
#     :return:
#     """
#     def grad(dy):
#         with tf.control_dependencies([dy]):
#             cdf = 0.5 * (1.0 + tf.erf(input_tensor / math.sqrt(2.0)))
#             d_cdf = tf.exp(-tf.square(input_tensor)/2.0) / math.sqrt(2.0*math.pi)
#         return dy * (cdf + input_tensor * d_cdf)
#
#     cdf = 0.5 * (1.0 + tf.erf(input_tensor / math.sqrt(2.0)))
#     return input_tensor * cdf, grad


def layer_norm(input_tensor, name=None, epsilon=1e-5):
    """Run layer normalization on the last dimension of the tensor."""
    name2use = f'LayerNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='LayerNorm'):
        dim = input_tensor.shape[-1].value
        gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0))

        cast_up_to_float32 = input_tensor.dtype == tf.bfloat16
        if cast_up_to_float32:
            input_tensor = tf.cast(input_tensor, dtype=tf.float32)

        mean = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        std = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keepdims=True)
        input_tensor = (input_tensor - mean) * tf.rsqrt(std + epsilon)
        input_tensor = input_tensor * gamma + beta

        if cast_up_to_float32:
            input_tensor = tf.cast(input_tensor, dtype=tf.bfloat16)
    return input_tensor


def group_norm(inputs,
               channels_axis=-1,
               num_groups=32,
               channels_per_group=None,
               epsilon=1e-5,
               mean_close_to_zero=True,
               name=None):
    """
    A better implementation of groupnorm
    :param inputs: n-dimensional inputs
    :param channels_axis. Which channel has the groups. All the other channels except channel 0 are considered
           reduction axes.
    :param mean_close_to_zero: The mean of `input` before ReLU will be close to zero
        when batch size >= 4k for Resnet-50 on TPU. If `True`, use
        `nn.sufficient_statistics` and `nn.normalize_moments` to calculate the
        variance. This is the same behavior as `fused` equals `True` in batch
        normalization. If `False`, use `nn.moments` to calculate the variance.
        When `mean` is close to zero, like 1e-4, use `mean` to calculate the
        variance may have poor result due to repeated roundoff error and
        denormalization in `mean`.  When `mean` is large, like 1e2,
        sum(`input`^2) is so large that only the high-order digits of the elements
        are being accumulated. Thus, use sum(`input` - `mean`)^2/n to calculate
        the variance has better accuracy compared to (sum(`input`^2)/n - `mean`^2)
        when `mean` is large.
    :return:
    """
    name2use = f'GroupNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='GroupNorm'):
        x_shape = get_shape_list(inputs)

        # Make it positive for convenience
        channels_axis = len(x_shape) + channels_axis if channels_axis < 0 else channels_axis

        # Reshape into groups
        channels = x_shape[channels_axis]
        if num_groups is None:
            assert channels_per_group is not None
            num_groups = channels // channels_per_group
        elif channels_per_group is None:
            channels_per_group = channels // num_groups
        else:
            if channels != num_groups * channels_per_group:
                raise ValueError("Num groups = {} channels per group = {} but channels = {}".format(
                    num_groups, channels_per_group, channels
                ))
        if channels % channels_per_group != 0:
            raise ValueError('%d channels is not commensurate with %d channels/gp.' %
                             (channels, channels_per_group))

        axes_before_channels = list(x_shape[:channels_axis])
        axes_after_channels = list(x_shape[channels_axis + 1:])
        new_shape = axes_before_channels + [num_groups, channels_per_group] + axes_after_channels
        x_reshape = tf.reshape(inputs, new_shape)

        # Cast up to float32 if it was originally float16
        cast_up_to_float32 = x_reshape.dtype == tf.bfloat16
        if cast_up_to_float32:
            x_reshape = tf.cast(x_reshape, tf.float32)

        # Determine the dimensions across which moments are calculated. Skip batch axis.
        moments_axes = [a + 1 if a >= channels_axis else a for a in range(1, len(x_shape))]

        # Calculate the moments.
        if mean_close_to_zero:
            # One pass algorithm returns better result when mean is close to zero.
            counts, means_ss, variance_ss, _ = tf.nn.sufficient_statistics(
                x_reshape, moments_axes, keep_dims=True)
            mean, variance = tf.nn.normalize_moments(
                counts, means_ss, variance_ss, shift=None)
        else:
            mean, variance = tf.nn.moments(x_reshape, moments_axes, keep_dims=True)

        x_normed = (x_reshape - mean) * tf.math.rsqrt(variance + epsilon)

        # This matches the shape of X
        params_shape_broadcast = ([1] * len(axes_before_channels) +
                                  [channels] +
                                  [1] * len(axes_after_channels))

        gammas = tf.get_variable(name='gamma', shape=[channels], initializer=tf.constant_initializer(1),
                                 dtype=tf.float32)
        gammas = tf.reshape(gammas, params_shape_broadcast)

        betas = tf.get_variable(name='beta', shape=[channels], initializer=tf.zeros_initializer(), dtype=tf.float32)
        betas = tf.reshape(betas, params_shape_broadcast)

        outputs = tf.reshape(x_normed, x_shape) * gammas + betas
        if cast_up_to_float32:
            return tf.cast(outputs, tf.bfloat16)
        return outputs


def embedder(x, name, vocab_size, embedding_size, initializer_range=0.02,
             use_one_hot_embeddings=True):
    """
    Helper function for creating embeddings on TPUs
    :param x: Input to be used
    :param name: What to call it
    :param vocab_size:
    :param embedding_size:
    :param initializer_range: Will be a truncated normal in this range
    :return:
    """

    embedding_table = tf.get_variable(
        name=name,
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range),
    )

    less_than_max = tf.assert_less_equal(tf.reduce_max(x), vocab_size - 1)
    gt_zero = tf.assert_greater_equal(tf.reduce_min(x), 0)
    with tf.control_dependencies([less_than_max, gt_zero]):
        if use_one_hot_embeddings:
            flat_input_ids = tf.reshape(x, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output_flat = tf.matmul(one_hot_input_ids, embedding_table)
            embedded_input = tf.reshape(output_flat, get_shape_list(x) + [embedding_size])
        else:
            embedded_input = tf.nn.embedding_lookup(embedding_table, x)

    return embedded_input, embedding_table


def position_embedder(seq_length, name, max_position_embeddings, embedding_size, offset=0,
                      initializer_range=0.02):
    """

    :param seq_length: Length of the sequence to position embed. Must be less than max_position_embeddings.
    :param name: Name of the embedding
    :param max_position_embeddings: Highest it'll go
    :param embedding_size: dimension to map to
    :param offset: Currently this isn't supported but it's so you can deal with caching. In that case
                   we don't want to run all the old sequences through the transformer
    :param initializer_range: for truncated normal initializer
    :return:
    """
    if offset > 0:
        raise ValueError("Offsetting the position embeddings not supported now")

    # Do something special for position embeddings
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name=name,
            shape=[max_position_embeddings, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=initializer_range),
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
        position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])[None]

    return position_embeddings, full_position_embeddings


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def get_ltr_attention_mask(nd, ns, dtype):
    """
    this is a TPU compatible version of tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd)
    where the lower right triangle contains 1s
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def create_unilm_attention_mask(is_bidirectional, is_padding=None):
    """
    Creates a hybrid left-to-right as well as bidirectional attention mask
    :param is_bidirectional: [batch_size, seq_length] that's 1 if bidirectional otherwise 0
    :return: A float tensor that's [batch_size, from_seq_length, to_seq_length].

    Information flows from to_seq to from_seq.

    amask[b,i,j] = 1.0 if i >= j, or, if is_bidirectional[b,j]

    If we were doing caching (which we aren't) then from_seq_length could be a bit longer as we could maybe attend
    to the cache items.

    NOTE: The semantics of this are the same as OpenAI's masking from left to right fn.
    """
    batch_size, seq_length = get_shape_list(is_bidirectional, expected_rank=2)

    ltr_attention_mask = tf.range(seq_length)[:, None] >= tf.range(seq_length)
    joint_attention_mask = tf.cast(is_bidirectional[:, None, :], tf.bool) | ltr_attention_mask[None]

    if is_padding is not None:
        joint_attention_mask = tf.logical_and(joint_attention_mask, tf.math.logical_not(is_padding)[:, None])
    return tf.cast(joint_attention_mask, dtype=tf.float32)


def mask_attention_for_ltr(attention_scores, attention_mask):
    """
    Mask attention so that we're only predicting going forward
    :param attention_scores: [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    :param attention_mask [query_length, key_length]
    :return: masked attention
    """
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    mask = attention_mask[None, None]
    return attention_scores * mask - tf.cast(1e10, attention_scores.dtype) * (1 - mask)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, reference_name_transform=None):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        rhs_name = name if reference_name_transform is None else reference_name_transform(name)

        if rhs_name not in name_to_variable:
            continue
        assignment_map[name] = rhs_name
        initialized_variable_names[rhs_name] = 1
        initialized_variable_names[rhs_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


def construct_host_call(scalars_to_log, model_dir, iterations_per_loop=100):
    """
    Constructs the host call function + arguments for logging. You can plug this directly into the TF Estimator
    :param scalars_to_log: {name: scalar} tensor that will be logged.
    :param model_dir: Where to put everything
    :param iterations_per_loop: How long to flush
    :return:
    """

    def host_call_fn(global_step, **kwargs):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Returns:
          List of summary ops to run on the CPU host.
        """
        # Outfeed supports int32 but global_step is expected to be int64.
        global_step = tf.reduce_mean(global_step)
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf.contrib.summary.create_file_writer(model_dir, max_queue=iterations_per_loop).as_default():
            with tf.contrib.summary.always_record_summaries():
                for k, v in sorted(kwargs.items(), key=lambda x: (len(x[0].split('/')), x[0])):
                    tf.contrib.summary.scalar(
                        k, tf.reduce_mean(v), step=global_step)
                return tf.contrib.summary.all_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    host_call_dict = {name: scalar[None] for name, scalar in scalars_to_log.items()}
    host_call_dict['global_step'] = tf.reshape(tf.compat.v1.train.get_or_create_global_step(), [1])

    return host_call_fn, host_call_dict


def pad_to_fixed_size(data, pad_value, output_shape, axis: Union[List, int]=0,
                      truncate=True, name=None):
    """
    Pads the data to be a fixed size in the dimensions specified by axis.

    :param data: n-dimensional input.
    :param pad_value: What we will pad with
    :param output_shape: The desired output shape. This has to cover everything, not just axis.
    :param truncate: If True (default), we will TRUNCATE in the dimensions specifed by axis if we're over.
    :param axis: The axes to pad in. Pass a list to pad multiple dims.
    :return:
    """
    with tf.name_scope(name, default_name='pad_to_fixed_size', values=output_shape):
        axes = [axis] if isinstance(axis, int) else axis

        # Truncate if too long.
        pad_data = tf.identity(data)
        if truncate:
            slice_obj = [slice(0, os_i if i in axes else None, None) for i, os_i in enumerate(output_shape)]
            pad_data = pad_data[tuple(slice_obj)]

        # Anything not being padded, we assume is the output shape.
        current_shape = get_shape_list(pad_data, expected_rank=len(output_shape))
        for i, os_i in enumerate(output_shape):
            if i not in axes:
                current_shape[i] = os_i

        asserts = []
        for ax in axes:
            asserts.append(
                tf.Assert(tf.less_equal(current_shape[ax], output_shape[ax]), [current_shape[ax], output_shape[ax], ax])
            )

        with tf.control_dependencies(asserts):
            for ax in axes:
                pad_length = output_shape[ax] - current_shape[ax]
                pad_shape = [pad_length if i == ax else cs_i
                             for i, cs_i in enumerate(current_shape)]

                paddings = pad_value * tf.ones(pad_shape, dtype=data.dtype)
                pad_data = tf.concat([pad_data, paddings], axis=ax)

                # Update the dimension we padded in
                current_shape[ax] = output_shape[ax]

        pad_data = tf.reshape(pad_data, output_shape)
        return pad_data


def bfloat16_getter():
    """
    This is the magic that you need in order to get bfloat16 to work without messing up everything
    
    usually if you use bfloat16_scope that changes the variable scopes. but instead you can do
      with variable_scope.variable_scope(
      '', custom_getter=bfloat16_scope()) as varscope:
    
    :return: the getter
    """

    def inner_custom_getter(getter, *args, **kwargs):
        """Custom getter that forces variables to have type self.variable_type."""
        cast_to_bfloat16 = False
        requested_dtype = kwargs['dtype']
        if requested_dtype == tf.bfloat16:
            # Only change the variable dtype if doing so does not decrease variable
            # precision.
            kwargs['dtype'] = tf.float32
            cast_to_bfloat16 = True
        var = getter(*args, **kwargs)
        # This if statement is needed to guard the cast, because batch norm
        # assigns directly to the return value of this custom getter. The cast
        # makes the return value not a variable so it cannot be assigned. Batch
        # norm variables are always in fp32 so this if statement is never
        # triggered for them.
        if cast_to_bfloat16:
            var = tf.cast(var, tf.bfloat16)
        return var

    return inner_custom_getter


def binomial_sample(n, p):
    """
    Sample from a binomial.

    {\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}}
    so the logs are given by
    log(n!) - log(k!) - log((n-k!)) + k*log(p) + (n-k)*log(1-p)

    :param n: the n term (int)
    :param p: the p term (float)
    :return:
    """
    with tf.name_scope('binomial_sample'):
        # We can drop the initial n! term becauuse thats a constant
        counts = tf.cast(tf.range(0, n + 1), dtype=tf.float32)
        n_float = tf.cast(n, dtype=tf.float32)

        logits = -tf.math.lgamma(1. + n_float - counts) - tf.math.lgamma(1. + counts) + counts * tf.math.log(p) + (
                n_float - counts) * tf.math.log1p(-p)
        res = tf.reshape(tf.random.categorical(logits[None], dtype=tf.int32, num_samples=1), [])
    return res


def encode_string(tf_string, string_len):
    """
    Encodes the string into something TPU-able

    :param tf_string: string
    :param string_len: length
    :return: an encoded thing
    """
    out_raw = tf.cast(tf.io.decode_raw(tf_string, out_type=tf.uint8), dtype=tf.int32)[:string_len]
    return pad_to_fixed_size(out_raw, 0, [string_len])


def random_categorical_without_replacement(logits, num_samples):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    :param logits: [N] logits that are unscaled log probabilities
    :param num_samples:  <= N
    :return: num_samples inds that don't have repeatz
    """
    z = -tf.log(-tf.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return tf.cast(indices, dtype=tf.int32)


def randomize_onehot_given_budget(budget, num_elements: int):
    """
    :param budget: a [batch_size] tensor of how many things [0 <= x < num_rows] to change.
    :param num_elements: How many things to change
    :return: [batch_size, num_elements] with tf.reduce_sum(result, 1) == budget.
    """
    batch_size = get_shape_list(budget, 1)[0]
    rand_matrix = tf.random.uniform([batch_size, num_elements], minval=0, maxval=1.0, dtype=tf.float32)

    budget_clipped = tf.maximum(tf.minimum(budget, num_elements - 1), 0)

    threshold = tf.squeeze(tf.gather(tf.sort(rand_matrix, 1), budget_clipped[:, None], batch_dims=1), 1)
    onehot = tf.less(rand_matrix, threshold[:, None])
    return tf.cast(onehot, dtype=tf.int32)


def _reverse_image(x):
    mult_factor = tf.constant([0.229, 0.224, 0.225])
    plus_factor = tf.constant([0.485, 0.456, 0.406])
    for i in range(len(get_shape_list(x)) - 1):
        mult_factor = tf.expand_dims(mult_factor, 0)
        plus_factor = tf.expand_dims(plus_factor, 0)
    return x * mult_factor + plus_factor


def raw_smooth_l1_loss(diff, delta=1.0, max_val=10.0):
    """
    Creates smooth L1 loss. The regular version is sometimes unstable so here what we do is if the difference
    is > some value, we will return the log instead.

    So it's then
        0.5 * x^2                                                    if |x| <= d
        0.5 * d^2 + d * (|x| - d)                                    if max_val > |x| > d
        0.5 * d^2 + d * (max_val - d) + d*log(1.0+|x|-max_val)       if |x| > max_val

    :param diff:
    :param delta:
    :param max_val: It turns into log after here
    :return:
    """
    abs_diff = tf.abs(diff)

    huber_loss = tf.where(
        tf.math.less(abs_diff, delta),
        0.5 * tf.square(diff),
        0.5 * (delta ** 2) + delta * (abs_diff - delta),
    )
    huber_loss_capped = tf.where(
        tf.math.less(abs_diff, max_val),
        huber_loss,
        0.5 * (delta ** 2) + delta * (max_val - delta) + delta * tf.math.log1p(tf.math.abs(abs_diff - max_val))
    )
    return huber_loss_capped


def load_classes():
    """
    Loads the joint categories.

    You could then do
            # is_valid_for_dataset = (~pd.isnull(classes_df[['coco_name','vg_name', 'oi_name']])).values

    :return:
    """
    cats_fn = os.path.join(os.path.dirname(__file__), '..', 'data', 'vgpretrain', 'joint_categories.csv')
    classes_df = pd.read_csv(cats_fn)
    return classes_df


def decode_string(x):
    return ''.join([chr(c) for c in x.astype(np.uint8) if c != 0])


def pad_feature_map(x, padding=0):
    if padding == 0:
        return x
    bhwc_padding = tf.constant([
        [0, 0],
        [padding, padding],
        [padding, padding],
        [0, 0],
    ])
    return tf.pad(x, bhwc_padding, mode='CONSTANT', constant_values=0)


def pytorch_compatible_conv(x,
                            out_channels: int, kernel_size: int, stride: int = 1,
                            padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True,
                            kernel_initializer=None,
                            bias_initializer=None,
                            name=None):
    """
    :param x: input.
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dilation:
    :param groups:
    :param bias:
    :param padding_mode:
    :param name: optional name
    :return:
    """

    # Pad it
    output = tf.layers.conv2d(
        pad_feature_map(x, padding),
        out_channels,
        kernel_size=[kernel_size, kernel_size],
        strides=[stride, stride],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        padding='valid',
        data_format='channels_last',
        use_bias=bias,
        name=name,
    )
    return output


def _sequence_xe_loss_noreduce(x, target_ids):
    """
    :param x:
    :param target_ids:
    :return: flattened xe loss
    """
    vocab_size = get_shape_list(x)[-1]
    target_ids_flat = tf.reshape(target_ids, [-1])
    # [batch_size * seq_length, vocab_size]
    one_hot_labels = tf.one_hot(target_ids_flat, depth=vocab_size, dtype=x.dtype)

    logprobs_flat = tf.nn.log_softmax(tf.reshape(x, [-1, vocab_size]), axis=-1)
    per_example_loss = -tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])
    return per_example_loss


def sequence_xe_loss(x, target_ids, label_weights=None):
    """
    Computes sequence cross entropy, ie for language modeling
    :param x: [d0, d1, ..., dn, vocab_size]
    :param target_ids: [d0, d1, ..., dn] -> inds 0 to vocab_size
    :param label_weights: if None then all target_ids >= 0 are valid, otherwise these are used for scaling
    :return:
    """
    vocab_size = get_shape_list(x)[-1]
    target_ids_flat = tf.reshape(target_ids, [-1])
    # [batch_size * seq_length, vocab_size]
    one_hot_labels = tf.one_hot(target_ids_flat, depth=vocab_size, dtype=x.dtype)

    logprobs_flat = tf.nn.log_softmax(tf.reshape(x, [-1, vocab_size]), axis=-1)
    per_example_loss = -tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])

    if label_weights is None:
        label_weights = tf.cast(tf.greater_equal(target_ids_flat, 0), dtype=x.dtype)
    else:
        label_weights = tf.reshape(label_weights, [-1])

    denom = tf.reduce_sum(label_weights) + 1e-5
    loss = tf.reduce_sum(label_weights * per_example_loss) / denom
    return loss


def normalize_image(image):
    """
    Normalize image
    :param image: [..., 3]
    :return:
    """
    with tf.name_scope('normalize_image'):
        img_shape = get_shape_list(image)
        if img_shape[-1] != 3:
            raise ValueError("provided image of shape {}".format(img_shape))
        ndims = len(img_shape) - 1

        offset = tf.constant([0.485, 0.456, 0.406])
        scale = tf.constant([0.229, 0.224, 0.225])
        for i in range(ndims):
            offset = tf.expand_dims(offset, axis=0)
            scale = tf.expand_dims(scale, axis=0)

        normalized_image = (image - offset) / scale
    return normalized_image

def unnormalize_image_np(image):
    """
    Undo the normalize operation
    :param image:
    :return:
    """
    ndims = image.ndim-1
    offset = np.array([0.485, 0.456, 0.406])
    scale = np.array([0.229, 0.224, 0.225])
    for i in range(ndims):
        offset = offset[None]
        scale = scale[None]
    img_inv = (image * scale) + offset
    img_fixed = np.round(img_inv * 255).clip(min=0, max=255)
    return img_fixed.astype(np.uint8)


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))

def transpose_singledim(v, ax0: int, ax1: int):
    """
    :param v:
    :param axes: two dims
    :return:
    """
    vs = get_shape_list(v)
    axes = [x for x in range(len(vs))]

    a0val = axes.pop(ax0)
    axes.insert(ax1, a0val)
    return tf.transpose(v, axes)



def stack_jagged(values, axis=0, name='stack_jagged'):
    """
    Given a list of tensors we'll pad to the max size in the other dims
    :param values:
    :param axis:
    :return:
    """
    sizes = [get_shape_list(v) for v in values]
    if not all([len(s) == len(sizes[0]) for s in sizes]):
        raise ValueError(f"Sizes arent compatible for stack_jagged:\n{sizes}")

    if not all([s.dtype == values[0].dtype for s in values]):
        raise ValueError("Dtypes not the same {}".format(values))

    new_vsize = np.stack(sizes).max(0).tolist()
    with tf.name_scope(name):
        # stack_val = tf.zeros(new_size, dtype=values[0].dtype)
        stack_l = []

        for i, (v_i, s_i) in enumerate(zip(values, sizes)):
            diffs = []
            for j, (s_ij, s2_ij) in enumerate(zip(s_i, new_vsize)):
                if s_ij != s2_ij:
                    diffs.append(j)
            if len(diffs) == 0:
                stack_l.append(v_i)
            else:
                stack_l.append(pad_to_fixed_size(v_i, pad_value=0.0, output_shape=new_vsize,
                                                 axis=diffs, truncate=False, name=f'idx{i}'))
        stack_v = tf.stack(stack_l, axis=axis)

        # Create [1 if GOOD or 0 if PAD]
        sizes_np = np.stack(sizes, 1)
        onehot_matrix = []
        for i, (s_i, s_imax) in enumerate(zip(sizes_np, new_vsize)):
            is_good = tf.greater(tf.constant(s_i, dtype=tf.int32, name=f'size{i}'),
                              tf.range(0, s_imax, dtype=tf.int32)[:, None])
            for j in range(len(new_vsize)):
                if i != j:
                    is_good = tf.expand_dims(is_good, axis=j, name=f'expand_{i}_{j}')
            onehot_matrix.append(is_good)
        onehot_matrix_reduce = reduce(lambda x, y: x & y, onehot_matrix)
        onehot_matrix_t = transpose_singledim(onehot_matrix_reduce, -1, axis)

    return stack_v, onehot_matrix_t


def basic_mlp(x, output_size, name='basic_mlp', dropout_prob=0.1, initializer_range=0.02,
              is_training=True, activation=None):
    """
    A very simple 1 layer MLP with a gelu activation, layernorm and some dropout
    :param x:
    :param output_size:
    :param name:
    :param dropout_prob:
    :param initializer_range
    :param is_training
    :return:
    """
    intermediate_size = get_shape_list(x)[-1]
    with tf.variable_scope(name):
        x_ln = layer_norm(x, name='ln_input')
        x0 = tf.layers.dense(x_ln, intermediate_size, activation=gelu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range),
                             name='intermediate')
        if is_training and dropout_prob > 0.1:
            x0 = dropout(x0, dropout_prob)
        x0_combine = layer_norm(x0 + x)

        x1 = tf.layers.dense(x0_combine, output_size,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range),
                             name='output_proj',
                             activation=activation,
                             )
    return x1

def switch_block(x_a, x_b, choose_a_prob=0.5):
    """
    Chooses A or B over the first dimension
    :param x_a:
    :param x_b:
    :param a_prob:
    :return:
    """
    bsize_a = get_shape_list(x_a)[0]
    bsize_b = get_shape_list(x_b)[0]
    assert bsize_a == bsize_b

    num_dims = len(get_shape_list(x_a))

    shape = [bsize_a] + [1] * (num_dims-1)
    choose_a = tf.cast(tf.less_equal(tf.random_uniform(shape), choose_a_prob), dtype=tf.float32)
    return x_a * choose_a + x_b * (1.0 - choose_a)
