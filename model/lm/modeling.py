import sys
sys.path.append('../../')
import copy
import json
import math
import os
import six
import tensorflow as tf

from model import optimization

from model.model_utils import get_assignment_map_from_checkpoint, get_shape_list, create_unilm_attention_mask, get_ltr_attention_mask, gelu, layer_norm, dropout, construct_host_call, bfloat16_getter
from model.transformer import create_initializer, attention_layer, residual_mlp_layer, embed, _top_p_sample
from data.zeroshot_lm_setup.encoder import get_encoder

encoder = get_encoder()
class LMConfig(object):
    """Configuration for `GroverModel`"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02):
        """Constructs NewsConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `GroverModel`.
          hidden_size: Size of the layers
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = 0

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `NewsConfig` from a Python dictionary of parameters."""
        config = LMConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `NewsConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_size(cls, size):
        assert size in ['base', 'mini']
        return cls.from_json_file(os.path.join(os.path.dirname(__file__), 'configs', f'{size}.json'))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class GroverModel(object):
    def __init__(self,
                 config: LMConfig,
                 is_training,
                 input_ids,
                 is_bidirectional,
                 cache=None,
                 do_cache=False,
                 pad_token_id=0,
                 scope=None,
                 reuse=False,
                 use_bfloat16=False,
                 bow_lookahead_for_conditional_ltr_pretraining=False,
                 hidden_state_for_conditional_ltr=None,
                 mask_for_cache=None
                 ):
        """
        :param config:
        :param is_training:
        :param input_ids: Tensor thats of size [batch_size, seq_length]
        :param cache: Optionally, a tensor to use that will contain cached information of the size
            [batch_size, num_layers, 2, num_heads, cache_length, features]
        :param do_cache: Whether to cache again.
        :param pad_token_id: Which token will be used for padding (probably 0.)
        :param scope: scope to run this on
        """
        self.config = copy.deepcopy(config)
        self.is_training = is_training
        self.pad_token_id = pad_token_id

        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0
        self.input_ids = input_ids
        self.is_bidirectional = is_bidirectional
        self.batch_size, self.seq_length = get_shape_list(self.input_ids, 2)

        if cache is None:
            caches = [None] * config.num_hidden_layers
            self.cache_length = 0
        else:
            batch_size_, num_layers_, two_, num_heads_, self.cache_length, features_ = get_shape_list(
                cache, expected_rank=6)
            assert batch_size_ == self.batch_size
            assert num_layers_ == config.num_hidden_layers
            assert two_ == 2
            assert num_heads_ == config.num_attention_heads
            assert features_ == (config.hidden_size // config.num_attention_heads)
            caches = tf.unstack(cache, axis=1)

        # Possible types
        # This is for debugging where the loss comes from
        # -1 ignore
        # 0 is bidirectional
        # 1 is left to right
        # 2 is left to right (+ cheating)
        self.type_names = ['bidirectional', 'left_to_right', 'left_to_right_lookahead']
        is_padding = tf.cast(tf.math.equal(self.input_ids, 0), dtype=tf.float32)
        is_bidirectional = tf.cast(self.is_bidirectional, dtype=tf.float32) * (1.0 - is_padding)
        is_left_to_right = 1.0 - is_bidirectional - is_padding
        self.token_types = tf.cast((is_padding * -1.0) + is_left_to_right * 1.0, dtype=tf.int32)

        with tf.variable_scope(scope, default_name='basiclm', reuse=reuse,
                               custom_getter=bfloat16_getter() if use_bfloat16 else None):
            with tf.variable_scope("embeddings"):
                embeddings, self.embedding_table = embed(
                    self.input_ids,
                    config.vocab_size,
                    config.hidden_size,
                    position_offset=self.cache_length,
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    use_one_hot_embeddings=True,
                    hidden_state_for_conditional_ltr=hidden_state_for_conditional_ltr,
                )
                embeddings = layer_norm(embeddings, name='embed_norm')

            if bow_lookahead_for_conditional_ltr_pretraining:
                tf.logging.info("Looking ahead 32 tokens for pretraining")
                with tf.variable_scope("bow_lookahead_mlp"):
                    h0 = tf.stop_gradient(embeddings[:, 1:32])
                    h0_mapup = tf.layers.dense(
                        h0,
                        config.intermediate_size,
                        activation=gelu,
                        kernel_initializer=create_initializer(config.initializer_range),
                        name='h0',
                    )
                    h0_mapdown = tf.layers.dense(
                        h0_mapup,
                        config.hidden_size,
                        name='output',
                        kernel_initializer=create_initializer(config.initializer_range))
                    h0_mapdown = dropout(h0_mapdown, self.config.hidden_dropout_prob)
                    h1 = layer_norm(h0 + h0_mapdown, name='mlp1')
                    h1_mean = layer_norm(tf.reduce_mean(h1, 1), name='mlp1_mean_ln')

                    h_to_append = tf.layers.dense(
                        h1_mean,
                        self.config.hidden_size,
                        kernel_initializer=create_initializer(self.config.initializer_range),
                        name='bow_mlp2',
                    )
                    h_to_append_ln2 = layer_norm(h_to_append, name='embs_ln2')

                    # Add it in 90% of the time
                    probs = [self.config.hidden_dropout_prob, 1.0-self.config.hidden_dropout_prob]
                    assert all([p > 1e-5 for p in probs])
                    actually_make_conditional = tf.squeeze(tf.random.categorical(tf.math.log([probs]), dtype=tf.int32, num_samples=self.batch_size), axis=0)
                    actually_make_conditional = tf.cast(actually_make_conditional, dtype=tf.float32)
                    actually_make_conditional *= tf.cast(tf.equal(self.is_bidirectional[:, 0], 0), dtype=tf.float32)

                    embeddings0 = embeddings[:, 0] * (1.0 - actually_make_conditional[:, None]) + h_to_append_ln2 * actually_make_conditional[:, None]
                    embeddings = tf.concat([embeddings0[:, None], embeddings[:, 1:]], 1)

                    ###########################
                    is_semibidirectional = tf.cast(tf.less(tf.range(self.seq_length, dtype=tf.int32), 32),
                                                   dtype=tf.float32)[None] * actually_make_conditional[:, None]
                    is_semibidirectional *= (1.0 - is_bidirectional)
                    is_semibidirectional -= is_padding
                    is_semibidirectional = tf.cast(is_semibidirectional, dtype=tf.int32)
                    self.token_types += is_semibidirectional


            if cache is not None:
                tf.logging.info("DOING CACHE SO NO BIDIRECTIONAL")
                mask = get_ltr_attention_mask(self.seq_length, self.seq_length + self.cache_length,
                                              dtype=embeddings.dtype)
                if mask_for_cache is not None:
                    # this is the hackiest shit ever but essentially we need to not attend to some things in input
                    assert mask.shape[0] == 1
                    mask = tf.tile(mask[None], [self.batch_size, 1, 1])
                    m4c_bsz, m4c_sl = get_shape_list(mask_for_cache, 2)
                    assert m4c_bsz == self.batch_size
                    assert m4c_sl <= self.seq_length + self.cache_length

                    mask = tf.concat([
                        mask[:, :, :m4c_sl] * tf.cast(mask_for_cache[:, None], dtype=mask.dtype),
                        mask[:, :, m4c_sl:],
                    ], 2)
            else:
                mask = create_unilm_attention_mask(is_bidirectional=self.is_bidirectional,
                                                   is_padding=tf.math.equal(self.input_ids, 0))

            # We keep the representation as a 2D tensor to avoid re-shaping it back and
            # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
            # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
            # help the optimizer.
            hidden_state = tf.reshape(embeddings, [self.batch_size * self.seq_length, self.config.hidden_size])
            if use_bfloat16:
                hidden_state = tf.cast(hidden_state, tf.bfloat16)
                mask = tf.cast(mask, tf.bfloat16)
            new_kvs = []
            all_hidden_states = [hidden_state]
            for layer_idx, layer_cache in enumerate(caches):
                with tf.variable_scope('layer{:02d}'.format(layer_idx), custom_getter=bfloat16_getter() if use_bfloat16 else None):
                    # [batch_size * seq_length, hidden_size]
                    attention_output, new_kv = attention_layer(
                        hidden_state,
                        mask,
                        batch_size=self.batch_size,
                        seq_length=self.seq_length,
                        size_per_head=config.hidden_size // config.num_attention_heads,
                        num_attention_heads=config.num_attention_heads,
                        initializer_range=config.initializer_range,
                        hidden_dropout_prob=self.config.hidden_dropout_prob,
                        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                        do_cache=do_cache,
                        cache=layer_cache,
                    )
                    new_kvs.append(new_kv)

                    hidden_state = residual_mlp_layer(hidden_state + attention_output,
                                                     intermediate_size=config.intermediate_size,
                                                     hidden_dropout_prob=self.config.hidden_dropout_prob)

                    all_hidden_states.append(hidden_state)

            self.hidden_state = hidden_state
            self.all_hidden_states = tf.stack(all_hidden_states)

            if use_bfloat16:
                self.hidden_state = tf.cast(self.hidden_state, tf.float32)
                self.all_hidden_states = tf.cast(self.all_hidden_states, tf.float32)

        self.new_kvs = tf.stack(new_kvs, axis=1) if do_cache else None

        # Note that the hidden state is still flat (batch_size*hidden_size)
        self.logits_flat = tf.matmul(self.hidden_state, self.embedding_table, transpose_b=True)

        # ehh whatever
        # output_bias = tf.get_variable('output_bias', shape=[config.vocab_size], initializer=tf.zeros_initializer())
        # self.logits_flat = tf.nn.bias_add(self.logits_flat, output_bias)

    @property
    def global_hidden_state(self):
        h = tf.reshape(self.hidden_state, [self.batch_size, self.seq_length, self.config.hidden_size])[:, 0]
        return h

    @property
    def log_probs(self):
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)
        return tf.reshape(logprobs_flat, [self.batch_size, self.seq_length, -1])

    def lm_loss(self, target_ids, label_weights=None, return_per_example_loss=False):
        """
        :param target_bonus: Increase the loss on the targets by this much.
        :return: stuff
        """
        target_ids_flat = tf.reshape(target_ids, [-1])

        # [batch_size * seq_length, vocab_size]
        one_hot_labels = tf.one_hot(target_ids_flat,
                                    depth=self.config.vocab_size,
                                    dtype=self.logits_flat.dtype)

        # [batch_size * seq_length, vocab_size]
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)

        per_example_loss = -tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])

        if label_weights is None:
            label_weights = tf.cast(tf.not_equal(target_ids_flat, self.pad_token_id), dtype=self.logits_flat.dtype)
        else:
            label_weights = tf.reshape(label_weights, [-1])

        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = tf.reduce_sum(label_weights * per_example_loss) / denominator

        # For logging
        ##################
        weighted_per_example_loss = label_weights * per_example_loss / denominator
        token_types_oh = tf.one_hot(tf.reshape(self.token_types,[-1]), depth=3, dtype=tf.float32)
        count_by_type = tf.reduce_mean(token_types_oh, 0)
        loss_by_type = tf.reduce_sum(weighted_per_example_loss[:, None] * token_types_oh, 0)
        logging_info = {}
        for i, name in enumerate(self.type_names):
            logging_info[f'loss_by_type/{name}_loss'] = loss_by_type[i]
            logging_info[f'loss_by_type/{name}_normloss'] = loss_by_type[i] / (count_by_type[i] + 1e-6)
            logging_info[f'loss_by_type/{name}_count'] = count_by_type[i]

        if return_per_example_loss:
            return loss, logging_info, per_example_loss
        return loss, logging_info

    # def pooled_output(self, clf_token):
    #     """
    #     Extract pooled output given a token that says where we should look
    #     :param clf_token:
    #     :return:
    #     """
    #     pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(self.input_ids, clf_token), tf.float32), 1), tf.int32)
    #     return tf.gather(self.hidden_state, tf.range(self.batch_size, dtype=tf.int32) * self.seq_length + pool_idx)


def model_fn_builder(config: LMConfig, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, target_bonus=4.0):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = GroverModel(
            config=config,
            is_training=is_training,
            input_ids=features["input_ids"],
            is_bidirectional=features['is_bidirectional'],
            pad_token_id=config.pad_token_id,
            bow_lookahead_for_conditional_ltr_pretraining=True,
        )

        loss, loss_diagnostics = model.lm_loss(target_ids=features['target_ids'], label_weights=features['target_weights'])

        if is_training:
            train_op, train_metrics = optimization.create_fixed_adam_optimizer_with_warmup(
                loss=loss,
                learning_rate=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=use_tpu,
                weight_decay_rate=1e-2,
                beta_2=0.98,
                clip_norm=1.0,
                param_overrides=[[["LayerNorm", "layer_norm", 'GroupNorm', "bias"], {"weight_decay_rate": 0}]],
            )
            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            train_op = None
            train_metrics = {}
            tvars = tf.trainable_variables()

        train_metrics['loss'] = loss
        train_metrics.update(loss_diagnostics)

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if 'global_step' in assignment_map:
                del assignment_map['global_step']
                del initialized_variable_names['global_step:0']

            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                host_call=construct_host_call(scalars_to_log=train_metrics, model_dir=params['model_dir'],
                                            iterations_per_loop=1000),
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(total_loss):
                loss = tf.metrics.mean(values=total_loss)
                return {
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [loss])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            gt_logprobs = tf.squeeze(tf.batch_gather(model.log_probs, model.target_ids[:, :, None]), axis=2)

            # Need top-p required under topp sampling!
            better_than_gt = model.log_probs > gt_logprobs[:, :, None]
            top_p_required = tf.reduce_sum(tf.cast(better_than_gt, tf.float32) * tf.exp(model.log_probs), axis=2)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={'gt_logprobs': gt_logprobs,
                             'top_p_required': top_p_required,
                             'is_target': features['is_target'],
                             'labels': features['input_ids'],
                             },
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def sample_step(tokens, ignore_ids, news_config, batch_size=1, p_for_topp=0.95, cache=None, do_topk=False):
    """
    Helper function that samples from grover for a single step
    :param tokens: [batch_size, n_ctx_b] tokens that we will predict from
    :param ignore_ids: [n_vocab] mask of the tokens we don't want to predict
    :param news_config: config for the GroverModel
    :param batch_size: batch size to use
    :param p_for_topp: top-p or top-k threshold
    :param cache: [batch_size, news_config.num_hidden_layers, 2,
                   news_config.num_attention_heads, n_ctx_a,
                   news_config.hidden_size // news_config.num_attention_heads] OR, None
    :return: new_tokens, size [batch_size]
             new_probs, also size [batch_size]
             new_cache, size [batch_size, news_config.num_hidden_layers, 2, n_ctx_b,
                   news_config.num_attention_heads, news_config.hidden_size // news_config.num_attention_heads]
    """
    model = GroverModel(
        config=news_config,
        is_training=False,
        input_ids=tokens,
        reuse=tf.AUTO_REUSE,
        scope='newslm',
        chop_off_last_token=False,
        do_cache=True,
        cache=cache,
    )

    # Extract the FINAL SEQ LENGTH
    batch_size_times_seq_length, vocab_size = get_shape_list(model.logits_flat, expected_rank=2)
    prev_probs = tf.exp(tf.squeeze(tf.batch_gather(model.log_probs[:, :-1], tokens[:, 1:, None]), axis=2))

    logits = tf.reshape(model.logits_flat, [batch_size, -1, vocab_size])
    next_logits = logits[:, -1]

    if do_topk:
        raise ValueError("No topk")
    else:
        sample_info = _top_p_sample(next_logits, ignore_ids=ignore_ids, num_samples=1, p=p_for_topp)

    new_tokens = tf.squeeze(sample_info['sample'], 1)
    new_probs = tf.squeeze(tf.batch_gather(sample_info['probs'], sample_info['sample']), 1)
    return {
        'new_tokens': new_tokens,
        'new_probs': new_probs,
        'new_probs_all': tf.nn.softmax(next_logits, dim=-1),
        'prev_probs': prev_probs,
        'new_cache': model.new_kvs,
    }


def initialize_from_context(initial_context, ignore_ids, news_config, p_for_topp=0.95, do_topk=False):
    """ same signature as sample_step"""
    batch_size, _ = get_shape_list(initial_context, expected_rank=2)

    context_output = sample_step(tokens=initial_context, ignore_ids=ignore_ids, news_config=news_config,
                                 batch_size=batch_size, p_for_topp=p_for_topp, cache=None, do_topk=do_topk)
    return {
        'tokens': tf.concat([initial_context, context_output['new_tokens'][:, None]], 1),
        'cache': context_output['new_cache'],
        'probs': context_output['new_probs'][:, None]
    }


def sample(news_config: LMConfig, initial_context, eos_token, ignore_ids=None, p_for_topp=0.95,
           do_topk=False, max_len=1025):
    """
    V1 version of: sample outputs from a model, and do it all at once
    :param news_config: Configuration used to construct the model
    :param initial_context: [batch_size, seq_length] that we'll start generating with.
                            Everything in the batch must be the same size.
    :param eos_token: Stop generating if you see this (tf scalar)
    :param ignore_ids: NEVER GENERATE THESE [vocab_size]
    :return:
    """
    batch_size, _ = get_shape_list(initial_context, expected_rank=2)

    if ignore_ids is None:
        ignore_ids = tf.constant([x == 0 for x in range(news_config.vocab_size)], dtype=tf.bool)

    with tf.name_scope('sample_sequence'):
        # Initial call to get cache
        context_output = initialize_from_context(initial_context, ignore_ids=ignore_ids, news_config=news_config,
                                                 p_for_topp=p_for_topp,
                                                 do_topk=do_topk)
        ctx = context_output['tokens']
        cache = context_output['cache']
        probs = context_output['probs']

        def body(ctx, cache, probs):
            """ for whatever reason this didn't work when I ran it on more than one at once... ugh."""
            next_outputs = sample_step(ctx[:, -1][:, None], ignore_ids=ignore_ids, news_config=news_config,
                                       batch_size=batch_size, p_for_topp=p_for_topp, cache=cache,
                                       do_topk=do_topk)

            # Update everything
            new_cache = tf.concat([cache, next_outputs['new_cache']], axis=-2)
            new_ids = tf.concat([ctx, next_outputs['new_tokens'][:, None]], axis=1)
            new_probs = tf.concat([probs, next_outputs['new_probs'][:, None]], axis=1)
            return [new_ids, new_cache, new_probs]

        def cond(ctx, cache, probs):
            is_eos = tf.equal(ctx, eos_token)
            return tf.math.logical_not(tf.reduce_all(tf.reduce_any(is_eos, axis=1)))

        tokens, cache, probs = tf.while_loop(
            cond=cond, body=body, maximum_iterations=max_len - get_shape_list(ctx)[1],
            loop_vars=[ctx, cache, probs],
            shape_invariants=[tf.TensorShape([batch_size, None]),
                              tf.TensorShape(
                                  [batch_size, news_config.num_hidden_layers, 2,
                                   news_config.num_attention_heads,
                                   None, news_config.hidden_size // news_config.num_attention_heads]),
                              tf.TensorShape([batch_size, None]),
                              ],
            back_prop=False,
        )
    return tokens, probs


def sample_seq2seq(news_config: LMConfig, initial_context, eos_token, ignore_ids=None, p_for_topp=0.95,
                   do_topk=False, max_len=1025):
    """
    Sample multiple outputs for a model in a seq2seq way.

    :param news_config: Configuration used to construct the model
    :param initial_context: [batch_size, seq_length] that we'll start generating with.
                            Invalid entries are padded.
    :param eos_token: Stop generating if you see this (tf scalar)
    :param ignore_ids: NEVER GENERATE THESE [vocab_size]
    :return:
    """
    batch_size, ctxb_end = get_shape_list(initial_context, expected_rank=2)
    # This just says 'ignore the pad character'
    if ignore_ids is None:
        ignore_ids = tf.constant([x == 0 for x in range(news_config.vocab_size)], dtype=tf.bool)

    with tf.name_scope('sample_sequence'):
        # Not everything might be the same size so we need to get lens

        lens = tf.reduce_sum(tf.cast(tf.not_equal(initial_context, news_config.pad_token_id), dtype=tf.int32), axis=1)

        seq_is_valid = tf.greater(lens, 0)
        ctxb_start = tf.reduce_min(tf.where(seq_is_valid, lens, ctxb_end * tf.ones_like(lens)))

        initial_ctx_part_a = tf.identity(initial_context[:, :ctxb_start])
        initial_ctx_part_b = tf.identity(initial_context[:, ctxb_start:])

        # Initial call to get cache
        context_output = sample_step(tokens=initial_ctx_part_a, ignore_ids=ignore_ids, news_config=news_config,
                                     batch_size=batch_size, p_for_topp=p_for_topp, cache=None, do_topk=do_topk)

        def _append_new_tokens(current_ctx, new_tokens):
            """ At each step we add tokens. Sometimes those tokens conflict with what we already have.
                This function fixes that. It doesnt fix probabilities though!"""
            current_ctx_len = get_shape_list(current_ctx, expected_rank=2)[1]

            new_tokens = tf.cond(
                current_ctx_len < ctxb_end,
                true_fn=lambda: tf.where(
                    tf.equal(initial_ctx_part_b[:, current_ctx_len - ctxb_start], news_config.pad_token_id),
                    new_tokens,
                    initial_ctx_part_b[:, current_ctx_len - ctxb_start]),
                false_fn=lambda: new_tokens,
            )
            # import ipdb
            # ipdb.set_trace()
            # if current_ctx_len < ctxb_end:
            #     existing_tokens = initial_ctx_part_b[:,current_ctx_len-ctxb_start]
            #
            #     new_tokens=tf.where(tf.equal(existing_tokens, news_config.pad_token_id),
            #              new_tokens,
            #              existing_tokens)

            return tf.concat([current_ctx, new_tokens[:, None]], 1)

        ctx = _append_new_tokens(initial_ctx_part_a, context_output['new_tokens'])
        cache = context_output['new_cache']
        probs = tf.concat([context_output['prev_probs'],
                           tf.batch_gather(context_output['new_probs_all'], ctx[:, -1,None])], 1)

        def body(ctx, cache, probs):
            """ for whatever reason this didn't work when I ran it on more than one at once... ugh."""
            next_outputs = sample_step(ctx[:, -1][:, None], ignore_ids=ignore_ids, news_config=news_config,
                                       batch_size=batch_size, p_for_topp=p_for_topp, cache=cache,
                                       do_topk=do_topk)

            # Update everything. We might need to use the old tokens.
            new_cache = tf.concat([cache, next_outputs['new_cache']], axis=-2)
            new_ids = _append_new_tokens(ctx, next_outputs['new_tokens'])
            new_probs = tf.concat([probs, tf.batch_gather(next_outputs['new_probs_all'], new_ids[:, -1,None])], 1)
            return [new_ids, new_cache, new_probs]

        def cond(ctx, cache, probs):
            is_eos = tf.equal(ctx, eos_token)
            seq_is_eos = tf.math.logical_or(tf.reduce_any(is_eos, axis=1), tf.math.logical_not(seq_is_valid))

            return tf.math.logical_not(tf.reduce_all(seq_is_eos))

        tokens, cache, probs = tf.while_loop(
            cond=cond, body=body, maximum_iterations=max_len - get_shape_list(ctx)[1],
            loop_vars=[ctx, cache, probs],
            shape_invariants=[tf.TensorShape([batch_size, None]),
                              tf.TensorShape(
                                  [batch_size, news_config.num_hidden_layers, 2,
                                   news_config.num_attention_heads,
                                   None, news_config.hidden_size // news_config.num_attention_heads]),
                              tf.TensorShape([batch_size, None]),
                              ],
            back_prop=False,
        )
    return tokens, probs


if __name__ == '__main__':
    from model.lm import dataloader
    tf.compat.v1.enable_eager_execution()

    input_fn = dataloader.input_fn_builder('0000of4096.tfrecord',
                                           is_training=True,
                                           seq_length=512)

    features, labels = input_fn(params={'batch_size': 2}).make_one_shot_iterator().get_next()
    config = LMConfig.from_json_file('configs/mini.json')
    model = GroverModel(
        config=config,
        is_training=True,
        input_ids=features["input_ids"],
        is_bidirectional=features['is_bidirectional'],
        pad_token_id=config.pad_token_id,
    )

    loss, info = model.lm_loss(target_ids=features['target_ids'], label_weights=features['target_weights'])
