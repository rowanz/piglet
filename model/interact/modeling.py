import tensorflow as tf
import sys

sys.path.append('../../')
from model.model_utils import bfloat16_getter, get_shape_list, gelu, dropout, layer_norm, sequence_xe_loss, \
    construct_host_call, get_assignment_map_from_checkpoint, _sequence_xe_loss_noreduce, stack_jagged, \
    get_ltr_attention_mask
from data.thor_constants import THOR_OBJECT_TYPE_TO_IND, THOR_AFFORDANCES, THOR_ACTION_TYPE_TO_IND, \
    load_instance_attribute_weights
from model.transformer import attention_layer, residual_mlp_layer, _argmax_sample, residual_mlp
import math
from model import optimization
from model.neat_config import NeatConfig
from model.interact.dataloader import names_and_arities


def embed_with_embedding_table(x, embedding_table, flatten=False):
    """
    Embed an int tensor with the embedding table. This ignores -1 things
    :param x:
    :param embedding_table:
    :param flatten: Keep it flat versus reshape to the original like size
    :return:
    """
    x_shape = get_shape_list(x)
    vocab_size, embedding_dim = get_shape_list(embedding_table, 2)
    # Need to do something weird bc tf.float32_ref exists
    one_hot_x = tf.one_hot(tf.reshape(x, [-1]),
                           dtype=embedding_table.dtype if embedding_table.dtype in (
                               tf.float32, tf.bfloat16) else tf.float32,
                           depth=vocab_size)
    output = tf.matmul(one_hot_x, embedding_table)

    if not flatten:
        output = tf.reshape(output, x_shape + [embedding_dim])
    return output


def embed_2d_with_embedding_table(x, embedding_table, flatten=False):
    """
    :param x: [..., num_affordances]
    :param embedding_table_stacked: [num_affordances, vocab_size, hidden_size]
    :return:
    """
    x_shape = get_shape_list(x)
    num_affordances, vocab_size, hidden_size = get_shape_list(embedding_table, 3)
    # assert x_shape[-1] == num_affordances

    x_oh = tf.one_hot(tf.reshape(x, [-1, num_affordances]), depth=vocab_size, dtype=tf.float32)
    x_embed = tf.einsum('bav,avh->bah', x_oh, embedding_table)
    if not flatten:
        x_embed = tf.reshape(x_embed, x_shape + [hidden_size])
    return x_embed


def summarize_transformer(object_embs, gt_affordances_embed, affordance_name_embed, num_layers=3,
                          dropout_prob=0.1, initializer_range=0.02):
    """
    Use a transformer to summarize the delta between the GT affordances and the prototype that we'd expect from the object

    :param object_embs: [batch_size, h]
    :param gt_affordances_embed: [batch_size, num_affordances, h]
    :param affordance_name_embed: [num_affordances, h]
    :param num_layers:
    :param dropout_prob:
    :param initializer_range:
    :return: [batch_size, h] fixed-size representations for each of the objects!
    """
    batch_size, hidden_size = get_shape_list(object_embs, 2)
    batch_size2, num_affordances, h2 = get_shape_list(gt_affordances_embed, 3)
    num_affordances3, h3 = get_shape_list(affordance_name_embed, 2)

    assert hidden_size % 64 == 0
    assert hidden_size == h2
    assert h2 == h3

    # [POOL_IDX, OBJECT_NAME, ... attrs ... ]
    seq_length = num_affordances + 1
    with tf.variable_scope("summarize_transformer"):
        with tf.variable_scope('embeddings'):
            # starting_embed = tf.get_variable(
            #     name='pooler',
            #     shape=[hidden_size],
            #     initializer=tf.truncated_normal_initializer(stddev=initializer_range),
            # )
            ctx = layer_norm(tf.concat([
                # tf.tile(starting_embed[None, None], [batch_size, 1, 1]),
                object_embs[:, None],
                gt_affordances_embed + affordance_name_embed[None],
            ], 1), name='embed_norm')

            hidden_state = tf.reshape(ctx, [batch_size * seq_length, -1])

            # No masks bc all embeddings are used
            mask = tf.ones((seq_length, seq_length), dtype=tf.float32)

        for layer_idx in range(num_layers):
            with tf.variable_scope(f'layer{layer_idx:02d}'):
                # [batch_size * seq_length, hidden_size]
                attention_output, _ = attention_layer(
                    hidden_state,
                    mask,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    size_per_head=64,
                    num_attention_heads=hidden_size // 64,
                    initializer_range=initializer_range,
                    hidden_dropout_prob=dropout_prob,
                    attention_probs_dropout_prob=dropout_prob,
                )
                hidden_state = residual_mlp_layer(hidden_state + attention_output,
                                                  intermediate_size=hidden_size * 4,
                                                  hidden_dropout_prob=dropout_prob)
        h0 = tf.reshape(hidden_state, [batch_size, seq_length, -1])[:, 0]
    return h0


def expand_transformer(object_full_state, gt_affordances_embed, affordance_ctx_name_embed,
                       affordance_trg_name_embed, num_layers=3, dropout_prob=0.1, initializer_range=0.02,
                       random_perms=True, reuse=False, layer_cache=None):
    """
    Use a transformer to predict what the actual affordances of the object are, from the state

    # The order will be
    (object hidden state)
    (nullctx, nullctxname, pred0name) -> pred0
    (gt0, gt0name, pred1name) -> pred1
                     ...
    (gt{n-1},  gt{n-1}name, predNname) -> predN


    :param object_full_state: [batch_size, h]
    :param gt_affordances_embed: [batch_size, num_affordances, h]
    :param affordance_ctx_name_embed: [num_affordances, h]
    :param affordance_trg_name_embed: [num_affordances, h]
    :param num_layers:
    :param random_perms: Randomly permute

    :return: hidden size of [batch_size, num_affordances, h]
    """
    batch_size, hidden_size = get_shape_list(object_full_state, 2)
    batch_size2, num_affordances, h2 = get_shape_list(gt_affordances_embed, 3)
    num_affordances3, h3 = get_shape_list(affordance_ctx_name_embed, 2)
    num_affordances4, h4 = get_shape_list(affordance_trg_name_embed, 2)

    assert hidden_size % 64 == 0
    assert hidden_size == h2
    assert h2 == h3

    # [OBJECT_NAME, ... attrs ... ]
    seq_length = num_affordances + 1
    with tf.variable_scope("expand_transformer", reuse=reuse):
        if random_perms:
            idxs = tf.argsort(tf.random.normal((batch_size, num_affordances)), 1)
        else:
            idxs = tf.tile(tf.range(num_affordances, dtype=tf.int32)[None], [batch_size, 1])

        with tf.variable_scope('embeddings'):
            null_ctx_embed = tf.get_variable(
                name='nullctx',
                shape=[hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=initializer_range),
            )

            ctx_embeds = tf.concat([
                tf.tile(null_ctx_embed[None, None], [batch_size, 1, 1]),
                tf.gather(gt_affordances_embed + affordance_ctx_name_embed[None], idxs[:, :-1], batch_dims=1),
            ], 1)
            trg_name_embeds = tf.gather(tf.tile(affordance_trg_name_embed[None], [batch_size, 1, 1]),
                                        idxs, batch_dims=1)

            ctx = layer_norm(tf.concat([
                object_full_state[:, None],
                ctx_embeds + trg_name_embeds,
            ], 1), name='embed_norm')

            # don't forget to wear a mask when you go outside!
            if layer_cache is not None:
                # Shrink hidden state and mask accordingly
                cache_length = get_shape_list(layer_cache, expected_rank=6)[-2]
                seq_length = 1
                ctx = ctx[:, -seq_length:]
                mask = get_ltr_attention_mask(1, 1 + cache_length, dtype=ctx.dtype)
            else:
                mask = get_ltr_attention_mask(seq_length, seq_length, dtype=ctx.dtype)
            hidden_state = tf.reshape(ctx, [batch_size * seq_length, -1])

        new_kvs = []
        for layer_idx in range(num_layers):
            with tf.variable_scope(f'layer{layer_idx:02d}'):
                # [batch_size * seq_length, hidden_size]
                attention_output, new_kv = attention_layer(
                    hidden_state,
                    mask,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    size_per_head=64,
                    num_attention_heads=hidden_size // 64,
                    initializer_range=initializer_range,
                    hidden_dropout_prob=dropout_prob,
                    attention_probs_dropout_prob=dropout_prob,
                    do_cache=True,
                    cache=layer_cache[:, layer_idx] if layer_cache is not None else None,
                )
                new_kvs.append(new_kv)
                hidden_state = residual_mlp_layer(hidden_state + attention_output,
                                                  intermediate_size=hidden_size * 4,
                                                  hidden_dropout_prob=dropout_prob)
    # [batch_size, num_attributes, H]
    if layer_cache is None:
        hidden_states_per_attr = tf.gather(tf.reshape(hidden_state, [batch_size, seq_length, -1]),
                                           tf.argsort(idxs, 1) + 1, batch_dims=1)
    else:
        hidden_states_per_attr = hidden_state[:, None]

    return hidden_states_per_attr, tf.stack(new_kvs, axis=1)


class StateChangePredictModel(object):
    def __init__(self, config: NeatConfig, is_training, object_types):
        """
        A model to predict what happens to some objects when you apply an action

        :param config:
        :param is_training:
        :param object_types: [batch_size, num_objects, (pre,post) aka 2]
        """
        self.config = config
        self.hidden_size = config.model['hidden_size']
        self.is_training = is_training
        if is_training:
            self.dropout_prob = config.model.get('dropout_prob', 0.1)
            tf.logging.info("Is training -> dropout={:.3f}".format(self.dropout_prob))
        else:
            self.dropout_prob = 0.0

        self.activation_fn = tf.nn.tanh if config.model.get('activation', 'tanh') == 'tanh' else tf.identity

        # First embed everything, some of these are static.
        with tf.variable_scope('embeddings'):
            # 1. Embed everything
            object_embedding_table = tf.get_variable(
                name='object_embs',
                shape=[len(THOR_OBJECT_TYPE_TO_IND), self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )
            # Technically we assume as input
            # [batch_size, num_objects (2), pre post (2)]
            # However those last two dimensions were flattened into [batch_size, 4]
            # Now we're flattening into [batch_size * 4]
            self.batch_size, self.num_objects = get_shape_list(object_types, 2)
            assert self.num_objects == 4
            self.object_embed = embed_with_embedding_table(object_types, object_embedding_table,
                                                           flatten=True)

            affordance_embed_table = []
            for i, (affordance_name, a) in enumerate(names_and_arities):
                if a == len(THOR_OBJECT_TYPE_TO_IND):
                    tf.logging.info(f"For {affordance_name}: i'm copying the object embedding table")
                    affordance_embed_table.append(object_embedding_table)
                else:
                    affordance_embed_table.append(tf.get_variable(
                        name=f'{affordance_name}',
                        shape=[max(a, 2), self.hidden_size],
                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                    ))

            # [num_affordances, vocab_size, hidden_size]
            self.affordance_embed_table, self.affordance_embed_table_mask = stack_jagged(affordance_embed_table, 0)
            self.num_affordances, self.affordance_vocab_size, _hsz = get_shape_list(self.affordance_embed_table, 3)
            tf.logging.info(f"Affordance embed table: ({self.num_affordances},{self.affordance_vocab_size},{_hsz})")

            self.affordance_emb_trg = tf.get_variable(
                name='affordance_embs_trg',
                shape=[len(names_and_arities), self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )
            self.affordance_emb_ctx = tf.get_variable(
                name='affordance_embs_ctx',
                shape=[len(names_and_arities), self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )

    def encode_affordances(self, object_states):
        """
        :param object_states: [batch_size, num_objects, num_affordances]
        :return: encoded hidden size. [batch_size, num_objects, hidden_size]
        """

        #######################################################
        # 2. Encoder side
        with tf.variable_scope('encode_affordances'):
            # [batch_size * num_objects, hidden_size]
            gt_affordances_embed_encoder = embed_2d_with_embedding_table(object_states,
                                                                         embedding_table=self.affordance_embed_table,
                                                                         flatten=True)
            gt_affordances_embed_encoder = dropout(gt_affordances_embed_encoder, dropout_prob=self.dropout_prob)

            encoded_h = summarize_transformer(self.object_embed, gt_affordances_embed_encoder,
                                                      self.affordance_emb_ctx,
                                                      dropout_prob=self.dropout_prob)
            encoded_h = tf.layers.dense(encoded_h, self.hidden_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='final_proj_without_ln')

            encoded_h = tf.reshape(encoded_h, [self.batch_size, self.num_objects, self.hidden_size])
        return self.activation_fn(encoded_h)

    def encode_action(self, action_id, action_args):
        """
        Encode the action using a representation of IT as well as a representation of the embedded objects


        :param action_id: [batch_size]
        :param action_args: [batch_size, 2]
        :return: action embed [batch_size, hidden_size]
        """
        batch_size, two_ = get_shape_list(action_args, 2)
        assert two_ == 2
        assert batch_size == self.batch_size

        # Pre and post are the same so just extract pre, doesnt matter
        object_embeds = tf.reshape(self.object_embed, [self.batch_size, 2, 2, self.hidden_size])[:, :, 0]

        with tf.variable_scope('encode_action'):
            # Encode action
            action_embedding_table = tf.get_variable(
                name='action_embs',
                shape=[len(THOR_ACTION_TYPE_TO_IND), self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )
            self.action_embedding_table = action_embedding_table
            action_embed = embed_with_embedding_table(action_id, action_embedding_table)

            # I originally got action args from
            # action_args = []
            #         for k in ['object_name', 'receptacle_name']:
            #             ok = item['action'][k]
            #
            #             if ok is None:
            #                 action_args.append(0)
            #
            #             elif ok == item['pre'][0]['index']:
            #                 action_args.append(1)
            #             elif ok == item['pre'][1]['index']:
            #                 action_args.append(2)
            #             else:
            #                 import ipdb
            #                 ipdb.set_trace()
            nullctx = tf.tile(tf.get_variable(
                name='nullobj',
                shape=[self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )[None, None], [self.batch_size, 1, 1])

            encoded_pre_and_zero = tf.concat([nullctx, object_embeds], 1)
            object_repr_and_receptacle_repr = tf.gather(encoded_pre_and_zero, action_args, batch_dims=1)
            object_repr_and_receptacle_repr = tf.reshape(object_repr_and_receptacle_repr,
                                                         [self.batch_size, 2 * self.hidden_size])

            action_embed0 = tf.concat([action_embed, object_repr_and_receptacle_repr], 1)
            return self.activation_fn(residual_mlp(action_embed0,
                                hidden_size=self.hidden_size,
                                final_size=self.hidden_size,
                                num_layers=2,
                                hidden_dropout_prob=self.dropout_prob
                                ))

    def apply_action_mlp(self, action_embed, encoded_h_pre):
        """
        :param action_embed: [batch_size, h]
        :param encoded_h_pre: [batch_size, num_objs (probably 2), h] -- one per thing we will predict.


        We can model this JOINTLY or applying the model to EACH THING.

        :return:
        """
        batch_size, num_objs_to_apply, hidden_size = get_shape_list(encoded_h_pre, 3)
        assert batch_size == self.batch_size
        assert hidden_size == self.hidden_size
        if self.config.model.get('fuse_action', True):
            tf.logging.info("Apply action MLP -> Fuse action!")
            # 3. Change the hidden state
            with tf.variable_scope('apply_action_mlp'):

                mlp_h = tf.concat([action_embed, tf.reshape(encoded_h_pre, [self.batch_size, -1])], 1)
                encoded_h_post_pred = residual_mlp(mlp_h,
                                          initial_proj=False,
                                          num_layers=2,
                                          hidden_size=3*self.hidden_size,
                                          final_size=num_objs_to_apply * self.hidden_size,
                                          hidden_dropout_prob=self.dropout_prob)
                encoded_h_post_pred = tf.reshape(encoded_h_post_pred, [self.batch_size, num_objs_to_apply, self.hidden_size])

                return self.activation_fn(encoded_h_post_pred)
        else:
            # 3. Change the hidden state
            with tf.variable_scope('apply_action_mlp'):
                mlp_h = tf.concat([tf.tile(action_embed[:, None], [1, num_objs_to_apply, 1]), encoded_h_pre], 2)

                mlp_h_2d = tf.reshape(mlp_h, [self.batch_size * num_objs_to_apply, self.hidden_size + hidden_size])

                encoded_h_post_pred = residual_mlp(mlp_h_2d, hidden_size=self.hidden_size, final_size=self.hidden_size,
                                           hidden_dropout_prob=self.dropout_prob)
                encoded_h_post_pred = tf.reshape(encoded_h_post_pred, [self.batch_size, num_objs_to_apply, self.hidden_size])

                return self.activation_fn(encoded_h_post_pred)

    def decode_affordances_when_gt_is_provided(self, all_encoded_h, gt_affordances_decoded):
        """

        :param all_encoded_h: [batch_size, num_objs, hidden_size]
        :param gt_affordances_decoded: [batch_size, num_objs, num_afforadnces]
        :return: [batch_size, num_objs, num_affordances, vocab_size_for_affordances]
        """
        # 4. Predict the states!
        with tf.variable_scope('decoder'):
            batch_size, num_duplicates_x_num_objs, hidden_size = get_shape_list(all_encoded_h, 3)
            assert batch_size == self.batch_size
            # assert num_duplicates_x_num_objs == 6
            assert hidden_size == self.hidden_size

            batch_size_, num_duplicates_x_num_objs_, num_affordances = get_shape_list(gt_affordances_decoded, 3)
            assert num_duplicates_x_num_objs_ == num_duplicates_x_num_objs
            assert batch_size_ == self.batch_size

            all_encoded_h = dropout(tf.reshape(all_encoded_h, [-1, self.hidden_size]), dropout_prob=self.dropout_prob)

            # Get GT affordances -- slightly different because we duplicated the postconditions for 2 losses
            gt_affordances_decoder_embed = embed_2d_with_embedding_table(gt_affordances_decoded,
                                                                         self.affordance_embed_table,
                                                                         flatten=True)

            # [batch_size, num_affordances, hidden_size]
            hidden_states_per_attr, _ = expand_transformer(
                object_full_state=all_encoded_h,
                gt_affordances_embed=gt_affordances_decoder_embed,
                affordance_ctx_name_embed=self.affordance_emb_ctx,
                affordance_trg_name_embed=self.affordance_emb_trg,
                dropout_prob=self.dropout_prob,
                random_perms=self.is_training and self.config.data.get('random_perms', False),
            )
            # GET the predictions
            affordances_pred = tf.einsum('bah,avh->bav', hidden_states_per_attr, self.affordance_embed_table)
            apb = tf.get_variable(
                name='affordance_pred_bias',
                shape=[len(names_and_arities), len(THOR_OBJECT_TYPE_TO_IND)],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )
            affordances_pred += apb[None]
            affordance_pred_by_type = tf.reshape(affordances_pred,
                                                 [batch_size, num_duplicates_x_num_objs,
                                                  len(names_and_arities), len(THOR_OBJECT_TYPE_TO_IND)])
            return affordance_pred_by_type

    def sample_step(self, encoded_h_flat, prev_affordances=None, cache=None, p=0.95):
        """

        :param encoded_h_flat: [Batch_size * num_objs, hidden_size]
        :param prev_affordances: [batch_size * num_objs, num_affordances up until now (maybe None)?
        :param cache:
        :return:
        """
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            batch_size, hidden_size = get_shape_list(encoded_h_flat, 2)

            if prev_affordances is None:
                num_affordances_to_now = 0
                prev_affordances_embed = tf.zeros((batch_size, 0, self.hidden_size))
            else:
                batch_size, num_affordances_to_now = get_shape_list(prev_affordances, 2)

                prev_affordances_embed = embed_2d_with_embedding_table(prev_affordances,
                                                                       self.affordance_embed_table[
                                                                       :num_affordances_to_now],
                                                                       flatten=True)
            prev_affordances_embed = tf.concat([prev_affordances_embed, tf.zeros((batch_size, 1, self.hidden_size))], 1)

            hidden_states_per_attr, new_kvs = expand_transformer(
                object_full_state=encoded_h_flat,
                gt_affordances_embed=prev_affordances_embed,
                affordance_ctx_name_embed=self.affordance_emb_ctx[:num_affordances_to_now + 1],
                affordance_trg_name_embed=self.affordance_emb_trg[:num_affordances_to_now + 1],
                dropout_prob=self.dropout_prob,
                random_perms=False,
                reuse=tf.AUTO_REUSE,
                layer_cache=cache
            )

            logits = tf.einsum('bh,vh->bv', hidden_states_per_attr[:, -1],
                               self.affordance_embed_table[num_affordances_to_now])
            apb = tf.get_variable(
                name='affordance_pred_bias',
                shape=[len(names_and_arities), len(THOR_OBJECT_TYPE_TO_IND)],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
            )[num_affordances_to_now]
            logits += apb[None]

            cur_name, cur_arity = names_and_arities[num_affordances_to_now]
            logits_mask = tf.cast(tf.less(tf.range(len(THOR_OBJECT_TYPE_TO_IND)), max(cur_arity, 2)), dtype=tf.float32)
            logits = logits * logits_mask - 1e10 * (1.0 - logits_mask)

            # sample_info = _top_p_sample(logits, num_samples=1, p=p)
            sample_info = _argmax_sample(logits)
            new_tokens = tf.squeeze(sample_info['sample'], 1)
            new_probs = tf.squeeze(tf.batch_gather(sample_info['probs'], sample_info['sample']), 1)

        return {
            'new_tokens': new_tokens,
            'new_probs': new_probs,
            'new_cache': new_kvs
        }

    def sample(self, encoded_h):
        """
        Decode into actual affordances
        :param encoded_h: [batch_size, num_objects, hidden_size]
        :return:
        """
        bsize0, num_objs0, hidden_size = get_shape_list(encoded_h, 3)
        encoded_h_flat = tf.reshape(encoded_h, [-1, self.hidden_size])
        batch_size = get_shape_list(encoded_h_flat, 2)[0]
        with tf.name_scope('sample'):
            h0 = self.sample_step(encoded_h_flat)

            ctx = h0['new_tokens'][:, None]
            cache = h0['new_cache']
            probs = h0['new_probs'][:, None]

            # Technically we don't need tf.while_loop here bc always doing it for the same number of steps
            for t in range(len(names_and_arities) - 1):
                next_outputs = self.sample_step(encoded_h_flat, prev_affordances=ctx, cache=cache)
                # Update everything
                cache = tf.concat([cache, next_outputs['new_cache']], axis=-2)
                ctx = tf.concat([ctx, next_outputs['new_tokens'][:, None]], axis=1)
                probs = tf.concat([probs, next_outputs['new_probs'][:, None]], axis=1)

        return {
            'tokens': tf.reshape(ctx, [bsize0, num_objs0, -1]),
            'probs': tf.reshape(probs, [bsize0, num_objs0, -1]),
        }

    def compute_losses(self, object_states, isvalid_by_type_o1o2,
                       encoded_h_pre,
                       encoded_h_post_gt,
                       encoded_h_post_pred,
                       affordance_pred_by_type,
                       gt_affordances_decoder,
                       isvalid_by_type):
        """

        :param object_states: [batch_size, 4, len(names_and_arities)
        :param isvalid_by_type_o1o2: first two objs whteher they're valid [batch_size, 2]
        :return:
        """
        batch_size, num_duplicates_x_num_objs, nlen_names_and_arities = get_shape_list(object_states, 3)

        # MAGNITUDE LOSSES
        ###################
        # Check if anything changed
        norms = {}
        losses = {}
        pre_states, post_states = tf.unstack(
            tf.reshape(object_states, [batch_size, 2, 2, len(names_and_arities)]), axis=2)
        did_change = tf.not_equal(pre_states, post_states)
        didchange_weight = tf.cast(tf.reduce_any(did_change, -1), dtype=tf.float32) * isvalid_by_type_o1o2
        nochange_weight = (1.0 - tf.cast(tf.reduce_any(did_change, -1), dtype=tf.float32)) * isvalid_by_type_o1o2

        ### How much did things change
        ###############
        encoded_h_delta = encoded_h_post_pred - encoded_h_pre
        encoded_h_delta_l2 = tf.sqrt(tf.reduce_mean(tf.square(encoded_h_delta), -1))

        norms['didchange_hdelta_l2'] = tf.reduce_sum(encoded_h_delta_l2 * didchange_weight) / (tf.reduce_sum(
            didchange_weight) + 1e-5)
        norms['nochange_hdelta_l2'] = tf.reduce_sum(encoded_h_delta_l2 * nochange_weight) / (tf.reduce_sum(
            nochange_weight) + 1e-5)

        # Delta between pred and GT
        ###
        # gt_mu = tf.stop_gradient(encoded_h_post_gt[:, :, :self.hidden_size])
        # pred_mu = encoded_h_post_pred[:, :, :self.hidden_size]
        #
        # #########################################
        # # VAE loss
        # all_mu, all_logvar = tf.split(tf.reshape(tf.concat([encoded_h_pre,
        #                                                     encoded_h_post_gt,
        #                                                     encoded_h_post_pred], 1),
        #                                          [-1, self.hidden_size * 2]), [self.hidden_size, self.hidden_size],
        #                               axis=-1)
        # kld = -0.5 * tf.reduce_mean(1.0 + all_logvar - tf.square(all_mu) - tf.exp(all_logvar))
        # losses['kld'] = kld

        #########################################
        gt_stop = tf.stop_gradient(encoded_h_post_gt)
        hidden_state_diff_l2 = tf.sqrt(tf.reduce_mean(tf.square(encoded_h_post_pred - gt_stop), -1))
        hidden_state_diff_l1 = tf.reduce_mean(tf.abs(encoded_h_post_pred - gt_stop), -1)
        norms['hidden_state_diff_l2'] = tf.reduce_sum(hidden_state_diff_l2 * isvalid_by_type_o1o2) / (
                tf.reduce_sum(isvalid_by_type_o1o2) + 1e-5)
        norms['hidden_state_diff_l1'] = tf.reduce_sum(hidden_state_diff_l1 * isvalid_by_type_o1o2) / (
                tf.reduce_sum(isvalid_by_type_o1o2) + 1e-5)

        hidden_state_magn_l2 = tf.sqrt(tf.reduce_mean(tf.square(gt_stop), -1))
        norms['hidden_state_magn_l2'] = tf.reduce_sum(hidden_state_magn_l2 * isvalid_by_type_o1o2) / (
                tf.reduce_sum(isvalid_by_type_o1o2) + 1e-5)

        # Upweight changed losses
        # did change: [batch_size, num_objs, num_affordances]
        for i, (affordance_name, arity_) in enumerate(names_and_arities):
            arity = max(arity_, 2)

            losses[f'state/{affordance_name}_post'] = sequence_xe_loss(
                affordance_pred_by_type[:, 4:, i, :arity],
                gt_affordances_decoder[:, 4:, i],
                label_weights=isvalid_by_type[:, 4:],
            )

            losses[f'state/{affordance_name}_pre'] = sequence_xe_loss(
                affordance_pred_by_type[:, 0:2, i, :arity],
                gt_affordances_decoder[:, 0:2, i],
                label_weights=isvalid_by_type[:, 0:2],  # + tf.cast(did_change[:, :, i], dtype=tf.float32) * 100.0,
            )

            losses[f'state/{affordance_name}_postgt'] = sequence_xe_loss(
                affordance_pred_by_type[:, 2:4, i, :arity],
                gt_affordances_decoder[:, 2:4, i],
                label_weights=isvalid_by_type[:, 2:4],  # + tf.cast(did_change[:, :, i], dtype=tf.float32) * 100.0,
            )
        # # Another way for losses
        # losses_all = _sequence_xe_loss_noreduce(affordance_pred_by_type, gt_affordances_decoder)
        # loss_mask = tf.reshape(tf.tile(isvalid_by_type[:, :, None], [1, 1, len(names_and_arities)]), [-1])
        # losses['state/all'] = tf.reduce_sum(losses_all * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-5)

        return losses, norms

def model_fn_builder(config: NeatConfig):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        batch_size = get_shape_list(features['actions/action_id'], expected_rank=1)[0]
        hidden_size = config.model['hidden_size']
        # activation_fn = tf.nn.tanh if config.model.get('activation', 'tanh') == 'tanh' else tf.identity

        scp_model = StateChangePredictModel(config,
                                            is_training=is_training,
                                            object_types=features['objects/object_types'],
                                            )

        encoded_h = scp_model.encode_affordances(features['objects/object_states'])

        encoded_h_pre = tf.gather(encoded_h, [0, 2], axis=1)
        encoded_h_post_gt = tf.gather(encoded_h, [1, 3], axis=1)

        action_embed = scp_model.encode_action(features['actions/action_id'], action_args=features['actions/action_args'])

        encoded_h_post_pred = scp_model.apply_action_mlp(action_embed, encoded_h_pre)

        #############################################################

        # Now construct a decoder
        # [batch_size, 3, #objs, hidden_size] -> [batch_size, 3 * objs, hidden_size]
        all_encoded_h = tf.concat([
            encoded_h_pre, # [0, 2]
            encoded_h_post_gt, # [1, 3]
            encoded_h_post_pred, # [1, 3]
        ], 1)

        gt_affordances_decoder = tf.gather(features['objects/object_states'], [0, 2, 1, 3, 1, 3], axis=1)
        isvalid_by_type = tf.cast(tf.gather(features['objects/is_valid'], [0, 2, 1, 3, 1, 3], axis=1), dtype=tf.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = scp_model.sample(all_encoded_h)
            predictions.update(**features)
            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                                   predictions=predictions)
        affordance_pred_by_type = scp_model.decode_affordances_when_gt_is_provided(all_encoded_h,
                                                                                   gt_affordances_decoder)

        ######################
        # For losses
        # action_logits = action_result['action_logits']

        ############################################
        # if params.get('demomode', False):
        #     action_logits['affordances_pred'] = affordance_pred_by_type[:, 4:]
        #     for k in action_logits:
        #         action_logits[k] = tf.nn.softmax(action_logits[k], axis=-1)
        #     return action_logits

        losses, norms = scp_model.compute_losses(
            object_states=features['objects/object_states'],
            isvalid_by_type_o1o2=isvalid_by_type[:, :2],
            encoded_h_pre=encoded_h_pre,
            encoded_h_post_gt=encoded_h_post_gt,
            encoded_h_post_pred=encoded_h_post_pred,
            affordance_pred_by_type=affordance_pred_by_type,
            gt_affordances_decoder=gt_affordances_decoder,
            isvalid_by_type=isvalid_by_type)
        # losses['action_success'] = sequence_xe_loss(action_logits['action_success'], features['actions/action_success'])

        loss = tf.add_n([x for x in losses.values()])

        for k, v in norms.items():
            losses[f'norms/{k}'] = v
        loss += 0.1 * norms['hidden_state_diff_l2']
        loss += 0.1 * norms['hidden_state_diff_l1']

        if is_training:
            tvars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'global_step' not in x.name]
        else:
            tvars = tf.trainable_variables()

        # ckpt_to_assignment_map = {}
        # initialized_variable_names = {}

        # init_checkpoint = config.model.get('init_checkpoint', None)
        # if init_checkpoint:
        #     regular_assignment_map, regular_initialized_variable_names = get_assignment_map_from_checkpoint(
        #         tvars, init_checkpoint=init_checkpoint
        #     )
        #
        #     # If you need to disable loading certain variables, comment something like this in
        #     # regular_assignment_map = {k: v for k, v in regular_assignment_map.items() if
        #     #                           all([x not in k for x in ('temporal_predict',
        #     #                                                     'roi_language_predict',
        #     #                                                     'roi_pool/pool_c5',
        #     #                                                     'aux_roi',
        #     #                                                     'second_fpn',
        #     #                                                     'img_mask',
        #     #                                                     'roi_pool/box_feats_proj/kernel')])}
        #
        #     ckpt_to_assignment_map['regular'] = regular_assignment_map
        #     initialized_variable_names.update(regular_initialized_variable_names)
        #
        # def scaffold_fn():
        #     """Loads pretrained model through scaffold function."""
        #     # ORDER BY PRIORITY
        #     return tf.train.Scaffold()

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            # if var.name in initialized_variable_names:
            #     init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        train_op, train_metrics = optimization.build_optimizer_from_config(
            loss=loss,
            optimizer_config=config.optimizer,
            device_config=config.device,
        )
        train_metrics.update(losses)
        # for k, v in affordance_loss_metrics.items():
        #     train_metrics[f'affordance_metrics/{k}'] = v

        host_call = construct_host_call(scalars_to_log=train_metrics,
                                        model_dir=config.device['output_dir'],
                                        iterations_per_loop=config.device.get('iterations_per_loop', 1000))
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=None,
            # scaffold_fn=scaffold_fn,
            host_call=host_call)

    return model_fn


if __name__ == '__main__':
    from model.interact import dataloader

    tf.compat.v1.enable_eager_execution()
    batch_size = 8
    config = NeatConfig.from_yaml('configs/local_debug.yaml')
    input_fn = dataloader.input_fn_builder(config, is_training=True)
    features, labels = input_fn(params={'batch_size': batch_size}).make_one_shot_iterator().get_next()
    lol = model_fn_builder(config)(features, labels, tf.estimator.ModeKeys.TRAIN, {'batch_size': batch_size})
    # model = TrajectoryMLP(is_training=True,
    #                       features=features,
    #                       hidden_size=config.model['hidden_size'],
    #                       )

