"""
Our model finetune
"""
import sys

sys.path.append('../../')
import os
import numpy as np

from model.lm.modeling import GroverModel, LMConfig, create_initializer, _top_p_sample
from model.model_utils import dropout, get_assignment_map_from_checkpoint, construct_host_call

from model.neat_config import NeatConfig
from model import optimization
import tensorflow as tf
import json

from model.model_utils import pad_to_fixed_size as p2fsz
from model.model_utils import get_shape_list, switch_block
import pandas as pd
from data.zeroshot_lm_setup.encoder import get_encoder, Encoder
from tfrecord.tfrecord_utils import S3TFRecordWriter, int64_list_feature, int64_feature
from data.concept_utils import get_concepts_from_bpe_encoded_text, concept_df, mapping_dict, get_glove_embeds
from model.predict_statechange.finetune_dataloader import get_statechange_dataset, create_tfrecord_statechange, \
    evaluate_statechange
from model.interact.dataloader import input_fn_builder
from model.interact.modeling import StateChangePredictModel
from data.thor_constants import numpy_to_instance_states
from model.transformer import residual_mlp

config = NeatConfig.from_args("Finetune script",
                              default_config_file='configs/local_debug.yaml',
                              extra_args=[{
                                  'name': '--do_train',
                                  'dest': 'do_train',
                                  'action': 'store_true', 'default': False,
                              }])


def finetune_model_fn_builder_statechange(config: NeatConfig):
    """Returns `model_fn` closure for TPUEstimator."""

    lm_config = LMConfig.from_size(config.model['lm_size'])

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        ###################################
        # encode language
        batch_size, seq_length = get_shape_list(features['ids/post_act2pre'], expected_rank=2)
        hidden_size = config.model['hidden_size']
        activation_fn = tf.nn.tanh if config.model.get('activation', 'tanh') == 'tanh' else tf.identity
        dropout_prob = config.model.get('dropout_prob', 0.1) if is_training else 0.0

        is_bidirectional = tf.cast(tf.greater(features['ids/pre_act2post'], 0), dtype=tf.int32)

        encoder_model = GroverModel(
            config=lm_config,
            is_training=is_training,
            input_ids=features['ids/pre_act2post'],
            is_bidirectional=is_bidirectional,
            pad_token_id=lm_config.pad_token_id,
            scope='basiclm',
        )


        ##################################################################
        # encode and handle ll stuff
        scp_model = StateChangePredictModel(config=config,
                                            is_training=is_training,
                                            object_types=features['objects/object_types'])
        encoded_h = scp_model.encode_affordances(features['objects/object_states'])
        encoded_h_pre = tf.gather(encoded_h, [0, 2], axis=1)
        encoded_h_post_gt = tf.gather(encoded_h, [1, 3], axis=1)

        # This will be used for the action representation.

        symb_action_embed = scp_model.encode_action(features['actions/action_id'],
                                                    action_args=features['actions/action_args'])
        with tf.variable_scope('encoded_hl_pseudoactionembed'):
            lang_action_embed = tf.layers.dense(
                encoder_model.global_hidden_state,
                hidden_size,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                name='lang_to_action_embed',
                activation=activation_fn,
            )

            if is_training:
                action_embed = switch_block(x_a=symb_action_embed, x_b=lang_action_embed,
                                            choose_a_prob=config.model['train_time_symbolic_actions_prob'])
            else:
                action_embed = symb_action_embed if config.model['test_time_use_symbolic_actions'] else lang_action_embed

        # TODO: Add another switch_block for encoded_h_pre.

        # Apply the action
        encoded_h_post_pred = scp_model.apply_action_mlp(action_embed, encoded_h_pre)

        # Now DECODE into the attr space
        all_encoded_h = tf.concat([
            encoded_h_pre,  # [0, 2]
            encoded_h_post_gt,  # [1, 3]
            encoded_h_post_pred,  # [1, 3]
        ], 1)

        gt_affordances_decoder = tf.gather(features['objects/object_states'], [0, 2, 1, 3, 1, 3], axis=1)
        isvalid_by_type = tf.cast(tf.gather(features['objects/is_valid'], [0, 2, 1, 3, 1, 3], axis=1),
                                  dtype=tf.float32)

        if config.model.get('generate', False):
            tf.logging.info("GENERATE")
            # DECODE
            dummy_start = tf.ones([batch_size, 1], dtype=tf.int32)

            # Pre, post_gt, post_pred.
            output_ids = tf.stack([
                tf.concat([dummy_start, features['ids/pre'][:, :-1]], 1),
                tf.concat([dummy_start, features['ids/post'][:, :-1]], 1),
                tf.concat([dummy_start, features['ids/post'][:, :-1]], 1),
            ], 1)

            output_ids_gt = tf.stack([
                features['ids/pre'],
                features['ids/post'],
                features['ids/post'],
            ], 1)

            h_for_decoder = tf.stack([
                tf.reshape(encoded_h_pre, [batch_size, -1]),
                tf.reshape(encoded_h_post_gt, [batch_size, -1]),
                tf.reshape(encoded_h_post_pred, [batch_size, -1]),
            ], 1)
            h_for_decoder = tf.layers.dense(
                dropout(h_for_decoder, dropout_prob=dropout_prob),
                lm_config.hidden_size,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='generation_hstate_ctx',
            )
            decoder = GroverModel(
                config=lm_config,
                is_training=is_training,
                input_ids=tf.reshape(output_ids, [batch_size * 3, seq_length]),
                is_bidirectional=tf.zeros([batch_size * 3, seq_length], dtype=tf.int32),
                pad_token_id=lm_config.pad_token_id,
                reuse=True,
                hidden_state_for_conditional_ltr=tf.reshape(h_for_decoder, [batch_size * 3, 1, lm_config.hidden_size]),
                scope='basiclm',
            )
            lm_loss, lm_loss_diagnostics, per_example_loss = decoder.lm_loss(target_ids=output_ids_gt, return_per_example_loss=True)
        else:
            lm_loss = None
            h_for_decoder=None
            per_example_loss = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = scp_model.sample(all_encoded_h)
            predictions.update(**features)

            if config.model.get('generate', False):
                # TOP_P=0.94
                # # Crude TPU sample
                # is_bidirectional = tf.zeros([batch_size*3, 1], dtype=tf.int32)
                # ctx0 = GroverModel(
                #     config=lm_config,
                #     is_training=False,
                #     input_ids=tf.ones([batch_size*3, 1], dtype=tf.int32),
                #     is_bidirectional=is_bidirectional,  # this doesnt matter
                #     do_cache=True,
                #     scope='basiclm',
                #     reuse=True,
                #     hidden_state_for_conditional_ltr=tf.reshape(h_for_decoder, [batch_size * 3, 1, lm_config.hidden_size]),
                # )
                # # SAMPLE
                # cache = ctx0.new_kvs
                # sample_info = _top_p_sample(ctx0.logits_flat, num_samples=1, p=TOP_P)
                # # logits = ctx0.logits_flat
                # probs = tf.gather(sample_info['probs'], sample_info['sample'], batch_dims=1)
                # tokens = sample_info['sample']
                #
                # for i in range(50):
                #     tf.logging.info(f"t={i}")
                #     ctx_t = GroverModel(
                #         config=lm_config,
                #         is_training=False,
                #         input_ids=tokens[:, -1, None],
                #         is_bidirectional=is_bidirectional,
                #         do_cache=True,
                #         scope='basiclm',
                #         reuse=True,
                #         cache=cache,
                #     )
                #     cache = tf.concat([cache, ctx_t.new_kvs], axis=-2)
                #     sample_info = _top_p_sample(ctx_t.logits_flat, num_samples=1, p=TOP_P)
                #     # logits = ctx0.logits_flat
                #     new_probs = tf.gather(sample_info['probs'], sample_info['sample'], batch_dims=1)
                #     probs = tf.concat([probs, new_probs], 1)
                #     tokens = tf.concat([tokens, sample_info['sample']], 1)
                # predictions['gen_tokens'] = tf.reshape(tokens, [batch_size, 3, -1])
                # predictions['gen_probs'] = tf.reshape(probs, [batch_size, 3, -1])
                predictions['per_example_neglogprob'] = tf.gather(tf.reshape(per_example_loss, [batch_size, 3, -1]), [0, 2], axis=1)

            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                                   predictions=predictions)

        affordance_pred_by_type = scp_model.decode_affordances_when_gt_is_provided(all_encoded_h,
                                                                                   gt_affordances_decoder)

        losses, norms = scp_model.compute_losses(
            object_states=features['objects/object_states'],
            isvalid_by_type_o1o2=isvalid_by_type[:, :2],
            encoded_h_pre=tf.stop_gradient(encoded_h_pre),
            encoded_h_post_gt=tf.stop_gradient(encoded_h_post_gt),
            encoded_h_post_pred=tf.stop_gradient(encoded_h_post_pred),
            affordance_pred_by_type=affordance_pred_by_type,
            gt_affordances_decoder=gt_affordances_decoder,
            isvalid_by_type=isvalid_by_type)

        if lm_loss is not None:
            losses['lm'] = lm_loss * config.model.get('lm_loss_coef', 1.0)

        loss = tf.add_n([x for x in losses.values()])

        for k, v in norms.items():
            losses[f'norms/{k}'] = v
        # loss += 0.1 * norms['hidden_state_diff_l2']
        # loss += 0.1 * norms['hidden_state_diff_l1']

        if is_training:
            tvars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'global_step' not in x.name]
        else:
            tvars = tf.trainable_variables()

        lm_init_checkpoint = config.model['init_checkpoint']
        lm_assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
            tvars, init_checkpoint=lm_init_checkpoint
        )

        interact_init_checkpoint = config.model.get('interact_checkpoint', None)
        if interact_init_checkpoint is not None:
            interact_assignment_map, interact_initialized_variable_names = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint=interact_init_checkpoint
            )
            initialized_variable_names.update(**interact_initialized_variable_names)

        def scaffold_fn():
            """Loads pretrained model through scaffold function."""
            tf.train.init_from_checkpoint(lm_init_checkpoint, lm_assignment_map)
            if interact_init_checkpoint is not None:
                tf.train.init_from_checkpoint(interact_init_checkpoint, interact_assignment_map)
            return tf.train.Scaffold()

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        train_op, train_metrics = optimization.build_optimizer_from_config(
            loss=loss,
            optimizer_config=config.optimizer,
            device_config=config.device,
        )
        train_metrics.update(losses)

        host_call = construct_host_call(scalars_to_log=train_metrics,
                                        model_dir=config.device['output_dir'],
                                        iterations_per_loop=config.device.get('iterations_per_loop', 1000))
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=None,
            scaffold_fn=scaffold_fn,
            host_call=host_call)

    return model_fn


####################
encoder = get_encoder()
all_lens = []
all_dsets = get_statechange_dataset()

for split, dataset_items in all_dsets.items():
    lens = create_tfrecord_statechange(encoder=encoder, dataset_items=dataset_items,
                                       out_fn=os.path.join(config.device['output_dir'], f'langdset-{split}.tfrecord'),
                                       pad_interval=1 if split == 'train' else config.device['val_batch_size'],
                                       include_precondition=True,
                                       include_action=True,
                                       include_postcondition=False)
    all_lens.extend(lens)
    if split == 'train':
        config.optimizer['num_train_steps'] = config.optimizer['num_epochs'] * len(lens) // config.device[
            'train_batch_size']
        config.optimizer['num_warmup_steps'] = int(
            config.optimizer['warmup_perc'] * config.optimizer['num_train_steps'])
        tf.logging.info("optimizing with \n{}".format('\n'.join(f'{k} : {v}' for k, v in config.optimizer.items())))

pad_len = max(all_lens)
tf.logging.info(f"USING MAX LEN = {pad_len}")
config.data['max_lang_seq_length'] = pad_len

model_fn = finetune_model_fn_builder_statechange(config)

if os.uname()[1] == 'shoob':
    tf.logging.info("Eager mode")
    tf.compat.v1.enable_eager_execution()
    batch_size = 8
    config.data['train_file'] = os.path.join(config.device['output_dir'], 'langdset-train.tfrecord')
    input_fn = input_fn_builder(config, is_training=True)
    features, labels = input_fn(params={'batch_size': batch_size}).make_one_shot_iterator().get_next()
    lol = model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, {'batch_size': batch_size})
    import ipdb
    ipdb.set_trace()

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
    # params={},
)

# Train the model!
###########################
ckpt_fn = os.path.join(config.device['output_dir'], 'model.ckpt-{}.index'.format(config.optimizer['num_train_steps']))
if not tf.io.gfile.exists(ckpt_fn):
    config.data['train_file'] = os.path.join(config.device['output_dir'], 'langdset-train.tfrecord')
    input_fn = input_fn_builder(config, is_training=True)
    estimator.train(input_fn=input_fn,
                    max_steps=config.optimizer['num_train_steps'])
else:
    tf.logging.info(f"Skipping train bc file exists: {ckpt_fn}")

evaluate_statechange(estimator, config)
#
# results = {}
# for split in ['val', 'test', 'train']:
#     config.data['val_file'] = os.path.join(config.device['output_dir'], f'{split}.tfrecord')
#     input_fn = input_fn_builder(config, is_training=False)
#     result = [x for x in estimator.predict(input_fn=input_fn) if (x['is_real_example'] == 1)]
#
#     # Eval the model
#     pred_states = []
#     gt_states = []
#     for res in result:
#         pred_state = numpy_to_instance_states(object_types=res['objects/object_types'][[0,2]],
#                                               object_states=res['tokens'][-2:])
#         gt_state = numpy_to_instance_states(object_types=res['objects/object_types'][[0,2]],
#                                               object_states=res['objects/object_states'][[1, 3]])
#
#         for obj_id in range(2):
#             ps_obj = {k[:-1]: v for k, v in pred_state.items() if k.endswith(str(obj_id))}
#             gs_obj = {k[:-1]: v for k, v in gt_state.items() if k.endswith(str(obj_id))}
#             ps_obj['object_ind'] = obj_id
#             gs_obj['object_ind'] = obj_id
#             if gs_obj['ObjectName'] != 'None':
#                 pred_states.append(ps_obj)
#                 gt_states.append(gs_obj)
#
#
#     results[f'{split}_pred'] = pd.DataFrame(pred_states)
#     results[f'{split}_gt'] = pd.DataFrame(gt_states)
#     agg_res = (results[f'{split}_pred'] == results[f'{split}_gt']).mean(0)
#
#     agg_res['all'] = (results[f'{split}_pred'] == results[f'{split}_gt']).all(1).mean()
#     print("Results {} {}".format(split, agg_res), flush=True)
#
# # Get accuracy per column
