"""
Intrinsic evaluation on the test set
"""

import sys

sys.path.append('../../')
import os
import numpy as np

from model.lm.modeling import GroverModel, LMConfig, create_initializer
from model.model_utils import dropout, get_assignment_map_from_checkpoint, construct_host_call, basic_mlp

from model.neat_config import NeatConfig
from model import optimization
import tensorflow as tf
import json

from model.model_utils import pad_to_fixed_size as p2fsz
from model.model_utils import get_shape_list
import pandas as pd
from data.zeroshot_lm_setup.encoder import get_encoder, Encoder
from tfrecord.tfrecord_utils import S3TFRecordWriter, int64_list_feature, int64_feature
from data.concept_utils import get_concepts_from_bpe_encoded_text, concept_df, mapping_dict, get_glove_embeds
from model.predict_statechange.finetune_dataloader import get_statechange_dataset, create_tfrecord_statechange
from model.interact.dataloader import input_fn_builder
from model.interact.modeling import StateChangePredictModel
from data.thor_constants import numpy_to_instance_states
from model.neat_config import NeatConfig
from model.interact.dataloader import input_fn_builder
from model.interact.modeling import model_fn_builder
from model.predict_statechange.finetune_dataloader import evaluate_statechange

config = NeatConfig.from_args("Train detector script", default_config_file='configs/jan12-basic1.yaml')
model_fn = model_fn_builder(config)

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
    # if split == 'train':
    #     config.optimizer['num_train_steps'] = config.optimizer['num_epochs'] * len(lens) // config.device[
    #         'train_batch_size']
    #     config.optimizer['num_warmup_steps'] = int(
    #         config.optimizer['warmup_perc'] * config.optimizer['num_train_steps'])
    #     tf.logging.info("optimizing with \n{}".format('\n'.join(f'{k} : {v}' for k, v in config.optimizer.items())))

pad_len = max(all_lens)
tf.logging.info(f"USING MAX LEN = {pad_len}")
config.data['max_lang_seq_length'] = pad_len

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
    # params={},
)

evaluate_statechange(estimator, config)
