""" Training script! """
import sys
sys.path.append('../')
import tensorflow as tf

from model.neat_config import NeatConfig
from model.interact.modeling import model_fn_builder
from model.interact.dataloader import input_fn_builder

config = NeatConfig.from_args("Train detector script", default_config_file='interact/configs/default_tpu.yaml')
model_fn = model_fn_builder(config)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
    # params={},
)

estimator.train(input_fn=input_fn_builder(config, is_training=True),
                max_steps=config.optimizer['num_train_steps'])