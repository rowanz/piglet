"""
Just trying this out. essentially this utility is going to force you to pass in a config file whenever you do anything

including debugging

Essentially there are 4 main components: data, model, optimizer, and device. (For other things we have a 'misc') option.

"""
import argparse
import inspect
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import pytz
import tensorflow as tf
import yaml


def _get_calling_filename():
    """
    Finds the first filename that called this module, that's not this filename.
    :return:
    """
    for calling_frame in inspect.stack()[1:]:
        calling_module = inspect.getmodule(calling_frame[0])
        if calling_module is not None:
            calling_filename_without_py = calling_module.__file__.replace('/','~')[:-3]
            if not calling_filename_without_py.endswith('neat_config'):
                return calling_filename_without_py
    return None


class NeatConfig(object):
    def __init__(self):
        self.data = {}
        self.model = {}
        self.optimizer = {}
        self.device = {}
        self.downstream = {}
        self.validate = {}
        self.misc = {}

    @classmethod
    def from_yaml(cls, config_file, disable_tpu=False):
        """
        Sets up config from a yaml file
        :param config_file: where 2 load from
        :param disable_tpu: whether to disable TPU mode
        :return:
        """
        with tf.gfile.Open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        print("~~~~~\nLOADED CONFIG FROM {}\n~~~~~\n".format(config_file), flush=True)
        return cls.from_dict(config_dict, orig_config_file=config_file, disable_tpu=disable_tpu)

    @classmethod
    def from_dict(cls, config_dict, orig_config_file=None, disable_tpu=False):
        """
        Loads from dict
        :param config_dict:
        :param orig_config_file: Where it was originally loaded from, if we want to print a helpful message
        :param disable_tpu: If we want to NOT use disable
        :return:
        """
        config = deepcopy(config_dict)
        if 'misc' not in config:
            config['misc'] = {}
        # Mandatory keys
        for key in ['data', 'model', 'optimizer', 'device']:
            if key not in config:
                raise ValueError("Configuration file {} is missing {}".format(orig_config_file, key))

        if disable_tpu:
            print("DISABLING TPU\n\n", flush=True)
            if config['device']['use_tpu']:
                config['device'] = {'use_tpu': False, 'output_dir': 'tmp/', 'train_batch_size': 1, 'val_batch_size': 1}
            config_cls = cls()
            config_cls.__dict__.update(config)
            return config_cls

        # Save config to the cloud, if the output directory is there
        if 'output_dir' not in config['device']:
            raise ValueError("Missing output directory")
        print("~~~\nWILL WRITE TO {}\n~~~\n".format(config['device']['output_dir']), flush=True)

        calling_file = _get_calling_filename()

        # Write to the server something like 'myscript_config.yaml
        path_to_write = 'config.yaml' if orig_config_file is None else orig_config_file.split('/')[-1]
        if calling_file is not None:
            path_to_write = f'{calling_file}_{path_to_write}'
        with tf.io.gfile.GFile(os.path.join(config['device']['output_dir'], path_to_write), 'w') as f:
            config['misc']['config_name'] = orig_config_file

            seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
            config['misc']['time'] = seattle_time.strftime("%Y-%m-%d %H:%M:%S")
            config['misc']['config_version'] = 'Sept 13, 2019'
            yaml.dump(config, f, default_flow_style=False)

        # Special handling of TPUs. I think this should work on GPUs too but haven't checked
        # if config['device']['use_tpu']:
        config['device']['tpu_run_config'] = get_tpu_run_config(config['device'])

        if config['misc'].get('verbose', True):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        # Special handling of globs in the data files
        for x in ['train_file', 'val_file', 'test_file']:
            if x in config['data']:
                v_orig = config['data'][x]

                v_list = []
                prefix_to_suffixes = defaultdict(list)
                for input_pattern in config['data'][x].split(','):
                    input_pattern_raw = input_pattern.rstrip('*')
                    for fn in tf.io.gfile.glob(input_pattern):
                        v_list.append(fn)
                        if fn.startswith(input_pattern_raw):
                            prefix_to_suffixes[input_pattern_raw].append(fn[len(input_pattern_raw):])

                if sum(len(v) for v in prefix_to_suffixes.values()) == len(v_list):
                    # Get a friendly printout
                    input_file_tree = []
                    for prefix, suffix_list in prefix_to_suffixes.items():
                        input_file_tree.append('{}[{}]'.format(prefix, ','.join(suffix_list)))
                    input_file_tree = '\n'.join(input_file_tree)

                    # These can get kinda long and unwieldy
                    if len(input_file_tree) > 100:
                        input_file_tree = input_file_tree[:100] + '...({} total)'.format(len(v_list))

                    print("Input files: {} ->\n{}\n\n".format(v_orig, input_file_tree), flush=True)
                else:
                    print("Input files: {} ->\n{}\n\n".format(v_orig, '   '.join(v_list)), flush=True)

                config['data'][f'{x}_expanded'] = v_list

        config_cls = cls()
        config_cls.__dict__.update(config)
        return config_cls

    @classmethod
    def from_args(cls, help_message="NeatConfig", default_config_file=None, extra_args=(),):
        parser = argparse.ArgumentParser(description=help_message)
        parser.add_argument(
            'config_file',
            nargs='?',
            help='Where the config.yaml is located',
            default=default_config_file,
            type=str,
        )
        for item in extra_args:
            item2 = {k: v for k, v in item.items()}
            name = item2.pop('name')
            parser.add_argument(name, **item2)
        args = parser.parse_args()
        if not args.config_file:
            raise ValueError("No config file provided!")

        if not tf.io.gfile.exists(args.config_file):
            raise ValueError("Config file {} not found?".format(args.config_file))
        out_cls = cls.from_yaml(args.config_file)
        for k, v in vars(args).items():
            out_cls.misc[k] = v
        return out_cls


def get_tpu_run_config(device_config):
    """
    Sets up the tpu
    :param device_config: The part of the config file that is
    :return:
    """
    tpu_cluster_resolver = None
    tpu_name = device_config.get('tpu_name', os.uname()[1])  # This is the hostname

    if device_config['use_tpu']:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=device_config.get('tpu_zone', None), project=device_config.get('gcp_project', None))
        tf.compat.v1.Session.reset(tpu_cluster_resolver.get_master())

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=device_config.get('master', None),
        model_dir=device_config['output_dir'],
        save_checkpoints_steps=device_config.get('iterations_per_loop', 1000),
        keep_checkpoint_max=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=device_config.get('iterations_per_loop', 1000),
            # num_shards=device_config.get('num_tpu_cores', 8), # Commented out because it's always 8 for testing.
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    if tf.io.gfile.exists(device_config['output_dir']):
        print(f"The output directory {device_config['output_dir']} exists!")
    return run_config
