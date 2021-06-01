import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import tensorflow as tf

from model.model_utils import get_shape_list


def build_optimizer_from_config(loss, optimizer_config, device_config=None):
    """
    This is a utility to build an optimizer from optimizer_config.

    :param loss: what to use.
    :param optimizer_config: k/v of options
    :param device_config: Additional options that can be rolled in
    :return: An optimizer
    """
    optimizer_types = {
        'retinanet_momentum_optimizer': retinanet_momentum_optimizer,
        'adam_optimizer': create_fixed_adam_optimizer_with_warmup,
    }
    if optimizer_config['type'] not in optimizer_types:
        raise ValueError("The optimizer type {} isn't supported".format(optimizer_config['type']))

    kwargs = deepcopy(optimizer_config)
    if device_config is not None:
        kwargs.update(deepcopy(device_config))
    del kwargs['type']
    return optimizer_types[optimizer_config['type']](loss, **kwargs)


def _print_var_list_for_debugging(var_list):
    """
    For debugging, print a list of vars. Sort by the shapes, also print the total size.
    :param var_list: list of vars.
    :return: Nothing!
    """
    if len(var_list) == 0:
        tf.logging.info('~~~ (N/A) ~~~')
        return
    sorted_vars = sorted([(_get_variable_name(x.name), tuple(get_shape_list(x))) for x in var_list],
                         key=lambda x: -np.prod(x[1]))
    total_size = sum([np.prod(x[1]) for x in sorted_vars])
    # Pretty print each line
    longest_name = max([len(x[0]) for x in sorted_vars])
    prints = [' {s:<{w}}'.format(s=x[0], w=longest_name) + '{}'.format(x[1]) for x in sorted_vars]
    for l in prints:
        tf.logging.info(l)
    tf.logging.info('~~~~ Total size = {} or {:.1f}M\n'.format(
        total_size, float(total_size) / 1000000.0
    ))


def create_fixed_adam_optimizer_with_warmup(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                                            weight_decay_rate=1e-4, param_overrides=None, freeze_scope=None,
                                            verbose=False, clip_norm=1.0, adafactor=False, epsilon=1e-6, beta_2=0.999,
                                            **kwargs):
    """
    Does AdamW optimization. Unlike the BERT optimizer, here I added bias correct which the original
    one didn't seem to have.

    :param loss:
    :param learning_rate: The default learning rate we'll use. All of the learning rates, including overridden ones
                          will get scaled during the initial `num_warmup_steps`.
    :param num_train_steps: How many steps to train for overall.
    :param num_warmup_steps: A number, presumably < num_train_steps which specifies for how long we warmup.
    :param use_tpu: Whether to use TPU. This is important because we need to duplicate the optimizer accross shards.
    :param weight_decay_rate: How much to decay the weights by default.
    :param param_overrides: Which parameters to override. This works like the following. You pass in a
                            LIST of LIST, DICTIONARY pairs. Each pair consists of a bunch of regular expressions
                            and if one of those are activated, we will override the default parameters in that instance.
                            For instance

                            ["LayerNorm", "layer_norm", 'GroupNorm', "bias"], {"weight_decay_rate": 0}

                            will set any parameter matching the first couple of regexes to have weight_decay_rate of 0.
    :param freeze_scope: OLD deprecated parameter that sets anything matching ["^freeze_scope/"] to have {"learning_rate": 0}
    :param verbose: Use this for extra debugging output
    :param kwargs: extra args, not needed
    :return:
    """

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Implements linear decay of the learning rate. This does it globally over all parameters
    # which should be OK.

    # Make it so that we scale the loss UP to learning_rate
    # scale * (1-(num_warmup_steps / num_train_steps)) = 1.0
    # scale = 1/(1-(num_warmup_steps / num_train_steps))
    # scale = num_train_steps /(num_train_steps - num_warmup_steps
    base_scale = float(num_train_steps) / (
                float(num_train_steps) - float(num_warmup_steps) + 1.0) if num_warmup_steps else 1.0
    learning_rate_scale = tf.compat.v1.train.polynomial_decay(
        tf.constant(value=base_scale, shape=[], dtype=tf.float32),
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * learning_rate`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float

        learning_rate_scale = tf.where(global_steps_int < warmup_steps_int, warmup_percent_done, learning_rate_scale)

    # Deal with the parameter overrides.
    # We can override:
    #     learning_rate. if learning_rate = 0 then we aren't training it at all.
    #     beta_1
    #     beta_2
    #     epsilon
    #     weight_decay_rate

    if param_overrides is None:
        param_overrides = []

    if freeze_scope is not None:
        print("NOTE! freeze_scope is deprecated. You can do the exact same thing by instead setting\n"
              "param_overrides: [[[\"^{}\"], {{\"learning_rate\": 0}}]]".format(freeze_scope))
        param_overrides.append([[f'^{freeze_scope}'], {'learning_rate': 0}])

    tvars = tf.trainable_variables()
    param_name_to_overridden_parameters = defaultdict(dict)
    for regexes, overridden_parameters in param_overrides:
        for k in overridden_parameters:
            if k not in ('learning_rate', 'weight_decay_rate', 'beta_1', 'beta_2', 'epsilon'):
                raise ValueError(
                    "Regex rule {} -> {} isn't OK because {} isn't a changable optimization parameter".format(
                        regexes, overridden_parameters, k
                    ))

        for regex in regexes:
            for p in tvars:
                param_name = _get_variable_name(p.name)
                if re.search(regex, param_name) is not None:
                    param_name_to_overridden_parameters[param_name].update(overridden_parameters)

    non_trainable_vars = [v for v in tvars
                          if not param_name_to_overridden_parameters[_get_variable_name(v.name)].get('learning_rate',
                                                                                                     1.0)]
    if len(non_trainable_vars) != 0:
        tf.logging.info("\n~~~~~ NOT training the following variables:")
        _print_var_list_for_debugging(non_trainable_vars)
        tvars = [v for v in tvars
                 if param_name_to_overridden_parameters[_get_variable_name(v.name)].get('learning_rate', 1.0)]

    # Get all possible conditions, just for debugging purposes.
    conditions_to_params = defaultdict(list)
    for v in tvars:
        conditions = param_name_to_overridden_parameters[_get_variable_name(v.name)]
        conditions_str = ','.join(f'{k}={v}' for k, v in sorted(conditions.items()))
        conditions_to_params[conditions_str].append(v)

    for conditions, param_list in conditions_to_params.items():
        if not conditions:
            tf.logging.info(
                "\n~~~~~ For the following params, using DEFAULTS \n{}".format(','.join(f'{k}={v}' for k, v in {
                    'learning_rate': learning_rate, 'weight_decay_rate': weight_decay_rate, 'beta_1': 0.9,
                    'beta_2': 0.98,
                    'eps': epsilon
                }.items())))
        else:
            tf.logging.info("\nFor the following params, overriding {}".format(conditions))
        _print_var_list_for_debugging(param_list)

    grads = tf.gradients(loss, tvars)

    if adafactor:
        optimizer = AdaFactorOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            learning_rate_scale=learning_rate_scale,
            beta_1=0.9,
            beta_2=beta_2,
            epsilon=epsilon,
            param_name_to_overridden_parameters=dict(param_name_to_overridden_parameters),
        )
    else:
        optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            learning_rate_scale=learning_rate_scale,
            beta_1=0.9,
            beta_2=beta_2,
            epsilon=epsilon,
            param_name_to_overridden_parameters=dict(param_name_to_overridden_parameters),
            make_things_dependent_on_grad=True,
        )

    train_metrics = {
        'learning_rate': learning_rate * learning_rate_scale,
        'minibatch_loss': loss,
    }

    if verbose:
        train_metrics['weight_decay_loss'] = tf.add_n([
            tf.nn.l2_loss(v) * param_name_to_overridden_parameters[
                _get_variable_name(v.name)].get('weight_decay_rate', weight_decay_rate)
            for v in tvars])

        # Clip grads AND log
        param_to_l2 = {_get_variable_name(x.name): tf.nn.l2_loss(y) for x, y in zip(tvars, grads) if y is not None}
        global_norm = tf.math.sqrt(2.0 * tf.add_n(list(param_to_l2.values())))

        if clip_norm > 0.0:
            tf.logging.info("clipping the global norm to {:.3f}".format(clip_norm))
            (grads, _) = tf.clip_by_global_norm(grads, use_norm=global_norm, clip_norm=clip_norm)
        else:
            tf.logging.info("Not clipping the global norm")

        # Log the global norms. I'm not worrying about grouping or any of that
        # so for language/layer00/key_layer/kernel
        #    and language/layer00/key_layer/bias
        # we log both these parameters as well as language/layer00/key_layer/, language/layer00/ ...
        all_groups = sorted(set(['/'.join(x.split('/')[:(depth + 1)]) for x in param_to_l2.keys()
                                 for depth in range(len(x.split('/')))]))

        for g in all_groups:
            # Hide some boring things
            if g.split('/')[-1] in ('beta', 'kernel', 'bias', 'gamma'):
                continue

            train_metrics[f'gradnorms/{g}'] = tf.math.sqrt(
                2.0 * tf.add_n([v for k, v in param_to_l2.items() if k.startswith(g)]))
        train_metrics['gradnorms/_global_norm'] = global_norm
    else:
        # Clip by global norm. I think we need this, but RoBERTa didn't use it so maybe not? idk. adding it anyways
        if clip_norm > 0.0:
            tf.logging.info("clipping the global norm to {:.3f}".format(clip_norm))
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
        else:
            tf.logging.info("Not clipping the global norm")

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    # + If you're using BN you need UPDATE_OPS to run also
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)],
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    return train_op, train_metrics


def _get_variable_name(param_name):
    """Get the variable name from the tensor name. This just strips off the trailing :0"""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name


class AdamOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.
    Rowan: I modified this from BERT by incorporating the bias correction
    """

    def __init__(self,
                 learning_rate,
                 learning_rate_scale=1.0,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 param_name_to_overridden_parameters=None,
                 name="AdamOptimizer",
                 make_things_dependent_on_grad=False):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.learning_rate_scale = learning_rate_scale
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.param_name_to_overridden_parameters = {} if param_name_to_overridden_parameters is None else param_name_to_overridden_parameters
        self.make_things_dependent_on_grad = make_things_dependent_on_grad

    def _get_hyperparam(self, param_name, hyperparam_name):
        """
        For the given parameter, get the right hyperparameter. It might have been overridden.
        :param param_name:
        :param hyperparam_name:
        :return:
        """
        if hyperparam_name not in ('learning_rate', 'weight_decay_rate', 'beta_1', 'beta_2', 'epsilon'):
            raise ValueError(f"Invalid hyperparameter name {hyperparam_name}")
        if param_name not in self.param_name_to_overridden_parameters:
            return getattr(self, hyperparam_name)
        overridden_params = self.param_name_to_overridden_parameters[param_name]

        return overridden_params.get(hyperparam_name, getattr(self, hyperparam_name))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = _get_variable_name(param.name)
            # Override parameters
            beta_1 = self._get_hyperparam(param_name, 'beta_1')
            beta_2 = self._get_hyperparam(param_name, 'beta_2')
            weight_decay_rate = self._get_hyperparam(param_name, 'weight_decay_rate')
            epsilon = self._get_hyperparam(param_name, 'epsilon')
            learning_rate = self._get_hyperparam(param_name, 'learning_rate') * self.learning_rate_scale

            grad_squared = tf.square(grad) + 1e-30

            if self.make_things_dependent_on_grad:
                # HACK: Make things dependent on grad.
                # This confounds the XLA rewriter and keeps it from fusing computations
                # across different variables.  This fusion is a bad for HBM usage, since
                # it causes the gradients to persist in memory.
                grad_squared_mean = tf.reduce_mean(grad_squared)
                learning_rate += grad_squared_mean * 1e-30
                epsilon += grad_squared_mean * 1e-30

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(beta_1, m) + tf.multiply(1.0 - beta_1, grad))
            next_v = (
                    tf.multiply(beta_2, v) + tf.multiply(1.0 - beta_2, grad_squared))

            update = next_m / (tf.sqrt(next_v) + epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if weight_decay_rate > 0:
                update += weight_decay_rate * param

            update_with_lr = learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)


class AdaFactorOptimizer(AdamOptimizer):
    """ Adafactor optimizer"""

    def __init__(self, *args, **kwargs):
        super(AdaFactorOptimizer, self).__init__(*args, **kwargs, name='AdaFactorOptimizer')
        self.epsilon1 = 1e-30
        self.use_locking = False

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = _get_variable_name(param.name)
            tf.logging.info(param_name)

            # Override parameters
            beta_1 = self._get_hyperparam(param_name, 'beta_1')
            beta_2 = self._get_hyperparam(param_name, 'beta_2')
            weight_decay_rate = self._get_hyperparam(param_name, 'weight_decay_rate')
            epsilon = self._get_hyperparam(param_name, 'epsilon')
            learning_rate = self._get_hyperparam(param_name, 'learning_rate') * self.learning_rate_scale

            shape_list = get_shape_list(param, expected_rank=[1, 2, 3, 4])
            decay_rate = beta_2
            grad_squared = tf.square(grad) + self.epsilon1

            # HACK: Make things dependent on grad.
            # This confounds the XLA rewriter and keeps it from fusing computations
            # across different variables.  This fusion is a bad for HBM usage, since
            # it causes the gradients to persist in memory.
            grad_squared_mean = tf.reduce_mean(grad_squared)
            decay_rate += grad_squared_mean * self.epsilon1
            learning_rate += grad_squared_mean * self.epsilon1
            # End hack

            if len(shape_list) >= 2:
                row_shape = shape_list[:-1]
                col_shape = shape_list[:-2] + shape_list[-1:]

                vr = tf.get_variable(
                    name=param_name + "/adafactor_vr",
                    shape=row_shape,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                vc = tf.get_variable(
                    name=param_name + "/adafactor_vc",
                    shape=col_shape,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

                next_vr = decay_rate * vr + (1 - decay_rate) * tf.reduce_mean(grad_squared, -1)
                next_vc = decay_rate * vc + (1 - decay_rate) * tf.reduce_mean(grad_squared, -2)

                assignments.append(vr.assign(next_vr, use_locking=self.use_locking))
                assignments.append(vc.assign(next_vc, use_locking=self.use_locking))

                long_term_mean = tf.reduce_mean(next_vr, -1, keepdims=True)
                r_factor = tf.rsqrt(next_vr / long_term_mean + epsilon)
                c_factor = tf.rsqrt(next_vc + epsilon)
                update = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
            else:
                v = tf.get_variable(
                    name=param_name + "/adafactor_v",
                    shape=shape_list,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                next_v = decay_rate * v + (1 - decay_rate) * grad_squared

                assignments.append(v.assign(next_v, use_locking=self.use_locking))
                update = grad * tf.rsqrt(next_v + epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if weight_decay_rate > 0:
                update += weight_decay_rate * param
            #
            # # Add bias correction
            # global_step_float = tf.cast(global_step + 1, tf.float32)
            # bias_correction1 = 1.0 - tf.pow(beta_1, global_step_float)
            # bias_correction2 = 1.0 - tf.pow(beta_2, global_step_float)
            #
            # step_size = learning_rate * tf.sqrt(bias_correction1) / bias_correction2

            update_with_lr = learning_rate * update

            next_param = param - update_with_lr

            assignments.append(param.assign(next_param, use_locking=self.use_locking))
        return tf.group(*assignments, name=name)


############################
def learning_rate_schedule(adjusted_learning_rate, lr_warmup_init,
                           lr_warmup_step, first_lr_drop_step,
                           second_lr_drop_step, global_step):
    """Handles linear scaling rule, gradual warmup, and LR decay."""
    # lr_warmup_init is the starting learning rate; the learning rate is linearly
    # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
    linear_warmup = (
            lr_warmup_init + (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step
                              * (adjusted_learning_rate - lr_warmup_init)))
    learning_rate = tf.where(global_step < lr_warmup_step, linear_warmup,
                             adjusted_learning_rate)
    lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                   [0.01, second_lr_drop_step]]
    for mult, start_global_step in lr_schedule:
        learning_rate = tf.where(global_step < start_global_step, learning_rate,
                                 adjusted_learning_rate * mult)
    return learning_rate


def retinanet_momentum_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                                 weight_decay_rate=1e-4, momentum=0.9, first_lr_drop_step=None,
                                 second_lr_drop_step=None, lr_warmup_init=None, **extra_args):
    """
    This is the default optimizer for retinanet. As such it probably doesn't work for other things.
    :param loss:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param weight_decay_rate:
    :param momentum:
    :param first_lr_drop_step:
    :param second_lr_drop_step:
    :param lr_warmup_init:
    :param extra_args:
    :return:
    """
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if first_lr_drop_step is None:
        first_lr_drop_step = int(num_train_steps * 2 / 3.)
    if second_lr_drop_step is None:
        second_lr_drop_step = num_train_steps - num_warmup_steps
    if lr_warmup_init is None:
        lr_warmup_init = learning_rate / 10.0

    learning_rate = learning_rate_schedule(
        adjusted_learning_rate=learning_rate,
        lr_warmup_init=lr_warmup_init,
        lr_warmup_step=num_warmup_steps,
        first_lr_drop_step=first_lr_drop_step,
        second_lr_drop_step=second_lr_drop_step,
        global_step=global_step,
    )

    weight_decay_loss = weight_decay_rate * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if ('batch_normalization' not in v.name) and ('GroupNorm' not in v.name) and ('bias' not in v.name)
    ])

    train_metrics = {
        'learning_rate': learning_rate,
        'minibatch_loss': loss,
        'weight_decay_loss': weight_decay_loss,
    }

    total_loss = loss + weight_decay_loss

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=momentum)
    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires `update_ops` to be executed alongside `train_op`.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # HACK Ignore things starting with resnet{resnet_depth}/conv2d/
    var_list = tf.trainable_variables()
    # var_list = [v for v in tf.trainable_variables() if
    #             (v.name.split('/')[1] != 'conv2d') or (not v.name.split('/')[0].startswith('resnet'))]

    minimize_op = optimizer.minimize(total_loss, global_step, var_list=var_list)
    train_op = tf.group(minimize_op, update_ops)

    return train_op, train_metrics
