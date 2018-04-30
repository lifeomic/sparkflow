import tensorflow as tf
from google.protobuf import json_format
import json


def build_graph(func):
    """
    :param func: Function that includes tensorflow graph
    :return json version of graph
    """
    first_graph = tf.Graph()
    with first_graph.as_default() as g:
        v = func()
        mg = json_format.MessageToJson(tf.train.export_meta_graph())
    return mg


def generate_config(**kwargs):
    return json.dumps(kwargs)


def build_adam_config(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False):
    return generate_config(learning_rate=learning_rate, beta1=beta1,
                           beta2=beta2, epsilon=epsilon, use_locking=use_locking)


def build_rmsprop_config(learning_rate=0.001, decay=0.9,
                         momentum=0.0, epsilon=1e-10, use_locking=False, centered=False):
    return generate_config(learning_rate=learning_rate, decay=decay, momentum=momentum,
                           epsilon=epsilon, use_locking=use_locking, centered=centered)


def build_momentum_config(learning_rate=0.001, momentum=0.9, use_locking=False, use_nesterov=False):
    return generate_config(learning_rate=learning_rate, momentum=momentum,
                           use_locking=use_locking, use_nesterov=use_nesterov)


def build_adadelta_config(learning_rate=0.001, rho=0.95, epsilon=1e-8, use_locking=False):
    return generate_config(learning_rate=learning_rate, rho=rho, epsilon=epsilon, use_locking=use_locking)


def build_adagrad_config(learning_rate=0.001, initial_accumulator=0.1, use_locking=False):
    return generate_config(learning_rate=learning_rate, initial_accumulator=initial_accumulator, use_locking=use_locking)


def build_gradient_descent(learning_rate=0.001, use_locking=False):
    return generate_config(learning_rate=learning_rate, use_locking=use_locking)
