import tensorflow as tf
from google.protobuf import json_format


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
