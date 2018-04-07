import numpy as np
import tensorflow as tf
import json
from google.protobuf import json_format
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row


def tensorflow_get_weights():
    vs = tf.trainable_variables()
    values = tf.get_default_session().run(vs)
    return values


def tensorflow_set_weights(weights):
    assign_ops = []
    feed_dict = {}
    vs = tf.trainable_variables()
    zipped_values = zip(vs, weights)
    for var, value in zipped_values:
        value = np.asarray(value)
        assign_placeholder = tf.placeholder(var.dtype, shape=value.shape)
        assign_op = var.assign(assign_placeholder)
        assign_ops.append(assign_op)
        feed_dict[assign_placeholder] = value
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


def convert_weights_to_json(weights):
    weights = [w.tolist() for w in weights]
    weights_list = json.dumps(weights)
    return weights_list


def convert_json_to_weights(json_weights):
    loaded_weights = json.loads(json_weights)
    loaded_weights = [np.asarray(x) for x in loaded_weights]
    return loaded_weights


def calculate_weights(collected_weights):
    size = len(collected_weights)
    start = collected_weights[0]
    if size > 1:
        for i in range(1, size):
            vs = collected_weights[i]
            for x in range(0, len(vs)):
                start[x] = start[x] + vs[x]
    return [x / size for x in start]


def predict_func(rows, graph_json, prediction, graph_weights, inp, activation, tf_input, tf_dropout=None, to_keep_dropout=False):
    rows = [r.asDict() for r in rows]
    graph = tf.MetaGraphDef()
    graph = json_format.Parse(graph_json, graph)
    loaded_weights = json.loads(graph_weights)
    loaded_weights = [np.asarray(x) for x in loaded_weights]

    A = []
    for row in rows:
        encoded = np.asarray(row[inp])
        A.append(encoded)
    A = np.asarray(A)

    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        tf.train.import_meta_graph(graph)
        sess.run(tf.global_variables_initializer())
        tensorflow_set_weights(loaded_weights)
        out_node = tf.get_default_graph().get_tensor_by_name(activation)
        dropout_v = 1.0 if tf_dropout is not None and to_keep_dropout else 0.0
        feed_dict = {tf_input: A} if tf_dropout is None else {tf_input: A, tf_dropout: dropout_v}

        pred = sess.run(out_node, feed_dict=feed_dict)
        for i in range(0, len(rows)):
            row = rows[i]
            row[prediction] = Vectors.dense(pred[i])

    rows = [Row(**a) for a in rows]
    return rows


def handle_features(data, is_supervised=False):
    features = []
    labels = []
    for feature in data:
        if is_supervised:
            x,y = feature
            if type(y) is int or type(y) is float:
                labels.append([y])
            else:
                labels.append(y)
        else:
            x = feature
        features.append(x)
    features = np.asarray(features)
    labels = np.asarray(labels) if is_supervised else None
    return features, labels


def handle_feed_dict(train, tfInput, tfLabel=None, labels=None, mini_batch_size=-1, idx=None):
    if mini_batch_size > train.shape[0]:
        mini_batch_size = train.shape[0]-1

    if mini_batch_size <= 0:
        if tfLabel == None:
            return {tfInput: train}
        else:
            return {tfInput: train, tfLabel: labels}

    if idx is not None:
        train_data = train[idx:idx+mini_batch_size]
        if tfLabel == None:
            return {tfInput: train_data}
        else:
            return {tfInput: train_data, tfLabel: labels[idx:idx+mini_batch_size]}

    num_dims = np.random.choice(train.shape[0], mini_batch_size, replace=False)
    features_mini = train[num_dims]

    if tfLabel == None:
        return {tfInput: features_mini}
    else:
        return {tfInput: features_mini, tfLabel: labels[num_dims]}


def handle_shuffle(features, labels):
    shuff_idxs = np.random.choice(features.shape[0], features.shape[0], replace=False)
    features = features[shuff_idxs]
    labels = labels[shuff_idxs] if labels is not None else None
    return features, labels

