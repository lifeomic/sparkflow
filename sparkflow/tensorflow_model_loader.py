import tensorflow_core as tf
from sparkflow.tensorflow_async import SparkAsyncDLModel
from google.protobuf import json_format
from pyspark.ml.pipeline import PipelineModel
import json


def load_tensorflow_model(
        path,
        inputCol,
        tfInput,
        tfOutput,
        predictionCol='predicted',
        tfDropout=None,
        toKeepDropout=False):
    with tf.Session(graph=tf.Graph()) as sess:
        new_saver = tf.train.import_meta_graph(path + '.meta')
        split = path.split('/')
        if len(split) > 1:
            new_saver.restore(sess, tf.train.latest_checkpoint("/".join(split[:-1])))
        else:
            new_saver.restore(sess, tf.train.latest_checkpoint(split[0]))
        vs = tf.trainable_variables()
        weights = sess.run(vs)
        json_graph = json_format.MessageToJson(tf.train.export_meta_graph())

    weights = [w.tolist() for w in weights]
    json_weights = json.dumps(weights)
    return SparkAsyncDLModel(
        inputCol=inputCol, modelJson=json_graph, modelWeights=json_weights,
        tfInput=tfInput, tfOutput=tfOutput, predictionCol=predictionCol, tfDropout=tfDropout, toKeepDropout=toKeepDropout
    )


def attach_tensorflow_model_to_pipeline(
        path,
        pipelineModel,
        inputCol,
        tfInput,
        tfOutput,
        predictionCol='predicted',
        tfDropout=None,
        toKeepDropout=False ):
    spark_model = load_tensorflow_model(path, inputCol, tfInput, tfOutput, predictionCol, tfDropout, toKeepDropout)
    return PipelineModel(stages=[pipelineModel, spark_model])
