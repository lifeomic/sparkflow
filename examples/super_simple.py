from pyspark.sql import SparkSession
import tensorflow as tf
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, Normalizer
from pyspark.ml.linalg import Vectors, Vector
import numpy as np
from google.protobuf import json_format
import random
from sparkflow.tensorflow_async import SparkAsyncDL, SparkAsyncTransformer
from pyspark.sql.functions import rand


def create_random_model():
    x = tf.placeholder(tf.float32, shape=[None, 10], name='x')
    layer1 = tf.layers.dense(x, 15, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 10, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 1, name='outer', activation=tf.nn.sigmoid)
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    loss = tf.losses.mean_squared_error(y, out)
    return loss

spark = SparkSession.builder \
    .appName("examples") \
    .master('local[8]').config('spark.driver.memory', '4g') \
    .getOrCreate()


dat = [(1.0, Vectors.dense(np.random.normal(0,1,10))) for _ in range(0, 20000)]
dat2 = [(0.0, Vectors.dense(np.random.normal(2,1,10))) for _ in range(0, 20000)]
dat.extend(dat2)
random.shuffle(dat)
processed = spark.createDataFrame(dat, ["label", "features"])

first_graph = tf.Graph()
with first_graph.as_default() as g:
    v = create_random_model()
    mg = json_format.MessageToJson(tf.train.export_meta_graph())

spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=mg,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='outer/Sigmoid:0',
    tfOptimizer='adam',
    tfLearningRate=.1,
    iters=2,
    partitions=4,
    predictionCol='predicted',
    labelCol='label',
    verbose=1
)

data = spark_model.fit(processed).save("blah")