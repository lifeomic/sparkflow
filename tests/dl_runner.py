from pyspark.sql import SparkSession
import tensorflow as tf
from pyspark.ml.linalg import Vectors
import numpy as np
from google.protobuf import json_format
import random
from sparkflow.tensorflow_async import SparkAsyncDL
from sparkflow.HogwildSparkModel import HogwildSparkModel
from sparkflow.graph_utils import build_graph, build_adam_config, build_rmsprop_config
from sparkflow.tensorflow_model_loader import load_tensorflow_model

random.seed(12345)

spark = SparkSession.builder \
    .appName("variant-deep") \
    .master('local[8]') \
    .config('spark.sql.pivotMaxValues', 100000) \
    .getOrCreate()


def create_model():
    x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
    layer1 = tf.layers.dense(x, 12, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 7, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 1, name='outer', activation=tf.nn.sigmoid)
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    loss = tf.losses.mean_squared_error(y, out)
    return loss


def create_random_model():
    x = tf.placeholder(tf.float32, shape=[None, 10], name='x')
    layer1 = tf.layers.dense(x, 12, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 7, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 1, name='outer', activation=tf.nn.sigmoid)
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    loss = tf.losses.mean_squared_error(y, out)
    return loss


def handle_assertions(spark_model, processed):
    data = spark_model.fit(processed).transform(processed).take(10)
    nb_errors = 0
    for d in data:
        lab = d['label']
        predicted = 1 if d['predicted'][0] >= 0.5 else 0
        if predicted != lab:
            nb_errors += 1
    assert nb_errors < len(data)


def generate_random_data():
    dat = [(1.0, Vectors.dense(np.random.normal(0,1,10))) for _ in range(0, 200)]
    dat2 = [(0.0, Vectors.dense(np.random.normal(2,1,10))) for _ in range(0, 200)]
    dat.extend(dat2)
    random.shuffle(dat)
    return spark.createDataFrame(dat, ["label", "features"])


def test_adam_optimizer_options():
    processed = generate_random_data()
    mg = build_graph(create_random_model)
    options = build_adam_config(learning_rate=0.1, beta1=0.85, beta2=0.98, epsilon=1e-8)
    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='outer/Sigmoid:0',
        tfOptimizer='adam',
        tfLearningRate=.1,
        iters=25,
        partitions=3,
        predictionCol='predicted',
        labelCol='label',
        verbose=1,
        optimizerOptions=options
    )
    handle_assertions(spark_model, processed)


def test_load_raw_model():
    xor = [(0.0, Vectors.sparse(2,[0,1],[0.0,0.0])),
           (0.0, Vectors.sparse(2,[0,1],[1.0,1.0])),
           (1.0, Vectors.sparse(2,[0],[1.0])),
           (1.0, Vectors.sparse(2,[1],[1.0]))]
    processed = spark.createDataFrame(xor, ["label", "features"])
    loaded = load_tensorflow_model(
        "./test_model/to_load",
        "features",
        "x:0",
        "out/Sigmoid:0"
    ).transform(processed).collect()
    assert loaded[0]['predicted'][0] < 0.1
    assert loaded[1]['predicted'][0] < 0.1
    assert loaded[2]['predicted'][0] > 0.7
    assert loaded[3]['predicted'][0] > 0.7


def test_small_sparse():
    xor = [(0.0, Vectors.sparse(2,[0,1],[0.0,0.0])),
           (0.0, Vectors.sparse(2,[0,1],[1.0,1.0])),
           (1.0, Vectors.sparse(2,[0],[1.0])),
           (1.0, Vectors.sparse(2,[1],[1.0]))]
    processed = spark.createDataFrame(xor, ["label", "features"])

    mg=build_graph(create_model)
    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='outer/Sigmoid:0',
        tfOptimizer='adam',
        tfLearningRate=.1,
        iters=35,
        partitions=4,
        predictionCol='predicted',
        labelCol='label'
    )
    assert spark_model.fit(processed).transform(processed).collect() is not None


def test_spark_hogwild():
    xor = [(0.0, Vectors.dense(np.array([0.0, 0.0]))),
           (0.0, Vectors.dense(np.array([1.0, 1.0]))),
           (1.0, Vectors.dense(np.array([1.0, 0.0]))),
           (1.0, Vectors.dense(np.array([0.0, 1.0])))]
    processed = spark.createDataFrame(xor, ["label", "features"]) \
        .coalesce(1).rdd.map(lambda x: (np.asarray(x["features"]), x["label"]))

    first_graph = tf.Graph()
    with first_graph.as_default() as g:
        v = create_model()
        mg = json_format.MessageToJson(tf.train.export_meta_graph())

    spark_model = HogwildSparkModel(
        tensorflowGraph=mg,
        iters=10,
        tfInput='x:0',
        tfLabel='y:0',
        optimizer=tf.train.AdamOptimizer(learning_rate=.1),
        master_url='localhost:5000'
    )

    try:
        weights = spark_model.train(processed)
        assert len(weights) > 0
    except Exception as e:
        spark_model.stop_server()
        raise Exception(e.message)


def test_overlapping_guassians():
    processed = generate_random_data()
    mg = build_graph(create_random_model)

    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='outer/Sigmoid:0',
        tfOptimizer='adam',
        tfLearningRate=.1,
        iters=35,
        partitions=4,
        predictionCol='predicted',
        labelCol='label'
    )
    handle_assertions(spark_model, processed)


def test_rmsprop():
    processed = generate_random_data()
    mg = build_graph(create_random_model)
    options = build_rmsprop_config(learning_rate=0.1, decay=0.95, momentum=0.1, centered=False)
    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='outer/Sigmoid:0',
        tfOptimizer='rmsprop',
        tfLearningRate=.1,
        iters=25,
        partitions=4,
        predictionCol='predicted',
        labelCol='label',
        optimizerOptions=options
    )
    handle_assertions(spark_model, processed)


def test_multi_partition_shuffle():
    processed = generate_random_data()
    mg = build_graph(create_random_model)
    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='outer/Sigmoid:0',
        tfOptimizer='adam',
        tfLearningRate=.1,
        iters=20,
        partitions=4,
        predictionCol='predicted',
        labelCol='label',
        partitionShuffles=2
    )
    handle_assertions(spark_model, processed)


