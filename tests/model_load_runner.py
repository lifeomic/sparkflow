import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from sparkflow.tensorflow_model_loader import load_tensorflow_model
from pyspark.ml.linalg import Vectors

X = np.array([[0.0, 0.0], [1.0,1.0], [1.0, 0.0], [0.0, 1.0]])
Y = np.asarray([[0.0], [0.0], [1.0], [1.0]])

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

layer = tf.layers.dense(x, 10, activation=tf.nn.tanh)
layer2 = tf.layers.dense(layer, 10, activation=tf.nn.tanh)
out = tf.layers.dense(layer2, 1, activation=tf.nn.sigmoid, name="out")

loss = tf.reduce_mean(0.5 * tf.square(y - out))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
mini_func = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        sess.run(mini_func, feed_dict={x: X, y: Y})
        print(sess.run(loss, feed_dict={x:X, y:Y}))
    saver.save(sess, "./test_model/to_load")
"""
spark = SparkSession.builder \
    .appName("variant-deep") \
    .master('local[8]') \
    .config('spark.sql.pivotMaxValues', 100000) \
    .getOrCreate()

xor = [(0.0, Vectors.sparse(2,[0,1],[0.0,0.0])),
       (0.0, Vectors.sparse(2,[0,1],[1.0,1.0])),
       (1.0, Vectors.sparse(2,[0],[1.0])),
       (1.0, Vectors.sparse(2,[1],[1.0]))]
processed = spark.createDataFrame(xor, ["label", "features"])

loaded = load_tensorflow_model(
    "./test_model/to_load",
    "features",
    "x:0",
    "y:0"
)

print(loaded.transform(processed).collect())
"""