# SparkFlow

This is an implementation of TensorFlow on Spark. The goal of this library is to provide a simple, understandable interface 
in using TensorFlow on Spark. With SparkFlow, you can easily integrate your deep learning model with a ML Spark Pipeline.
Underneath, SparkFlow uses a parameter server to train the TensorFlow network in a distributed manner. Through the api,
the user can specify the style of training, whether that is Hogwild or async with locking.

[![Build Status](https://api.travis-ci.org/lifeomic/sparkflow.svg?branch=master)](https://travis-ci.org/lifeomic/sparkflow)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/lifeomic/sparkflow/blob/master/LICENSE)

## Why should I use this?
While there are other libraries that use TensorFlow on Apache Spark, SparkFlow's objective is to work seamlessly 
with ML Pipelines, provide a simple interface for training TensorFlow graphs, and give basic abstractions for 
faster development. For training, SparkFlow uses a parameter server which lives on the driver and allows for asynchronous training. This tool 
provides faster training time when using big data.

## Installation

Install SparkFlow via pip: `pip install sparkflow`

SparkFlow requires Apache Spark >= 2.0, flask, dill, and TensorFlow to be installed. As of sparkflow >= 0.7.0, only 
python >= 3.5 will be supported.


## Example

#### Simple MNIST Deep Learning Example

```python
from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
import tensorflow as tf
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession


#simple tensorflow network
def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 10)
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .getOrCreate()
    
    df = spark.read.option("inferSchema", "true").csv('mnist_train.csv')
    mg = build_graph(small_model)
    #Assemble and one hot encode
    va = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
    encoded = OneHotEncoder(inputCol='_c0', outputCol='labels', dropLast=False)
    
    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='out:0',
        tfLearningRate=.001,
        iters=20,
        predictionCol='predicted',
        labelCol='labels',
        verbose=1
    )
    
    p = Pipeline(stages=[va, encoded, spark_model]).fit(df)
    p.write().overwrite().save("location")
``` 
Please not that as of SparkFlow version 0.7.0, the parameter server uses a spawn process. This means that global spark sessions 
should be avoided and that python functions should be placed outside of the `__name__ == '__main__'` clause

For a couple more, visit the examples directory. These examples can be run with Docker as well from the provided Dockerfile and 
Makefile. This can be done with the following command:

```bash
make docker-build
make docker-run-dnn
```

Once built, there are also commands to run the example CNN and an autoencoder.


## Documentation

#### Saving and Loading Pipelines

Since saving and loading custom ML Transformers in pure python has not been implemented in PySpark, an extension has been
added here to make that possible. In order to save a Pyspark Pipeline with Apache Spark, one will need to use the overwrite function:

```python
p = Pipeline(stages=[va, encoded, spark_model]).fit(df)
p.write().overwrite().save("location")
```

For loading, a Pipeline wrapper has been provided in the pipeline_utils file. An example is below:

```python
from sparkflow.pipeline_util import PysparkPipelineWrapper
from pyspark.ml.pipeline import PipelineModel

p = PysparkPipelineWrapper.unwrap(PipelineModel.load('location'))
``` 
Then you can perform predictions, etc with:

```python
predictions = p.transform(df)
```

#### Serializing Tensorflow Graph for SparkAsyncDL

You may have already noticed the build_graph function in the example above. This serializes the Tensorflow graph for training on Spark.
The build_graph function only takes one parameter, which is a function that should include the Tensorflow variables.
Below is an example Tensorflow graph function:

```python


def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
    out = tf.layers.dense(layer2, 10)
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss
```

Then to use the build_graph function:

```python
from sparkflow.graph_utils import build_graph
mg = build_graph(small_model)
```

#### Using SparkAsyncDL and Options

SparkAsyncDL has a few options that one can use for training. Not all of the parameters are required. Below is a description 
of each of the parameters:

```
inputCol: Spark dataframe inputCol. Similar to other spark ml inputCols
tensorflowGraph: The protobuf tensorflow graph. You can use the utility function in graph_utils to generate the graph for you
tfInput: The tensorflow input. This points us to the input variable name that you would like to use for training
tfLabel: The tensorflow label. This is the variable name for the label.
tfOutput: The tensorflow raw output. This is for your loss function.
tfOptimizer: The optimization function you would like to use for training. Defaults to adam
tfLearningRate: Learning rate of the optimization function
iters: number of iterations of training
predictionCol: The prediction column name on the spark dataframe for transformations
partitions: Number of partitions to use for training (recommended on partition per instance)
miniBatchSize: size of the mini batch. A size of -1 means train on all rows
miniStochasticIters: If using a mini batch, you can choose number of mini iters you would like to do with the batch size above per epoch. A value of -1 means that you would like to run mini-batches on all data in the partition
acquireLock: If you do not want to utilize hogwild training, this will set a lock
shufflePerIter: Specifies if you want to shuffle the features after each iteration
tfDropout: Specifies the dropout variable. This is important for predictions
toKeepDropout: Due to conflicting TF implementations, this specifies whether the dropout function means to keep a percentage of values or to drop a percentage of values.
verbose: Specifies log level of training results
labelCol: Label column for training
partitionShuffles: This will shuffle your data after iterations are completed, then run again. For example,
if you have 2 partition shuffles and 100 iterations, it will run 100 iterations then reshuffle and run 100 iterations again.
The repartition hits performance and should be used with care.
optimizerOptions: Json options to apply to tensorflow optimizers.
```

#### Optimization Configuration

As of SparkFlow version 0.2.1, TensorFlow optimization configuration options can be added to SparkAsyncDL for more control 
over the optimizer. While the user can supply the configuration json directly, there are a few provided utility 
functions that include the parameters necessary. An example is provided below.

```python

from sparkflow.graph_utils import build_adam_config


adam_config = build_adam_config(learning_rate=0.001, beta1=0.9, beta2=0.999)
spark_model = SparkAsyncDL(
    ...,
    optimizerOptions=adam_config
)
```

#### Loading pre-trained Tensorflow model

To load a pre-trained Tensorflow model and use it as a spark pipeline, it can be achieved using the following code:

```python
from sparkflow.tensorflow_model_loader import load_tensorflow_model

df = spark.read.parquet("data")
loaded_model = load_tensorflow_model(
    path="./test_model/to_load",
    inputCol="features",
    tfInput="x:0",
    tfOutput="out/Sigmoid:0"
)
data_with_predictions = loaded_model.transform(df)
```


## Running

One big thing to remember, especially for larger networks, is to add the `--executor cores 1` option to spark to ensure
each instance is only training one copy of the network. This will especially be needed for gpu training as well.


## Contributing

Contributions are always welcome. This could be fixing a bug, changing documentation, or adding a new feature. To test 
new changes against existing tests, we have provided a Docker container which takes in an argument of the python version. 
This allows the user to check their work before pushing to Github, where travis-ci will run.

For 2.7 (sparkflow <= 0.6.0):
```
docker build -t local-test --build-arg PYTHON_VERSION=2.7 .
docker run --rm local-test:latest bash -i -c "python tests/dl_runner.py"
```

For 3.6
```
docker build -t local-test --build-arg PYTHON_VERSION=3.6 .
docker run --rm local-test:latest bash -i -c "python tests/dl_runner.py"
```


## Future planned features 

* Hyperopt implementation for smaller and larger datasets
* AWS EMR guides


## Literature and Inspiration

* HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent: https://arxiv.org/pdf/1106.5730.pdf
* Elephas: https://github.com/maxpumperla/elephas
* Scaling Distributed Machine Learning with the Parameter Server: https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf
