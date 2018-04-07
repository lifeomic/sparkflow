# Lifeomic SparkFlow

Alpha implementation of Tensorflow on Spark. The goal of this library is to provide a simple, understandable interface 
in using Tensorflow on Spark. With SparkFlow, you can easily integrate your deep learning model with a Spark Pipeline.
Underneath, SparkFlow uses a parameter server to train the Tensorflow network in a distributed manor. Through the api,
the user can specify the style of training, whether that is Hogwild or async with locking.

## Installation

Install sparkflow via pip: `pip install sparkflow`

SparkFlow requires Apache Spark >= 2.0, flask, and Tensorflow to all be installed.


## Example

#### Simple MNIST Deep Learning Example

```python
from sparkflow.graph_utils import build_graph
from sparkflow.tensorflow_async import SparkAsyncDL
import tensorflow as tf
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.pipeline import Pipeline
    
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
    iters=1,
    predictionCol='predicted',
    labelCol='labels',
    verbose=1
)

p = Pipeline(stages=[va, encoded, spark_model]).fit(df)
p.write().overwrite().save("location")
``` 

For a couple more, visit the examples directory. More examples will be coming up soon.


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
miniStochasticIters: If using a mini batch, you can choose number of mini iters you would like to do a value of -1 means that you would only like to do one run
acquireLock: If you do not want to utilize hogwild training, this will set a lock
shufflePerIter: Specifies if you want to shuffle the features after each iteration
tfDropout: Specifies the dropout variable. This is important for predictions
toKeepDropout: Due to conflicting TF implementations, this specifies whether the dropout function means to keep a percentage of values or to drop a percentage of values.
verbose: Specifies log level of training results
labelCol: Label column for training
```

## Running

One big thing to remember, especially for larger networks, is to add the `--executor cores 1` option to spark to ensure
each instance is only training one network. This will also be needed for gpu training as well.


## Known Issues and Future Plans

One known issue is the performance speed of saving large sparkflow models. We currently have plans to fix this in the near term.
Also, once PySpark implements saving/loading capabilities for estimators, we will point our models to that, instead of our 
custom solutions.

A hyper-opt implementation is on the roadmap as well, and will be implemented within the next couple of months.

