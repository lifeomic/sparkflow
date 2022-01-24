import tensorflow_core as tf
from .pipeline_util import PysparkReaderWriter
import numpy as np

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, HasLabelCol
from pyspark.ml.base import Estimator
from pyspark.ml import Model
from pyspark.ml.util import Identifiable, MLReadable, MLWritable
from pyspark import keyword_only
from HogwildSparkModel import HogwildSparkModel
from ml_util import convert_weights_to_json, predict_func
from pyspark import SparkContext
import json


def build_optimizer(optimizer_name, learning_rate, optimizer_options):

    available_optimizers = {
        'adam': tf.train.AdamOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'gradient_descent': tf.train.GradientDescentOptimizer,
        'adagrad_da': tf.train.AdagradDAOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'proximal_adagrad': tf.train.ProximalAdagradOptimizer,
        'proximal_gradient_descent': tf.train.ProximalGradientDescentOptimizer
    }

    if optimizer_options is None:
        optimizer_options = {
            "learning_rate": learning_rate,
            "use_locking": False
        }
        if optimizer_name == 'momentum':
            optimizer_options['momentum'] = 0.9

    if optimizer_name in available_optimizers:
        return available_optimizers[optimizer_name](**optimizer_options)
    return available_optimizers['gradient_descent'](**optimizer_options)


def handle_data(data, inp_col, label_col):
    if label_col is None:
        return np.asarray(data[inp_col])
    return np.asarray(data[inp_col]), data[label_col]


class SparkAsyncDLModel(Model, HasInputCol, HasPredictionCol, PysparkReaderWriter, MLReadable, MLWritable, Identifiable):

    modelJson = Param(Params._dummy(), "modelJson", "", typeConverter=TypeConverters.toString)
    modelWeights = Param(Params._dummy(), "modelWeights", "", typeConverter=TypeConverters.toString)
    tfOutput = Param(Params._dummy(), "tfOutput", "", typeConverter=TypeConverters.toString)
    tfInput = Param(Params._dummy(), "tfInput", "", typeConverter=TypeConverters.toString)
    tfDropout = Param(Params._dummy(), "tfDropout", "", typeConverter=TypeConverters.toString)
    toKeepDropout = Param(Params._dummy(), "toKeepDropout", "", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self,
                 inputCol=None,
                 modelJson=None,
                 modelWeights=None,
                 tfInput=None,
                 tfOutput=None,
                 tfDropout=None,
                 toKeepDropout=None,
                 predictionCol=None):
        super(SparkAsyncDLModel, self).__init__()
        self._setDefault(modelJson=None,  inputCol='encoded',
                         predictionCol='predicted', tfOutput=None, tfInput=None,
                         modelWeights=None, tfDropout=None, toKeepDropout=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self,
                  inputCol=None,
                  modelJson=None,
                  modelWeights=None,
                  tfInput=None,
                  tfOutput=None,
                  tfDropout=None,
                  toKeepDropout=None,
                  predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        inp = self.getOrDefault(self.inputCol)
        out = self.getOrDefault(self.predictionCol)
        mod_json = self.getOrDefault(self.modelJson)
        mod_weights = self.getOrDefault(self.modelWeights)
        tf_input = self.getOrDefault(self.tfInput)
        tf_output = self.getOrDefault(self.tfOutput)
        tf_dropout = self.getOrDefault(self.tfDropout)
        to_keep_dropout = self.getOrDefault(self.toKeepDropout)
        return dataset.rdd.mapPartitions(lambda x: predict_func(x, mod_json, out, mod_weights, inp, tf_output, tf_input, tf_dropout, to_keep_dropout)).toDF()


class SparkAsyncDL(Estimator, HasInputCol, HasPredictionCol, HasLabelCol,PysparkReaderWriter, MLReadable, MLWritable, Identifiable):

    tensorflowGraph = Param(Params._dummy(), "tensorflowGraph", "", typeConverter=TypeConverters.toString)
    tfInput = Param(Params._dummy(), "tfInput", "", typeConverter=TypeConverters.toString)
    tfOutput = Param(Params._dummy(), "tfOutput", "", typeConverter=TypeConverters.toString)
    tfLabel = Param(Params._dummy(), "tfLabel", "", typeConverter=TypeConverters.toString)
    tfOptimizer = Param(Params._dummy(), "tfOptimizer", "", typeConverter=TypeConverters.toString)
    tfLearningRate = Param(Params._dummy(), "tfLearningRate", "", typeConverter=TypeConverters.toFloat)
    iters = Param(Params._dummy(), "iters", "", typeConverter=TypeConverters.toInt)
    partitions = Param(Params._dummy(), "partitions", "", typeConverter=TypeConverters.toInt)
    miniBatchSize = Param(Params._dummy(), "miniBatchSize", "", typeConverter=TypeConverters.toInt)
    miniStochasticIters = Param(Params._dummy(), "miniStochasticIters", "", typeConverter=TypeConverters.toInt)
    verbose = Param(Params._dummy(), "verbose", "", typeConverter=TypeConverters.toInt)
    acquireLock = Param(Params._dummy(), "acquireLock", "", typeConverter=TypeConverters.toBoolean)
    shufflePerIter = Param(Params._dummy(), "shufflePerIter", "", typeConverter=TypeConverters.toBoolean)
    tfDropout = Param(Params._dummy(), "tfDropout", "", typeConverter=TypeConverters.toString)
    toKeepDropout = Param(Params._dummy(), "toKeepDropout", "", typeConverter=TypeConverters.toBoolean)
    partitionShuffles = Param(Params._dummy(), "partitionShuffles", "", typeConverter=TypeConverters.toInt)
    optimizerOptions = Param(Params._dummy(), "optimizerOptions", "", typeConverter=TypeConverters.toString)
    port = Param(Params._dummy(), "port", "", typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self,
                 inputCol=None,
                 tensorflowGraph=None,
                 tfInput=None,
                 tfLabel=None,
                 tfOutput=None,
                 tfOptimizer=None,
                 tfLearningRate=None,
                 iters=None,
                 predictionCol=None,
                 partitions=None,
                 miniBatchSize = None,
                 miniStochasticIters=None,
                 acquireLock=None,
                 shufflePerIter=None,
                 tfDropout=None,
                 toKeepDropout=None,
                 verbose=None,
                 labelCol=None,
                 partitionShuffles=None,
                 optimizerOptions=None,
                 port=None):
        """
        :param inputCol: Spark dataframe inputCol. Similar to other spark ml inputCols
        :param tensorflowGraph: The protobuf tensorflow graph. You can use the utility function in graph_utils
        to generate the graph for you
        :param tfInput: The tensorflow input. This points us to the input variable name that you would like to use
        for training
        :param tfLabel: The tensorflow label. This is the variable name for the label.
        :param tfOutput: The tensorflow raw output. This is for your loss function.
        :param tfOptimizer: The optimization function you would like to use for training. Defaults to adam
        :param tfLearningRate: Learning rate of the optimization function
        :param iters: number of iterations of training
        :param predictionCol: The prediction column name on the spark dataframe for transformations
        :param partitions: Number of partitions to use for training (recommended on partition per instance)
        :param miniBatchSize: size of the mini batch. A size of -1 means train on all rows
        :param miniStochasticIters: If using a mini batch, you can choose number of mini iters you would like to do with the
        batch size above per epoch. A value of -1 means that you would like to run mini-batches on all data in the partition
        :param acquireLock: If you do not want to utilize hogwild training, this will set a lock
        :param shufflePerIter: Specifies if you want to shuffle the features after each iteration
        :param tfDropout: Specifies the dropout variable. This is important for predictions
        :param toKeepDropout: Due to conflicting TF implementations, this specifies whether the dropout function means
        to keep a percentage of values or to drop a percentage of values.
        :param verbose: Specifies log level of training results
        :param labelCol: Label column for training
        :param partitionShuffles: This will shuffle your data after iterations are completed, then run again. For example,
        if you have 2 partition shuffles and 100 iterations, it will run 100 iterations then reshuffle and run 100 iterations again.
        The repartition hits performance and should be used with care.
        :param optimizerOptions: Json options to apply to tensorflow optimizers.
        :param port: Flask Port
        """
        super(SparkAsyncDL, self).__init__()
        self._setDefault(inputCol='transformed', tensorflowGraph='',
                         tfInput='x:0', tfLabel=None, tfOutput='out/Sigmoid:0',
                         tfOptimizer='adam', tfLearningRate=.01, partitions=5,
                         miniBatchSize=128, miniStochasticIters=-1,
                         shufflePerIter=True, tfDropout=None, acquireLock=False, verbose=0,
                         iters=1000, toKeepDropout=False, predictionCol='predicted', labelCol=None,
                         partitionShuffles=1, optimizerOptions=None, port=5000)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self,
                  inputCol=None,
                  tensorflowGraph=None,
                  tfInput=None,
                  tfLabel=None,
                  tfOutput=None,
                  tfOptimizer=None,
                  tfLearningRate=None,
                  iters=None,
                  predictionCol=None,
                  partitions=None,
                  miniBatchSize = None,
                  miniStochasticIters=None,
                  acquireLock=None,
                  shufflePerIter=None,
                  tfDropout=None,
                  toKeepDropout=None,
                  verbose=None,
                  labelCol=None,
                  partitionShuffles=None,
                  optimizerOptions=None,
                  port=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getTensorflowGraph(self):
        return self.getOrDefault(self.tensorflowGraph)

    def getIters(self):
        return self.getOrDefault(self.iters)

    def getTfInput(self):
        return self.getOrDefault(self.tfInput)

    def getTfLabel(self):
        return self.getOrDefault(self.tfLabel)

    def getTfOutput(self):
        return self.getOrDefault(self.tfOutput)

    def getTfOptimizer(self):
        return self.getOrDefault(self.tfOptimizer)

    def getTfLearningRate(self):
        return self.getOrDefault(self.tfLearningRate)

    def getPartitions(self):
        return self.getOrDefault(self.partitions)

    def getMiniBatchSize(self):
        return self.getOrDefault(self.miniBatchSize)

    def getMiniStochasticIters(self):
        return self.getOrDefault(self.miniStochasticIters)

    def getVerbose(self):
        return self.getOrDefault(self.verbose)

    def getAqcuireLock(self):
        return self.getOrDefault(self.acquireLock)

    def getShufflePerIter(self):
        return self.getOrDefault(self.shufflePerIter)

    def getTfDropout(self):
        return self.getOrDefault(self.tfDropout)

    def getToKeepDropout(self):
        return self.getOrDefault(self.toKeepDropout)

    def getPartitionShuffles(self):
        return self.getOrDefault(self.partitionShuffles)

    def getOptimizerOptions(self):
        return self.getOrDefault(self.optimizerOptions)

    def getPort(self):
        return self.getOrDefault(self.port)

    def _fit(self, dataset):
        inp_col = self.getInputCol()
        graph_json = self.getTensorflowGraph()
        iters = self.getIters()
        label = self.getLabelCol()
        prediction = self.getPredictionCol()
        tf_input = self.getTfInput()
        tf_label = self.getTfLabel()
        tf_output = self.getTfOutput()
        optimizer_options = self.getOptimizerOptions()
        if optimizer_options is not None:
            optimizer_options = json.loads(optimizer_options)
        tf_optimizer = build_optimizer(self.getTfOptimizer(), self.getTfLearningRate(), optimizer_options)
        partitions = self.getPartitions()
        acquire_lock = self.getAqcuireLock()
        mbs = self.getMiniBatchSize()
        msi = self.getMiniStochasticIters()
        verbose = self.getVerbose()
        spi = self.getShufflePerIter()
        tf_dropout = self.getTfDropout()
        to_keep_dropout = self.getToKeepDropout()
        partition_shuffles = self.getPartitionShuffles()
        port = self.getPort()

        df = dataset.rdd.map(lambda x: handle_data(x, inp_col, label))
        df = df.coalesce(partitions) if partitions < df.getNumPartitions() else df

        spark_model = HogwildSparkModel(
            tensorflowGraph=graph_json,
            iters=iters,
            tfInput=tf_input,
            tfLabel=tf_label,
            optimizer=tf_optimizer,
            master_url=SparkContext._active_spark_context.getConf().get("spark.driver.host").__str__() + ":" + str(port),
            acquire_lock=acquire_lock,
            mini_batch=mbs,
            mini_stochastic_iters=msi,
            shuffle=spi,
            verbose=verbose,
            partition_shuffles=partition_shuffles,
            port=port
        )

        weights = spark_model.train(df)
        json_weights = convert_weights_to_json(weights)

        return SparkAsyncDLModel(
            inputCol=inp_col,
            modelJson=graph_json,
            modelWeights=json_weights,
            tfOutput=tf_output,
            tfInput=tf_input,
            tfDropout=tf_dropout,
            toKeepDropout=to_keep_dropout,
            predictionCol=prediction
        )
