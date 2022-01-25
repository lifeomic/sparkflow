from flask import Flask, request
import six.moves.cPickle as pickle
from ml_util import tensorflow_get_weights, tensorflow_set_weights, handle_features, handle_feed_dict, handle_shuffle

from google.protobuf import json_format
import socket
import time
import tensorflow_core as tf
import itertools
from RWLock import RWLock
from multiprocessing import Process
import multiprocessing
import uuid
import requests


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def get_server_weights(master_url='0.0.0.0:5000'):
    """
    This will get the raw weights, pickle load them, and return.
    """
    r = requests.get('http://{0}/parameters'.format(master_url))
    weights = pickle.loads(r.content)
    return weights


def put_deltas_to_server(delta, master_url='0.0.0.0:5000'):
    """
    This updates the master parameters. We just use simple pickle serialization here.
    """
    requests.post('http://{0}/update'.format(master_url), data=pickle.dumps(delta, -1))


def handle_model(data, graph_json, tfInput, tfLabel=None,
                 master_url='0.0.0.0:5000', iters=1000,
                 mini_batch_size=-1, shuffle=True,
                 mini_stochastic_iters=-1, verbose=0, loss_callback=None):
    is_supervised = tfLabel is not None
    features, labels = handle_features(data, is_supervised)

    gd = tf.MetaGraphDef()
    gd = json_format.Parse(graph_json, gd)
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        tf.train.import_meta_graph(gd)
        loss_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        sess.run(tf.global_variables_initializer())
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss_variable, trainable_variables)
        grads = list(zip(grads, trainable_variables))
        partition_id = uuid.uuid4().hex
        for i in range(0, iters):
            weights = get_server_weights(master_url)
            tensorflow_set_weights(weights, vs=trainable_variables)
            if shuffle:
                features, labels = handle_shuffle(features, labels)

            if mini_stochastic_iters >= 1:
                for _ in range(0, mini_stochastic_iters):
                    gradients = []
                    feed_dict = handle_feed_dict(features, tfInput, tfLabel, labels, mini_batch_size)
                    for x in range(len(grads)):
                        gradients.append(grads[x][0].eval(feed_dict=feed_dict))
                    try:
                        put_deltas_to_server(gradients, master_url)
                    except Exception:
                        print("Timeout error from partition %s" % partition_id)
            elif mini_batch_size >= 1:
                for r in range(0, len(features), mini_batch_size):
                    gradients = []
                    weights = get_server_weights(master_url)
                    tensorflow_set_weights(weights, vs=trainable_variables)
                    feed_dict = handle_feed_dict(features, tfInput, tfLabel, labels, mini_batch_size, idx=r)
                    for x in range(len(grads)):
                        gradients.append(grads[x][0].eval(feed_dict=feed_dict))
                    try:
                        put_deltas_to_server(gradients, master_url)
                    except Exception:
                        print("Timeout error from partition %s" % partition_id)
            else:
                gradients = []
                feed_dict = handle_feed_dict(features, tfInput, tfLabel, labels, mini_batch_size)
                for x in range(len(grads)):
                    gradients.append(grads[x][0].eval(feed_dict=feed_dict))
                try:
                    put_deltas_to_server(gradients, master_url)
                except Exception:
                    print("Timeout error from partition %s" % partition_id)

            if verbose or loss_callback:
                feed_dict = handle_feed_dict(features, tfInput, tfLabel, labels, -1)
                loss = sess.run(loss_variable, feed_dict=feed_dict)
                if verbose:
                    print("Partition Id: %s, Iteration: %i, Loss: %f" % (partition_id, i, loss))
                if loss_callback:
                    loss_callback(loss, i, partition_id)


class HogwildSparkModel(object):
    """
    Hogwild implementation of spark and tensorflow. This sets up a service with Flask, and each of the nodes will
    send their computed gradients to the driver, where they will be updated randomly. Without being thread safe,
    this provides stochasticity, avoiding biases for large models
    """

    def __init__(self,
                 tensorflowGraph=None,
                 iters=1000,
                 tfInput=None,
                 tfLabel=None,
                 optimizer=None,
                 master_url=None,
                 serverStartup=8,
                 acquire_lock=False,
                 mini_batch=-1,
                 mini_stochastic_iters=-1,
                 shuffle=True,
                 verbose=0,
                 partition_shuffles=1,
                 loss_callback=None,
                 port=5000):
        self.tensorflowGraph = tensorflowGraph
        self.iters = iters
        self.tfInput = tfInput
        self.tfLabel = tfLabel
        self.acquire_lock = acquire_lock
        graph = tf.MetaGraphDef()
        metagraph = json_format.Parse(tensorflowGraph, graph)
        self.start_server(metagraph, optimizer, port)
        #allow server to start up on separate thread
        time.sleep(serverStartup)
        self.mini_batch = mini_batch
        self.mini_stochastic_iters = mini_stochastic_iters
        self.verbose = verbose
        self.shuffle = shuffle
        self.partition_shuffles= partition_shuffles
        self.loss_callback = loss_callback
        self.master_url = master_url if master_url is not None else HogwildSparkModel.determine_master(port)
        self.port = port

    @staticmethod
    def determine_master(port):
        """
        Get the url of the driver node. This is kind of crap on mac.
        """
        try:
            master_url = socket.gethostbyname(socket.gethostname()) + ':' + str(port)
            return master_url
        except:
            return 'localhost:' + str(port)

    def start_server(self, tg, optimizer, port):
        """
        Starts the server with a copy of the argument for weird tensorflow multiprocessing issues
        """
        try:
            multiprocessing.set_start_method('spawn')
        except Exception as e:
            pass
        self.server = Process(target=self.start_service, args=(tg, optimizer, port))
        self.server.daemon = True
        self.server.start()

    def stop_server(self):
        """
        Needs to get called when training is done
        """
        self.server.terminate()
        self.server.join()

    def start_service(self, metagraph, optimizer, port):
        """
        Asynchronous flask service. This may be a bit confusing why the server starts here and not init.
        It is basically because this is ran in a separate process, and when python call fork, we want to fork from this
        thread and not the master thread
        """
        app = Flask(__name__)
        self.app = app
        max_errors = self.iters
        lock = RWLock()

        server = tf.train.Server.create_local_server()
        new_graph = tf.Graph()
        with new_graph.as_default():
            tf.train.import_meta_graph(metagraph)
            loss_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(loss_variable, trainable_variables)
            grads = list(zip(grads, trainable_variables))
            train_op = optimizer.apply_gradients(grads)
            init = tf.global_variables_initializer()

        glob_session = tf.Session(server.target, graph=new_graph)
        with new_graph.as_default():
            with glob_session.as_default():
                glob_session.run(init)
                self.weights = tensorflow_get_weights(trainable_variables)

        cont = itertools.count()
        lock_acquired = self.acquire_lock

        @app.route('/')
        def home():
            return 'Lifeomic'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            if lock_acquired:
                lock.acquire_read()
            vs = pickle.dumps(self.weights)
            if lock_acquired:
                lock.release()
            return vs

        @app.route('/update', methods=['POST'])
        def update_parameters():
            with new_graph.as_default():
                gradients = pickle.loads(request.data)
                nu_feed = {}
                for x, grad_var in enumerate(grads):
                    nu_feed[grad_var[0]] = gradients[x]

                if lock_acquired:
                    lock.acquire_write()

                with glob_session.as_default():
                    try:
                        glob_session.run(train_op, feed_dict=nu_feed)
                        self.weights = tensorflow_get_weights(trainable_variables)
                    except:
                        error_cnt = cont.next()
                        if error_cnt >= max_errors:
                            raise Exception("Too many failures during training")
                    finally:
                        if lock_acquired:
                            lock.release()

            return 'completed'

        self.app.run(host='0.0.0.0', use_reloader=False, threaded=True, port=port)

    def train(self, rdd):
        try:
            tgraph = self.tensorflowGraph
            tfInput = self.tfInput
            tfLabel = self.tfLabel
            master_url = self.master_url
            iters = self.iters
            mbs = self.mini_batch
            msi = self.mini_stochastic_iters
            cb = self.loss_callback
            verbose = self.verbose
            shuffle = self.shuffle
            for i in range(self.partition_shuffles):
                rdd.foreachPartition(lambda x: handle_model(x, tgraph, tfInput,
                                                            tfLabel=tfLabel, master_url=master_url,
                                                            iters=iters, mini_batch_size=mbs, shuffle=shuffle,
                                                            mini_stochastic_iters=msi, verbose=verbose,
                                                            loss_callback=cb))
                if self.partition_shuffles - i > 1:
                    num_partitions = rdd.getNumPartitions()
                    rdd = rdd.repartition(num_partitions)
            server_weights = get_server_weights(master_url)
            self.stop_server()
            return server_weights
        except Exception as e:
            self.stop_server()
            raise e

