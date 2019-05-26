from multiprocessing import get_context
from multiprocessing import Process
import itertools
from sparkflow.RWLock import RWLock
from flask import Flask, request
import six.moves.cPickle as pickle
import tensorflow as tf
from google.protobuf import json_format
import os
from sparkflow.ml_util import tensorflow_get_weights
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Server(object):

    def __init__(self, metagraph, optimizer, port, max_errors, lock):
        self.metagraph = metagraph
        self.optimizer = optimizer
        self.port = port
        self.max_errors = max_errors
        self.lock = lock
        self.weights = None
        self.tensorflow_server = Process(target=self.start_local_server)
        try:
            self.server = get_context("spawn").Process(target=self.run)
        except Exception as e:
            self.server = Process(target=self.run())

    def start_local_server(self):
        cluster = tf.train.ClusterSpec({"local": ["localhost:2222"]})
        server = tf.train.Server(cluster, job_name="local", task_index=0)
        server.join()

    def start(self):
        self.tensorflow_server.start()
        self.server.start()

    def stop(self):
        self.server.terminate()
        self.tensorflow_server.terminate()
        self.server.join()
        self.tensorflow_server.join()

    def run(self):
        """
        Asynchronous flask service. This may be a bit confusing why the server starts here and not init.
        It is basically because this is ran in a separate process, and when python call fork, we want to fork from this
        thread and not the master thread
        """
        app = Flask(__name__)
        self.app = app
        max_errors = self.max_errors
        lock = RWLock()

        def get_graph():
            mgd = tf.MetaGraphDef()
            metagraph = json_format.Parse(self.metagraph, mgd)
            new_graph = tf.Graph()
            with new_graph.as_default():
                tf.train.import_meta_graph(metagraph)
                loss_variable = tf.get_collection(tf.GraphKeys.LOSSES)[0]
                trainable_variables = tf.trainable_variables()
                grads = tf.gradients(loss_variable, trainable_variables)
                grads = list(zip(grads, trainable_variables))
                train_op = self.optimizer.apply_gradients(grads)

            def inner():
                return new_graph, train_op, grads

            return inner

        g_func = get_graph()
        new_graph, _, _ = g_func()

        with tf.Session("grpc://localhost:2222", graph=new_graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.weights = tensorflow_get_weights(tf.trainable_variables())

        cont = itertools.count()
        lock_acquired = self.lock

        @app.route('/')
        def home():
            return 'Lifeomic'

        @app.route('/parameters', methods=['GET'])
        def get_parameters():
            if lock_acquired:
                lock.acquire_read()
            vs = pickle.dumps(self.weights, -1)
            if lock_acquired:
                lock.release()
            return vs

        @app.route('/update', methods=['POST'])
        def update_parameters():
            n_graph, train_op, grads = g_func()
            with tf.Session("grpc://localhost:2222", graph=n_graph) as sess:
                trainable_variables = tf.trainable_variables()
                gradients = pickle.loads(request.data)
                nu_feed = {}
                for x, grad_var in enumerate(grads):
                    nu_feed[grad_var[0]] = gradients[x]

                if lock_acquired:
                    lock.acquire_write()

                try:
                    sess.run(train_op, feed_dict=nu_feed)
                    self.weights = tensorflow_get_weights(trainable_variables, sess=sess)
                except:
                    error_cnt = cont.next()
                    if error_cnt >= max_errors:
                        raise Exception("Too many failures during training")
                finally:
                    if lock_acquired:
                        lock.release()

                return 'completed'

        self.app.run(host='0.0.0.0', use_reloader=False, threaded=True, port=self.port)


