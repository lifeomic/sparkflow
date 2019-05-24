from multiprocessing import Process
import itertools
from sparkflow.RWLock import RWLock
from flask import Flask, request
import six.moves.cPickle as pickle
import tensorflow as tf
from sparkflow.ml_util import tensorflow_get_weights


class FlaskServer(Process):

    def __init__(self, metagraph, optimizer, port, max_errors, lock):
        Process.__init__(self)
        self.metagraph = metagraph
        self.optimizer = optimizer
        self.port = port
        self.max_errors = max_errors
        self.lock = lock

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

        server = tf.train.Server.create_local_server()
        new_graph = tf.Graph()
        with new_graph.as_default():
            tf.train.import_meta_graph(self.metagraph)
            loss_variable = tf.get_collection(tf.GraphKeys.LOSSES)[0]
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(loss_variable, trainable_variables)
            grads = list(zip(grads, trainable_variables))
            train_op = self.optimizer.apply_gradients(grads)
            init = tf.global_variables_initializer()

        glob_session = tf.Session(server.target, graph=new_graph)
        with new_graph.as_default():
            with glob_session.as_default():
                glob_session.run(init)
                self.weights = tensorflow_get_weights(trainable_variables)

        cont = itertools.count()
        lock_acquired = self.lock

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

        self.app.run(host='0.0.0.0', use_reloader=False, threaded=True, port=self.port)


class Server(object):

    def __init__(self, metagraph, optimizer, port, max_errors, lock):
        self.metagraph = metagraph
        self.optimizer = optimizer
        self.port = port
        self.lock = lock
        self.max_errors = max_errors
        self.server = self.start_server(metagraph, optimizer, port)

    def start_server(self, tg, optimizer, port):
        """
        Starts the server with a copy of the argument for weird tensorflow multiprocessing issues
        """
        server = FlaskServer(tg, optimizer, port, int(self.max_errors), bool(self.lock))
        server.daemon = True
        server.start()
        return server

    def stop_server(self):
        """
        Needs to get called when training is done
        """
        self.server.terminate()
        self.server.join()
