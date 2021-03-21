import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    # initial = tf.constant(0.1, shape=shape)
    #initial = tf.zeros(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ModelCNNMnist(ModelCNNAbstract):

    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.W_conv2 = weight_variable([5, 5, 32, 32])
        self.b_conv2 = bias_variable([32])
        self.W_fc1 = weight_variable([7 * 7 * 32, 256])
        self.b_fc1 = bias_variable([256])
        self.W_fc2 = weight_variable([256, 10])
        self.b_fc2 = bias_variable([10])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_norm1 = tf.nn.lrn(self.h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
        self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_pool2 = max_pool_2x2(self.h_norm2)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 32])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        #self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)



        self.all_weights = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

        self.star_vars = []
        for v in range(len(self.all_weights)):
            self.star_vars.append(tf.zeros(self.all_weights[v].shape))  # without sess ??

        # get total number of weights
        self.n_all_weights = 0
        for v in self.all_weights:
           self.n_all_weights += tf.reshape(v,[-1]).shape[0]
        self.n_all_weights=tf.dtypes.cast(self.n_all_weights, tf.float32)
        #print(type(n_all_weights))


        self.L1regularizer = 0
        for v in range(len(self.all_weights)):
            self.L1regularizer += tf.reduce_sum(abs(self.all_weights[v]))

        self.L2regularizer=0
        for v in range(len(self.all_weights)):
            self.L2regularizer+=tf.reduce_sum(tf.math.square(self.all_weights[v]))

        self.EWCregularizer=0
        #print(self.star_vars)
        #for v in range(len(self.all_weights)):
         #   self.EWCregularizer += tf.reduce_sum(tf.math.square(self.all_weights[v]-self.star_vars[v]))





        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.math.log(self.y), reduction_indices=[1]))+ tf.math.divide(1,self.n_all_weights)*self.L2regularizer

                             #+ tf.math.divide(0.02,self.n_all_weights) * self.EWCregularizer

        # if self.star_vars == []:
        #     self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        # else:
        #     self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))



        self._assignment_init()

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True

    def compute_fisher(self, imgset, num_samples=200):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.fisher = []
        for v in range(len(self.all_weights)):
            self.fisher.append(np.zeros(self.all_weights[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.math.log(probs), 1)[0][0])

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = self.session.run(tf.gradients(tf.math.log(probs[0, class_ind]), self.all_weights),
                                    feed_dict={self.x: imgset[im_ind:im_ind + 1]})
            # square the derivatives and add to total
            for v in range(len(self.fisher)):
                self.fisher[v] += np.square(ders[v])

        # divide totals by number of samples
        for v in range(len(self.fisher)):
            self.fisher[v] /= num_samples

        return self.fisher


    def star(self,train_image):
        # used for saving optimal weights after most recent task training
        self.star_vars=[]
        for v in range(len(self.all_weights)):
            self.star_vars.append(self.session.run(self.all_weights[v]))  # without sess ??

        self.fisher = self.compute_fisher(train_image)

        self.EWCregularizer = 0
        #print(self.star_vars)
        for v in range(len(self.all_weights)):
            self.EWCregularizer += tf.reduce_sum(tf.multiply(self.fisher[v].astype(np.float32),tf.math.square(self.all_weights[v] - self.star_vars[v])))



        #self.cross_entropy = tf.reduce_mean(
         #   -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) + tf.math.divide(100,
                                                                                              #self.n_all_weights) * self.EWCregularizer

        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.math.log(self.y), reduction_indices=[1])) + tf.math.divide(0.5,2) *self.EWCregularizer


        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        # self.sess = tf.Session()

    def gradient_model(self, index, imgs, labels, w, sampleIndices):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)


        #start=time.time()


        grad_var_list = self.session.run(self.grad_model,
            feed_dict={self.x: [imgs[i] for i in sampleIndices],
                       self.y_: [labels[i] for i in sampleIndices],self.index :  [0, index]})


        #print('time',time.time()-start)
        #print(np.array(grad_var_list).shape)

        grad_flatten_list = []

        for l in grad_var_list:
            grad_flatten_list.append(np.reshape(l[0], l[0].size))
            #grad_flatten_list.append(l[0])
            # counter+=len(np.reshape(l[0], l[0].size))
            # delimitation.append(counter)

        grad_flatten_array = np.hstack(grad_flatten_list)

        del grad_var_list
        del grad_flatten_list


        return grad_flatten_array




