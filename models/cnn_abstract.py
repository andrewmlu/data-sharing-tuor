import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import abc

LOSS_ACC_BATCH_SIZE = 100 # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class ModelCNNAbstract(abc.ABC):

    def __init__(self):
        self.graph_created = False
        pass

    @abc.abstractmethod
    def create_graph(self, learning_rate=None):
        # The below variables need to be defined in the child class
        self.all_weights = None
        self.x = None
        self.context=None
        self.context_l = None
        self.y_ = None
        self.y = None
        self.cross_entropy = None
        self.cross_entropy_vector = None
        self.acc = None

        self.init = None
        self.all_assignment_placeholders = None
        self.all_assignment_operations = None

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = None


        self.session = None  # Used for consecutive training
        self.operation=None
        self.grad_validation_placeholder=None
        self.test=None
        self.test2=None





    def _optimizer_init(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = 0.0   # The learning rate should not have effect when not using optimizer
        self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.optimizer_op = self.optimizer.minimize(self.cross_entropy)

    def _assignment_init(self):
        self.init = tf.compat.v1.global_variables_initializer()

        self.all_assignment_placeholders = []
        self.all_assignment_operations = []
        for w in self.all_weights:
            p = tf.compat.v1.placeholder(tf.float32, shape=w.get_shape())
            self.all_assignment_placeholders.append(p)
            self.all_assignment_operations.append(w.assign(p))

    def _session_init(self):
        self.session = tf.compat.v1.Session()

    def get_weight_dimension(self, imgs, labels):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        dim = 0

        for weight in self.all_weights:
            tmp = 1
            # l = sess.run(tf.shape(weight))
            l = weight.get_shape()
            for i in range(0, len(l)):
                tmp *= l[i].value

            dim += tmp

        return dim

    def get_init_weight(self, dim, rand_seed=None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        if rand_seed is not None:
            # Random seed only works at graph initialization, so recreate graph here
            self.session.close()
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(rand_seed)
            self.create_graph(learning_rate=self.learning_rate)  # This creates the session as well

        # with tf.Session() as sess:
        self.session.run(self.init)

        weight_flatten_list = []
        for weight in self.all_weights:
            weight_var = self.session.run(weight)
            weight_flatten_list.append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array = np.hstack(weight_flatten_list)


        #dim_w = self.get_weight_dimension(imgs, labels)
        self.grad_validation_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(dim))
        grad_local_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(dim))

        grad_var_list=self.grad
        grad_flatten_list = []
        for l in grad_var_list:
            grad_flatten_list.append(tf.reshape(l[0],[-1]))


        self.test=tf.reduce_sum(tf.cast(tf.equal(tf.sign(tf.concat(grad_flatten_list,0)),tf.sign(self.grad_validation_placeholder)),tf.float32))

        #tf.reduce_sum(tf.cast(myOtherTensor, tf.float32))


        self.operation=tf.norm(tf.subtract(tf.concat(grad_flatten_list,0), self.grad_validation_placeholder))

        #self.operation2=tf.norm(self.h_fc1,axis=1)




            # sess.close()

        return weight_flatten_array



    def assign_flattened_weight(self, sess, w):
        start_index = 0

        for k in range(0, len(self.all_weights)):
            weight = self.all_weights[k]

            tmp = 1
            # l = sess.run(tf.shape(weight))
            l = weight.get_shape()
            for i in range(0, len(l)):
                tmp *= l[i].value

            weight_var = np.reshape(w[start_index : start_index+tmp], l)
            # sess.run(weight.assign(weight_var))
            sess.run(self.all_assignment_operations[k], feed_dict={self.all_assignment_placeholders[k]: weight_var})

            del weight_var

            start_index = start_index + tmp



    def gradient_context(self, imgs, labels, w, sampleIndices,context_0,context_l):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        grad_var_list = self.session.run(self.grad, feed_dict={self.x: [imgs[i] for i in sampleIndices], self.y_: [labels[i] for i in sampleIndices], self.context: context_0, self.context_l: context_l})

        grad_flatten_list = []
        #delimitation=[]
        #delimitation.append(0)
        #counter=0

        for l in grad_var_list:
            grad_flatten_list.append(np.reshape(l[0], l[0].size))
            #counter+=len(np.reshape(l[0], l[0].size))
            #delimitation.append(counter)


        grad_flatten_array = np.hstack(grad_flatten_list)

        del grad_var_list
        del grad_flatten_list

            # sess.close()

        #np.save('delimitation-cnn.npy',delimitation)

        return grad_flatten_array

    def get_softmax_output_minus_true_label(self,imgs, labels,sampleIndices):
        #return (self.session.run(self.y, feed_dict={self.x: [imgs[i] for i in sampleIndices]}),labels[0])
        for i in sampleIndices:
            diff_outputsoftmax_truelabel = self.session.run(self.y, feed_dict={self.x: [imgs[i]]})

            #print(diff_outputsoftmax_truelabel[0]-labels[i])

        return diff_outputsoftmax_truelabel[0]-labels[i]





    #gradient of the loss
    def gradient(self, imgs, labels, w, sampleIndices):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        grad_var_list = self.session.run(self.grad, feed_dict={self.x: [imgs[i] for i in sampleIndices], self.y_: [labels[i] for i in sampleIndices]})


        grad_flatten_list = []

        for l in grad_var_list:
            grad_flatten_list.append(np.reshape(l[0], l[0].size))
            #counter+=len(np.reshape(l[0], l[0].size))
            #delimitation.append(counter)


        grad_flatten_array = np.hstack(grad_flatten_list)

        del grad_var_list
        del grad_flatten_list



            # sess.close()

        #np.save('delimitation-cnn.npy',delimitation)

        return grad_flatten_array




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

    def get_difference(self, imgs, labels, w, sampleIndices,val_grad):
        #tf.reset_default_graph()
        #self.create_graph()

        #if not self.graph_created:
            #raise Exception('Graph is not created. Call create_graph() first.')

        diff_sign = self.session.run(self.test,
                                feed_dict={self.grad_validation_placeholder: val_grad,
                                           self.x: [imgs[i] for i in sampleIndices],
                                           self.y_: [labels[i] for i in sampleIndices]})
        #print(test)

        #diff=self.session.run(self.operation,
               #feed_dict={self.grad_validation_placeholder:val_grad,self.x: [imgs[i] for i in sampleIndices],
                                                               #self.y_: [labels[i] for i in sampleIndices]})

        return diff_sign

    def norm_last_layer_output(self, imgs, labels,sampleIndices=None):

        if sampleIndices==None:
            last_layer = self.session.run(self.operation2, feed_dict={self.x: imgs,
                                                                 self.y_: labels})
        else:

            last_layer=self.session.run(self.operation2, feed_dict={self.x: [imgs[i] for i in sampleIndices],
                                               self.y_: [labels[i] for i in sampleIndices]})

        return last_layer



    def loss_context(self, imgs, labels,context,context_l, w, sampleIndices = None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        if sampleIndices is None:
            sampleIndices = range(0,len(labels))

        val = 0
        l = []
        for k in range(0, len(sampleIndices)):
            l.append(sampleIndices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sampleIndices) - 1:
                val += self.session.run(self.cross_entropy,
                                     feed_dict={self.x: [imgs[i] for i in l],
                                                self.y_: [labels[i] for i in l],self.context :context,self.context_l:context_l}) \
                       * float(len(l)) / len(sampleIndices)

                l = []

            # sess.close()

        return val

    def loss(self, imgs, labels, w, sampleIndices=None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        val = 0
        l = []
        for k in range(0, len(sampleIndices)):
            l.append(sampleIndices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sampleIndices) - 1:
                val += self.session.run(self.cross_entropy,
                                        feed_dict={self.x: [imgs[i] for i in l],
                                                   self.y_: [labels[i] for i in l]}) \
                       * float(len(l)) / len(sampleIndices)

                l = []

            # sess.close()

        return val


    def loss_vector(self, imgs, labels, w, sampleIndices=None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        val = 0
        l = []

        val = self.session.run(self.cross_entropy_vector,
                                        feed_dict={self.x: [imgs[i] for i in sampleIndices],
                                                   self.y_: [labels[i] for i in sampleIndices]})


        return val

    def accuracy_context(self,imgs,labels,context,context_l, w,sampleIndices = None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        val = 0
        l = []
        for k in range(0, len(sampleIndices)):
            l.append(sampleIndices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sampleIndices) - 1:

                val += self.session.run(self.acc, feed_dict={self.x: [imgs[i] for i in l],
                                                          self.y_: [labels[i] for i in l],self.context:context, self.context_l:context_l}) \
                       * float(len(l)) / len(sampleIndices)

                l = []

            # sess.close()

        return val

    def accuracy(self,imgs,labels,w,sampleIndices = None):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        val = 0
        l = []
        for k in range(0, len(sampleIndices)):
            l.append(sampleIndices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sampleIndices) - 1:

                val += self.session.run(self.acc, feed_dict={self.x: [imgs[i] for i in l],
                                                          self.y_: [labels[i] for i in l]}) \
                       * float(len(l)) / len(sampleIndices)

                l = []

            # sess.close()

        return val




    def start_consecutive_training(self, w_init):
        # tf.reset_default_graph()
        # self.create_graph(learning_rate=learning_rate)

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # self.session = tf.Session()
        self.assign_flattened_weight(self.session, w_init)

    def end_consecutive_training_and_get_weights(self):
        weight_flatten_list = []
        for weight in self.all_weights:
            weight_var = self.session.run(weight)
            weight_flatten_list.append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array = np.hstack(weight_flatten_list)

        # self.session.close()
        # self.session = None

        return weight_flatten_array



    def run_one_step_consecutive_training(self, imgs, labels, sampleIndices):
        self.session.run(self.optimizer_op,
                 feed_dict={self.x: [imgs[i] for i in sampleIndices], self.y_: [labels[i] for i in sampleIndices]})

    def predict(self, img, w):
        # tf.reset_default_graph()
        # self.create_graph()

        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        # with tf.Session() as sess:
        self.assign_flattened_weight(self.session, w)

        pred = self.session.run(self.y, feed_dict={self.x: [img]})
            # sess.close()

        return pred[0]   # self.y gives an array of predictions
