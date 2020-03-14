## coding utf-8

'''
Declaration of the network configuration:

Inputs:
    - input_dims: dictionary of dimension information
        x_dim: dimension of features
        num_event: number of competing events (this does not include censoring label)
        num_category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
                      which is equivalent to the output dimension
    - config:
        h_dim_shared & num_layers_shared: number of nodes and number of fully-connected layers for the shared subnetwork
        h_dim_CS & num_layers_CS: number of nodes and number of fully-connected layers for the cause-specific subnetworks
        activation: 'relu', 'selu', 'tanh' etc.
        initial_weight: Xavier initialization is used as a baseline.
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as FC

class DeepHit:
    def __init__(self, sess, name, input_dims, config):
        self.sess = sess
        self.name = name

        # input dimensions
        self.x_dim = input_dims['x_dim']
        self.num_event = input_dims['num_event']
        self.num_category = input_dims['num_category']

        # network configuration
        self.h_dim_shared = config['h_dim_shared']
        self.h_dim_CS = config['h_dim_CS']
        self.num_layers_shared = config['num_layers_shared']
        self.num_layers_CS = config['num_layers_CS']

        self.activation = config['activation']
        self.initial_weight = tf.contrib.layers.xavier_initializer()
        self.reg_weight     = tf.contrib.layers.l2_regularizer(scale=1.0)
        self.reg_weight_out = tf.contrib.layers.l1_regularizer(scale=1.0)

        self.build_net()


    def build_net(self):
        with tf.variable_scope(self.name):
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            self.alpha = tf.placeholder(tf.float32, [], name='alpha')
            self.beta = 1. - self.alpha
            self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='inputs')
            self.e = tf.placeholder(tf.float32, [None, 1], name='events')
            self.t = tf.placeholder(tf.float32, [None, 1], name='time')
            self.fc_mask1 = tf.placeholder(tf.float32,
                                           shape=[None, self.num_event, self.num_category],
                                           name='mask1')
            self.fc_mask2 = tf.placeholder(tf.float32,
                                           shape=[None, self.num_category],
                                           name='mask2')
            ## shared subnetworks
            shared_out = create_FCNet(self.x, self.num_layers_shared, self.h_dim_shared,
                                      self.activation, self.h_dim_shared, self.activation,
                                      self.initial_weight, self.keep_prob, self.reg_weight)
            last_x = self.x
            h = tf.concat([last_x, shared_out], axis=1)
            out = []
            for _ in range(self.num_event):
                cs_out = create_FCNet(h, self.num_layers_CS, self.h_dim_CS, self.activation,
                                      self.h_dim_CS, self.activation, self.initial_weight,
                                      self.keep_prob, self.reg_weight)
                out.append(cs_out)
            out = tf.stack(out, axis=1)
            out = tf.reshape(out, [-1, self.num_event * self.h_dim_CS])
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)
            out = FC(out, self.num_event * self.num_category, activation_fn=tf.nn.softmax,
                         weights_initializer=self.initial_weight, weights_regularizer=self.reg_weight_out,
                         scope='output')
            self.out = tf.reshape(out, [-1, self.num_event, self.num_category])

                ## compute loss
            self.loss_log_likelihood()
            self.loss_ranking()
            self.total_loss = self.alpha * self.LOSS_1 + self.beta * self.LOSS_2
            self.minimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.total_loss)


    def loss_log_likelihood(self):
        event = tf.sign(self.e)
        tmp = tf.reduce_sum(
            tf.reduce_sum(self.fc_mask1 * self.out, reduction_indices=2),
            reduction_indices=1, keepdims=True)
        uncensored_likelihood = event * tf.log(tmp)
        censored_likelihood = (1. - event) * tf.log(tmp)
        self.LOSS_1 = - tf.reduce_mean(uncensored_likelihood + 1. * censored_likelihood)


    def loss_ranking(self):
        sigma = tf.constant(0.1, dtype=tf.float32)
        eta = []
        for event in range(self.num_event):
            vec = tf.ones_like(self.t, dtype=tf.float32)
            ind = tf.cast(tf.equal(self.e, event+1), dtype=tf.float32)
            ind = tf.diag(tf.squeeze(ind))
            tmp_e = tf.reshape(tf.slice(self.out,
                               [0, event, 0], [-1, 1, -1]), [-1, self.num_category])
            R = tf.matmul(tmp_e, tf.transpose(self.fc_mask2))  # no need to divide by each individual dominator
            # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

            diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
            R = tf.matmul(vec, tf.transpose(diag_R)) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
            R = tf.transpose(R)  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

            T = tf.nn.relu(
                tf.sign(tf.matmul(vec, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(vec))))
            # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

            T = tf.matmul(ind, T)  # only remains T_{ij}=1 when event occured for subject i

            tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma), reduction_indices=1, keep_dims=True)

            eta.append(tmp_eta)
        eta = tf.stack(eta, axis=1)  # stack referenced on subjects
        eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_event]), reduction_indices=1, keep_dims=True)

        self.LOSS_2 = tf.reduce_sum(eta)  # sum over num_Events


    def compute_loss(self, data, mask, params, keep_prob, lr):
        '''
    Loss function:
        - 1. log likelihood (this includes log-likelihood of subjects who are censored)
        - 2. ranking loss (this is calculated only for acceptable pairs. For more information about the
             definition, please refer to

             `Deephit: A deep learning approach to survival analysis with competing risks`)
        '''
        (x_mb, e_mb, t_mb) = data
        (m1_mb, m2_mb) = mask
        (alpha, beta) = params
        return self.sess.run(self.total_loss, feed_dict={
            self.x:x_mb, self.e:e_mb, self.t:t_mb, self.fc_mask1:m1_mb,
            self.fc_mask2:m2_mb, self.alpha:alpha, self.beta:beta,
            self.keep_prob:keep_prob, self.learning_rate:lr})


    def train(self, data, mask, params, keep_prob, lr):
        (x_mb, e_mb, t_mb) = data
        (m1_mb, m2_mb) = mask
        (alpha, beta) = params
        return self.sess.run([self.minimizer, self.total_loss], feed_dict={
            self.x:x_mb, self.e:e_mb, self.t:t_mb, self.fc_mask1: m1_mb,
            self.fc_mask2:m2_mb, self.alpha:alpha, self.beta:beta,
            self.batch_size: np.shape(x_mb)[0], self.keep_prob:keep_prob,
            self.learning_rate:lr})


    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.out,
                             feed_dict={self.x: x_test,
                                        self.batch_size: np.shape(x_test)[0],
                                        self.keep_prob: keep_prob})


def create_FCNet(inputs, num_layers, h_dim, h_fn, o_dim,
                 o_fn, w_init=None,
                 keep_prob=1.0, w_reg=None):
    '''
        GOAL             : Create FC network with different specifications
        inputs (tensor)  : input tensor
        num_layers       : number of layers in FCNet
        h_dim  (int)     : number of hidden units
        h_fn             : activation function for hidden layers (default: tf.nn.relu)
        o_dim  (int)     : number of output units
        o_fn             : activation function for output layers (defalut: None)
        w_init           : initialization for weight matrix (defalut: Xavier)
        keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
    '''
    # default active functions (hidden: relu, out: None)
    if h_fn is None:
        h_fn = tf.nn.relu
    if o_fn is None:
        o_fn = None

    # default initialization functions (weight: Xavier, bias: None)
    if w_init is None:
        w_init = tf.contrib.layers.xavier_initializer()

    for layer in range(num_layers):
        if num_layers == 1:
            out = FC(inputs, o_dim, activation_fn=o_fn,
                     weights_initializer=w_init, weights_regularizer=w_reg)
        else:
            if layer == 0:
                h = FC(inputs, h_dim, activation_fn=h_fn,
                       weights_initializer=w_init, weights_regularizer=w_reg)
                if not keep_prob is None:
                    h = tf.nn.dropout(h, keep_prob=keep_prob)

            elif layer > 0 and layer != (num_layers-1):
                h = FC(h, h_dim, activation_fn=h_fn,
                       weights_initializer=w_init, weights_regularizer=w_reg)
                if not keep_prob is None:
                    h = tf.nn.dropout(h, keep_prob=keep_prob)

            else:
                out = FC(h, o_dim, activation_fn=o_fn,
                         weights_initializer=w_init, weights_regularizer=w_reg)
    return out