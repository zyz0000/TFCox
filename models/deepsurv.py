import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from TFCox.utils import check_config
from TFCox.utils import check_surv_data
from TFCox.utils import prepare_surv_data
from TFCox.utils import baseline_survival_function
from TFCox.utils import plot_train_curve, plot_surv_curve
from TFCox.evals.concordance import concordance_index


class DeepSurv(object):
    def __init__(self, input_nodes, hidden_layer_nodes, config={}):
        '''
        Deep Survival Neural Network Class Constructor.
        :param input_nodes: int
               Equal to the number of features(covariates).
        :param hidden_layer_nodes:list
               The number of nodes in hidden layers.
        :param config:dict
               Configuration of hyper-parameters. Default configuration is below:
                default_network_config = {
                'learning_rate' : 0.01,
                'learning_rate_decay': 1.0,
                'activation': 'relu',
                'L2_reg': 0.0,
                'L1_reg': 0.0,
                'optimizer': 'adam',
                'dropout_rate': 1.0,
                'seed': 0
                }
        '''
        super(DeepSurv, self).__init__()
        #neural nodes setting
        self.input_nodes = input_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        assert hidden_layer_nodes[-1] == 1 #To check if the output is a one-dimensional number.
        #network hyper-parameters
        check_config(config)
        self.config = config

        #graph level random seed
        tf.compat.v1.random.set_random_seed(config['seed'])

        self.global_step = tf.compat.v1.get_variable('global_step',
                                            initializer=tf.constant(0),
                                            trainable=False)
        self.dropout_rate = tf.compat.v1.placeholder(tf.float32)
        self.X = tf.compat.v1.placeholder(tf.float32, [None, input_nodes], name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='Y')


    def create_fc_layer(self, x, output_dim, scope):
        '''
        create full-connected layer
        :param x:features.
        :param output_dim:the dimensionality of output.
        :param scope:parameter scope.
        :return:full-connected layer.
        '''
        with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):
            #Initialize weights and bias
            w = tf.compat.v1.get_variable('weights', [x.shape[1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.compat.v1.get_variable('bias', [output_dim],
                             initializer=tf.constant_initializer(0.0))
            #Add weights and bias to collections
            tf.add_to_collection('var_weight', w)
            tf.add_to_collection('var_bias', b)

            layer_out = tf.nn.dropout(tf.matmul(x, w)+b, rate=self.dropout_rate)

            if self.config['activation'] == 'relu':
                layer_out = tf.nn.relu(layer_out)
            elif self.config['activation'] == 'sigmoid':
                layer_out = tf.nn.sigmoid(layer_out)
            elif self.config['activation'] == 'tanh':
                layer_out = tf.nn.tanh(layer_out)
            elif self.config['activation'] == 'leaky_relu':
                layer_out = tf.nn.leaky_relu(layer_out)
            else:
                raise NotImplementedError("Activation not recognized. "
                                          "Must be one of 'relu', 'sigmoid', 'tanh', 'leaky_relu'. ")

            return layer_out


    def create_network(self):
        with tf.name_scope('hidden_layers'):
            x = self.X
            for i, num_nodes in enumerate(self.hidden_layer_nodes, start=1):
                x = self.create_fc_layer(x, num_nodes, 'layer'+str(i))
            self.Y_hat = x


    def create_loss_function(self):
        with tf.name_scope('loss'):
            #obtain T and E from self.Y. Negative values of T mean E is 0.
            Y = tf.squeeze(self.Y)
            Y_hat = tf.squeeze(self.Y_hat)
            Y_label_T = tf.abs(Y)
            Y_label_E = tf.cast(tf.greater(Y, 0), tf.int32)
            obs = tf.cast(tf.reduce_sum(Y_label_E), tf.float32)

            Y_hat_hr = tf.exp(Y_hat)
            Y_hat_cumsum = tf.math.log(tf.cumsum(Y_hat_hr))

            #start calculating loss function
            #get segment from T
            unique_vals, segment_ids = tf.unique(Y_label_T)
            #get segment max
            loss_s2_v = tf.math.segment_max(Y_hat_cumsum, segment_ids)
            #get segment count
            loss_s2_count = tf.math.segment_sum(Y_label_E, segment_ids)
            #compute S2
            loss_s2 = tf.reduce_sum(tf.multiply(tf.cast(loss_s2_v, tf.float32),
                                                tf.cast(loss_s2_count, tf.float32)))
            #compute S1
            loss_s1 = tf.reduce_sum(tf.multiply(Y_hat, tf.cast(Y_label_E, tf.float32)))
            #compute Breslow loss
            loss_Breslow = tf.math.divide(tf.subtract(loss_s2, loss_s1), obs)

            #compute regularization term
            reg_item = tf.contrib.layers.l1_l2_regularizer(
                scale_l1=self.config['L1_reg'], scale_l2=self.config['L2_reg']
            )
            loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.get_collection("var_weight"))
            #total loss = loss_Breslow + l1_norm_loss + l2_norm_loss
            self.loss = tf.add(loss_Breslow, loss_reg)


    def create_optimizer(self):
        #SGD optimizer
        if self.config['optimizer'] == 'sgd':
            lr = tf.train.exponential_decay(
                self.config['learning_rate'],
                self.global_step,
                1,
                self.config['learning_rate_decay']
            )
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(
                self.loss, global_step=self.global_step)
        #Adam optimizer
        elif self.config["optimizer"] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)
        #RMSprop optimizer
        elif self.config["optimizer"] == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(
                self.config["learning_rate"]).minimize(self.loss,global_step=self.global_step)
        else:
            raise NotImplementedError('Optimizer not recognized')


    def build_graph(self):
        self.create_network()
        self.create_loss_function()
        self.create_optimizer()
        self.sess = tf.Session()


    def close_session(self):
        self.sess.close()
        print('Current session closed')


    def train(self, X, y, num_steps,
              batch_size=1,
              save_path='',
              load_path='',
              plot=False,
              silent=False):
        '''
        Train the neural network.
        :param X: pandas.DataFrame
               Covariates of survival data.
        :param y: pandas.DataFrame
               Labels of survival data.
        :param num_steps: int
               Number of training steps.
        :param mini_batch:int
               The number of skipping training steps. Model would be saved after each `batch_size`.`
        :param save_path: string
               The path to save model.None means no saving.
        :param load_path: string
               The path to load model.None means no loading.
        :param plot: boolean
               Whether to plot the training curves.
        :param silent: boolean
               Whether to print info on the screen.
        :return:dict
               C-index and loss function during training.
        '''

        self.indices, self.X_train, self.y_train = prepare_surv_data(X, y)
        feed_dict = {
            self.dropout_rate: self.config['dropout_rate'],
            self.X : self.X_train.values,
            self.Y : self.y_train.values
        }
        self.sess.run(tf.global_variables_initializer())

        if load_path != '':
            saver = tf.train.Saver()
            saver.restore(self.sess, load_path)

        total_loss = 0.0
        tf.reset_default_graph()
        initial_step = self.global_step.eval(session=self.sess)
        watch_list = {'loss': [], 'metrics':[]}
        for index in range(initial_step, initial_step + num_steps):
            y_hat, loss, _ = self.sess.run([self.Y_hat, self.loss, self.optimizer],
                                           feed_dict=feed_dict)
            watch_list['loss'].append(loss)
            watch_list['metrics'].append(concordance_index(self.y_train.values, -y_hat))
            total_loss += loss
            if (index+1) % batch_size ==0:
                if silent is False:
                    print('Average loss at step {}: {:.4f}, metrics at step {}: {:.4f}'.format(
                        index+1, total_loss / batch_size, index+1,
                        concordance_index(self.y_train.values, -y_hat)
                    ))
                total_loss = 0.0

        if save_path != '':
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)

        if plot:
            plot_train_curve(watch_list['loss'], watch_list['metrics'])

        # update baseline survival function
        self.hr = self.predict(self.X_train, log=False)
        # use training data to estimate the baseline survival function BSF
        self.BSF = baseline_survival_function(
            np.squeeze(np.greater(self.y_train.values, 0).astype(np.int)),
            np.squeeze(self.y_train.values),
            np.squeeze(self.hr)
        )
        return watch_list

    def predict(self, X, log=True):
        '''
        Predict the log hazard rate using training data.
        :param X:pandas.DataFrame
               Input data with covariates with shape (n_obs, input_nodes).
        :param log:boolean
               If True, the return value will be log hazard rate, otherwise real hazard rate.
        :return:numpy.array
               Predicted log hazard rate, or real hazard rate with shape (n_obs, 1).
        '''
        #dropout_rate should be set to 1.0 when making prediction.
        log_hr = self.sess.run([self.Y_hat],
                               feed_dict={self.X: X.values, self.dropout_rate: 0.0})
        log_hr = log_hr[0]
        return (log_hr if log else np.exp(log_hr))

    def eval(self, X, y):
        '''
        Evaluate labeled data using the CI metrics under current trained model.
        :param X:pandas.DataFrame
               Covariates of survival data.
        :param y:pandas.DataFrame
               Labels of survival data.
        :return:float
               CI metrics.
        '''
        pred = - self.predict(X)
        return concordance_index(y.values, pred)

    def predict_surv_func(self, new_X, plot=False):
        '''
        Predict survival function based on newly observed X.
        :param new_X:pandas.DataFrame
            Input data with covariates, shape of which is (n, input_nodes).
        :param plot:boolean
            Whether to plot the estimated sample survival curve.
        :return:DataFrame
            Predicted survival function.
        '''
        pred_hr = self.predict(new_X, log=False)
        surv_func = pd.DataFrame(
            self.BSF.iloc[:,0].values ** (pred_hr),
            columns=self.BSF.index.values
        )
        surv_func = surv_func.loc[:, filter(lambda x: x>=0, surv_func.columns)]
        if plot:
            plot_surv_curve(surv_func, title='Estimated Survival Function $\hat{S}(x)$')
        return surv_func