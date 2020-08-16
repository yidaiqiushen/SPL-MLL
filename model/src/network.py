import sys
import utils
import numpy as np
import tensorflow as tf
from tensor_assign_2D import tensor_assign_2D

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class Network(object):
    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variables()
        self.d = self.config.labels_dim
        self.B = tf.get_variable(name="B", initializer=tf.eye(self.d))
        self.A = tf.get_variable(name="A", shape=[self.d, self.d], initializer=tf.truncated_normal_initializer())


    def weight_variable(self, shape, name):
        # return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_Fx_variables(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim1], "weight_x1")
        W2 = self.weight_variable([self.config.solver.hidden_dim1, self.config.solver.hidden_dim2], "weight_x2")
        W3 = self.weight_variable([self.config.solver.hidden_dim2, self.config.labels_dim], "weight_x3")
        b1 = self.bias_variable([self.config.solver.hidden_dim1], "bias_x1")
        b2 = self.bias_variable([self.config.solver.hidden_dim2], "bias_x2")
        b3 = self.bias_variable([self.config.labels_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    def Fx(self, X, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob)
        hidden2 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        y_embedding = tf.nn.sigmoid(hidden3)
        return y_embedding

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def prediction(self, X, keep_prob):
        f = self.Fx(X, keep_prob)
        y_pred = tf.matmul(tf.matmul(f, self.B), self.A)
        return y_pred

    def loss_function(self, X, Y, keep_prob):
        C1 = tf.matmul(tf.cast((self.Fx(X, keep_prob) - Y), tf.float32), self.B)
        C2 = Y - tf.matmul(tf.matmul(Y, self.B), tf.cast(self.A, tf.float32))
        C3 = self.B - tf.eye(self.d)
        Bi = []
        for i in range(self.d):
            Bi.append(tf.norm(self.B[i, :]))
        L = self.config.lambda1*tf.trace(tf.matmul(C3, tf.transpose(C3))) \
            + self.config.lambda2*tf.reduce_sum(Bi)
        E = tf.trace(tf.matmul(C1, tf.transpose(C1))) + tf.trace(tf.matmul(C2, tf.transpose(C2))) + L
        return E

    def train_step(self, loss):
        # Optional operation ：At each iteration, the off-diagonal elements are set to 0.
        # for i in range(self.d):
        #      for j in range(self.d):
        #          if i != j:
        #              self.B = tensor_assign_2D(self.B, [i, j], 0)
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)

