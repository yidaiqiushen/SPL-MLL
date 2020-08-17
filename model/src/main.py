import os
import time
import numpy as np
import tensorflow as tf
from model.src.parser import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import evaluate
from save_result import save_result
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Model(object):
    def __init__(self, config):
        self.epoch_count = 0
        self.config = config
        self.data = DataSet(config)
        self.add_placeholders()
        self.summarizer = tf.summary
        self.net = Network(config, self.summarizer)
        self.optimizer = self.config.solver.optimizer
        self.y_pred = self.net.prediction(self.x, self.keep_prob)
        self.loss = self.net.loss_function(self.x, self.y, self.keep_prob)
        self.accuracy = self.net.accuracy(self.y_pred, self.y)
        self.summarizer.scalar("accuracy", self.accuracy)
        self.summarizer.scalar("loss", self.loss)
        self.train = self.net.train_step(self.loss)
        self.B = self.net.B
        self.A = self.net.A
        self.n_epoch_to_decay = list(range(800, 20000, 1000))[::-1]
        self.next_epoch_to_decay = self.n_epoch_to_decay.pop()
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.local_init = tf.local_variables_initializer()
        self.kf = KFold(n_splits=10, random_state=0, shuffle=True)

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.features_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.labels_dim])
        self.keep_prob = tf.placeholder(tf.float32)

    def train_epoch(self, sess, summarizer):
        merged_summary = self.summarizer.merge_all()
        err, accuracy = list(), list()
        X, Y = self.data.get_train()
        for train, val in self.kf.split(X, y=Y):
            feed_dict = {self.x: X[train], self.y: Y[train], self.keep_prob: self.config.solver.dropout}
            # attention!
            summ, _, loss_, accuracy_ = sess.run([merged_summary, self.train,
                                                  self.loss, self.accuracy], feed_dict=feed_dict)
            summarizer.add_summary(summ)
            err.append(loss_)
            accuracy.append(accuracy_)
        return np.mean(err), np.mean(accuracy)

    def do_eval(self, sess, data):
        if data == "validation":
            err, accuracy = list(), list()
            X, Y = self.data.get_validation()
            for train, val in self.kf.split(X, y=Y):
                feed_dict = {self.x: X[val], self.y: Y[val], self.keep_prob: 1}
                loss_, Y_pred, accuracy_ = sess.run([self.loss, self.y_pred, self.accuracy], feed_dict=feed_dict)
                metrics = evaluate(predictions=Y_pred, labels=Y[val])
                err.append(loss_)
                accuracy.append(accuracy_)
            return np.mean(err), np.mean(accuracy), metrics

        if data == "test":
            X, Y = self.data.get_test()
            feed_dict = {self.x: X, self.y: Y, self.keep_prob: 1}
            loss_, Y_pred, accuracy_ = sess.run([self.loss, self.y_pred, self.accuracy], feed_dict=feed_dict)
            metrics = evaluate(predictions=Y_pred, labels=Y)
            return loss_, accuracy_, metrics

    def fit(self, sess, summarizer):
        sess.run(self.init)
        sess.run(self.local_init)
        max_epochs = self.config.max_epochs
        self.epoch_count = 0
        max_micro_f1 = 0
        max_macro_f1 = 0
        while self.epoch_count < max_epochs:
            if self.config.load:
                break
            loss_train, accuracy_train = self.train_epoch(sess, summarizer['train'])
            loss_val, accuracy_val, metrics_val = self.do_eval(sess, "validation")
            if self.epoch_count == self.next_epoch_to_decay:
                if len(self.n_epoch_to_decay) == 0:
                    self.next_epoch_to_decay = -1
                else:
                    self.next_epoch_to_decay = self.n_epoch_to_decay.pop()
                self.config.learning_rate *= self.config.lr_decay_factor
                print('Decaying learning rate ...')
                print(self.config.learning_rate)
            
            if max_micro_f1 < metrics_val['micro_f1'] and max_macro_f1 < metrics_val['macro_f1']:
                print(self.config.ckptdir_path)
                print("cur_max_Mi-F1 = %g, cur_max_Ma-F1 = %g, cur_epoch = %g." % (
                    metrics_val['micro_f1'], metrics_val['macro_f1'], self.epoch_count))
                self.saver.save(sess, self.config.ckptdir_path + "model.ckpt")
            max_micro_f1 = max(max_micro_f1, metrics_val['micro_f1'])
            max_macro_f1 = max(max_macro_f1, metrics_val['macro_f1'])

            if self.epoch_count % 5 == 0:
                print("After %d training epoch(s), Training : Loss = %g, Validation : Loss = %g." % (
                self.epoch_count, loss_train, loss_val))
                print("train_accuracy = %g, val_accuracy = %g." % (accuracy_train, accuracy_val))
                print("Micro-F1 = %g, Macro-F1 = %g." % (metrics_val['micro_f1'], metrics_val['macro_f1']))
            self.epoch_count += 1
        returnDict = {"train_loss": loss_train, "val_loss": loss_val, "train_accuracy": accuracy_train,
                      "val_accuracy": accuracy_val}
        return returnDict

    def add_summaries(self, sess):
        if self.config.load or self.config.debug:
            path_ = os.path.join("../results/tensorboard" + self.config.dataset_name)
        else:
            path_ = os.path.join("../bin/results/tensorboard" + self.config.dataset_name)
        summary_writer_train = tf.summary.FileWriter(path_ + "/train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(path_ + "/val", sess.graph)
        summary_writer_test = tf.summary.FileWriter(path_ + "/test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)

    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)
    tf_config = tf.ConfigProto(allow_soft_placement=True)  # device_count = {'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    if config.load:
        print("=> Loading model from checkpoint")
        model.saver.restore(sess, config.ckptdir_path + "model.ckpt")
        print(config.ckptdir_path)
    else:
        print("=> No model loaded from checkpoint")
    return model, sess


def test_model(config):
    print("\033[92m=>\033[0m Testing Model")
    print(config.dataset_name)
    model, sess = init_model(config)
    with sess:
        loss_test, accuracy_test, metrics_test = model.do_eval(sess, "test")
        return loss_test, accuracy_test, metrics_test


def train_model(config):
    print("\033[92m=>\033[0m Training Model")
    print(config.dataset_name)
    model, sess = init_model(config)
    with sess:
        summary_writers = model.add_summaries(sess)
        train_dict = model.fit(sess, summary_writers)
        return train_dict


def main():
    start_time = time.time()
    args = Parser().get_parser().parse_args()
    config = Config(args)
    if config.load:
        loss_test, accuracy_test, metrics_test = test_model(config)
        print("loss_test = %g, accuracy_test = %g." % (loss_test, accuracy_test))

        output = "\n=> Test : Ranking Loss = {}, Hamming Loss = {}, Average Precision = {}," \
                 "Micro-F1 = {}, Macro-F1 = {}".format(metrics_test['ranking_loss'],
                                                       metrics_test['hamming_loss'],
                                                       metrics_test['average_precision'],
                                                       metrics_test['micro_f1'], metrics_test['macro_f1'])
        print(output)
        save_result(config.dataset_name, metrics_test)
    else:
        train_dict = train_model(config)
        output = "=> Train Loss : {}, Validation Loss : {}, Validation Accuracy : {}, Train Accuracy : {}". \
            format(train_dict["train_loss"], train_dict["val_loss"], train_dict["val_accuracy"],
                   train_dict["train_accuracy"])
        # print(output)


if __name__ == '__main__':
    np.random.seed(1234)
    main()






