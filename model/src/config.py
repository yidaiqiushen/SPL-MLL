import os
import utils
import tensorflow as tf


class Config(object):
    def __init__(self, args):
        self.codebase_root_path = args.path
        self.folder_suffix = args.folder_suffix
        self.project_name = args.project
        self.dataset_name = args.dataset
        self.retrain = args.retrain
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.patience_increase = args.patience_increase
        self.improvement_threshold = args.improvement_threshold
        self.save_after = args.save_after
        self.epoch_freq = args.epoch_freq
        self.debug = args.debug
        self.load = args.load
        self.have_patience = args.have_patience
        self.learning_rate = args.lr
        self.lr_decay_factor = args.lr_decay_factor
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        class Solver(object):
            def __init__(self, t_args):
                self.learning_rate = t_args.lr
                self.lr_decay_factor = args.lr_decay_factor
                self.hidden_dim1 = t_args.hidden1
                self.hidden_dim2 = t_args.hidden2
                self.dropout = t_args.dropout
                if t_args.opt.lower() not in ["adam", "rmsprop", "sgd"]:
                    raise ValueError('Undefined type of optmizer')
                else:
                    self.optimizer = {"adam": tf.train.AdamOptimizer, "rmsprop": tf.train.RMSPropOptimizer, "sgd": tf.train.GradientDescentOptimizer}[t_args.opt.lower()]

        self.solver = Solver(args)
        self.project_path, self.project_prefix_path, self.dataset_path, self.train_path, self.test_path, self.ckptdir_path = self.set_paths()
        self.features_dim, self.labels_dim = self.set_dims()

    def set_paths(self):
        project_path = utils.path_exists(self.codebase_root_path)
        project_prefix_path = ""
        dataset_path = utils.path_exists(os.path.join(self.codebase_root_path, "../data", self.dataset_name))
        ckptdir_path = utils.path_exists(os.path.join(self.codebase_root_path, "./bin/ckpt/", self.dataset_name, ""))
        train_path = os.path.join(dataset_path, self.dataset_name + "-train")
        test_path = os.path.join(dataset_path, self.dataset_name + "-test")

        return project_path, project_prefix_path, dataset_path, train_path, test_path, ckptdir_path

    def set_dims(self):
        with open(os.path.join(self.dataset_path, "count.txt"), "r") as f:
            return [int (i) for i in f.read().split("\n") if i != ""]