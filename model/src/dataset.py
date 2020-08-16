
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle


class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.train, self.test, self.validation = None, None, None
        self.path = self.config.dataset_path

    def get_data(self, path, noise=False):
        data_mat = sio.loadmat(path)
        if('data' in data_mat.keys()):
            data = data_mat['data']
        else:
            data = data_mat['label']
        if noise == True :
            data = data + np.random.normal(0, 0.001, data.shape)
        return data

    def get_train(self):
        if self.train == None:
            X = self.get_data(self.config.train_path + "-features.mat", False)
            Y1 = self.get_data(self.config.train_path + "-labels.mat")
            Y = Y1.transpose()
            self.train = X, Y
        else :
            X, Y = self.train
        return X, Y

    def get_validation(self):
        if self.validation == None:
            X = self.get_data(self.config.train_path + "-features.mat", False)
            Y1 = self.get_data(self.config.train_path + "-labels.mat")
            Y = Y1.transpose()
            self.validation = X, Y
        else :
            X, Y = self.validation
        return X, Y

    def get_test(self):
        if self.test == None:
            X = self.get_data(self.config.test_path + "-features.mat")
            Y1 = self.get_data(self.config.test_path + "-labels.mat")
            Y = Y1.transpose()
            self.test = X, Y
        else:
            X, Y = self.test
        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        start = 0
        batch_size = self.config.batch_size
        total_examples = len(X)
        total_batch = int(total_examples/ batch_size) # fix the last batch
        while start < total_batch:
            end = start + batch_size
            x = X[start : end, :]
            y = Y[start : end, :]
            start += 1
            yield (x, y, int(total_batch))
