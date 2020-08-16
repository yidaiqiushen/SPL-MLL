
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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

    def get_train_validation(self):
        if self.train == None:
            X = self.get_data(self.config.train_path + "-features.mat", True)
            # Y = self.get_data(self.config.train_path + "-labels.mat").transpose()
            Y1 = self.get_data(self.config.train_path + "-labels.mat")
            Y = Y1.transpose()
            X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.2, random_state=0)
            # length = X.shape[0]
            # X, Y = X[0 : int(5/6 * length) , :], Y[0 : int(5/6 * length), :]
            self.train = X_train, Y_train
            self.validation = X_val, Y_val
        else :
            X_train, Y_train = self.train
            X_val, Y_val = self.validation
        return X_train, Y_train, X_val, Y_val

    # def get_validation(self):
    #     if self.validation == None:
    #         X = self.get_data(self.config.train_path + "-features.mat")
    #         # Y = self.get_data(self.config.train_path + "-labels.mat").transpose()
    #         Y1 = self.get_data(self.config.train_path + "-labels.mat")
    #         Y = Y1.transpose()
    #         length = X.shape[0]
    #         X, Y = X[0 : int(1/6 * length) , :], Y[0 : int(1/6 * length), :]
    #         self.validation = X, Y
    #     else :
    #         X, Y = self.validation
    #     return X, Y

    def get_test(self):
        if self.test == None:
            X = self.get_data(self.config.test_path + "-features.mat")
            # Y = self.get_data(self.config.test_path + "-labels.mat").transpose()
            Y1 = self.get_data(self.config.test_path + "-labels.mat")
            Y = Y1.transpose()
            # length = X.shape[0]
            # X, Y = X[0: int(0.8 * length), :], Y[0: int(0.8 * length), :]
            self.test = X, Y
        else:
            X, Y = self.test
        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        # func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        # X, Y = func()

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
