import numpy as np
import keras


class EarlyStopping(keras.callbacks.EarlyStopping):
    """ adapts the Keras EarlyStopping class to work across continuous training """
    def __init__ (self, *args, **kwargs):
        super(EarlyStopping, self).__init__(*args, **kwargs)
        # initialize best here, b/c super does that in 'on_training_begin'
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    def on_train_begin(self, logs=None):
        """ overwrite this with nothing to keep the 'state' """
        pass


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std
