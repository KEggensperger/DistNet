import numpy as np


class LearningRateSchedule(object):
    def __init__(self, initial_lr, final_lr_fraction, expected_num_epochs):
        self.initial_lr = initial_lr
        self.final_lr = final_lr_fraction*initial_lr
        self.expected_num_epochs = expected_num_epochs

    def __call__(self, epoch):
        raise NotImplementedError('You need to overwrite the __call__ method of this class')


class ExponentialDecay(LearningRateSchedule):

    def __init__(self, initial_lr, final_lr_fraction, expected_num_epochs, **kwargs):
        super().__init__(initial_lr, final_lr_fraction, expected_num_epochs)
        self.gamma = np.log(1/final_lr_fraction)/expected_num_epochs

    def __call__(self, epoch):
        return(self.initial_lr * np.exp(-self.gamma*epoch))

    @classmethod
    def get_hyperparameters( cls, **kwargs):
        return({})