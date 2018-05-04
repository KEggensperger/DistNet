import keras
import ConfigSpace as CS


class Optimizer(object):
    def __init__(self, config):
        """ reads config to create the optimizer for keras"""

    def __call__(self, **kwargs):
        """returns an instance of the optimizer itself"""
        raise NotImplementedError


class SGD(Optimizer):
    """ TODO: should Nesterov always be true? """
    def __init__(self, config):
        self.momentum = config['SGD_momentum']
        self.clipvalue = 1e-2

    def __call__(self):
        return(keras.optimizers.SGD(lr=0, momentum=self.momentum,
                                    nesterov=True,
                                    clipvalue=self.clipvalue))

    @classmethod
    def get_hyperparameters(cls, #initial_lr_range=[1e-4,1e-1,5e-1,True],
                            final_lr_fraction_range=[1e-4, 1e-2,1, True],
                            #initial_lr_range=[10e-6, 10e-4, 10e-3, True],
                            initial_lr_range=[1e-4,1e-3,5e-1,True],
                            momentum_range=[0, 0.9, 0.99, False]#,
                            #clipvalue_range=[10e-5, 10e-1, 10, True]
                            ):
        """ ranges specify the lower bound, default value,
        upper bound and whether it is varied on a log scale """

        initial_lr = CS.UniformFloatHyperparameter("SGD_initial_lr",
                    lower=initial_lr_range[0],
                    default=initial_lr_range[1],
                    upper=initial_lr_range[2],
                    log=initial_lr_range[3])

        final_lr_fraction = CS.UniformFloatHyperparameter("SGD_final_lr_fraction",
                    lower=final_lr_fraction_range[0],
                    default=final_lr_fraction_range[1],
                    upper=final_lr_fraction_range[2],
                    log=final_lr_fraction_range[3])

        momentum = CS.UniformFloatHyperparameter("SGD_momentum",
                    lower=momentum_range[0],
                    default=momentum_range[1],
                    upper=momentum_range[2],
                    log=momentum_range[3])

        return({cls.__name__: [initial_lr, final_lr_fraction, momentum]})