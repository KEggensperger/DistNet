import keras.backend as K

from .fcnet import FCNetBase


class FCNetDistribution(FCNetBase):

    def __init__(self, config, n_inputs, n_outputs, expected_num_epochs, verbose=1,
                 early_stopping=False, early_stopping_kwargs=None):
        config = dict(config)
        config["loss_function"] = [self.__loss__, ]
        config["output_activation"] = K.exp
        print(config)
        super().__init__(config, n_inputs=n_inputs, n_outputs=n_outputs,
                         expected_num_epochs=expected_num_epochs, verbose=verbose,
                         early_stopping=early_stopping,
                         early_stopping_kwargs=None, metrics=[self.__loss__])

    @staticmethod
    def get_dist_config_space(
                num_layers_range=[1, 2, 10], seed=None,
                use_dropout=True, use_l2_regularization=True,
                activations=['relu', 'tanh', 'sigmoid'],
                batch_normalization=[True, False],
                optimizers=['SGD', 'RMSprop', 'Adam'],
                learning_rate_schedules=['ExponentialDecay']):
        return FCNetBase.get_config_space(
            loss_functions=["own"],
            output_activations=["own"],
            num_layers_range=num_layers_range, seed=seed,
            use_dropout=use_dropout,
            use_l2_regularization=use_l2_regularization,
            activations=activations,
            batch_normalization=batch_normalization,
            optimizers=optimizers,
            learning_rate_schedules=learning_rate_schedules)

    @staticmethod
    def __loss__(y_true, y_pred):
        raise NotImplementedError()


class ParamFCNetInvGaussFloc(FCNetDistribution):

    def __init__(self, config, n_inputs, expected_num_epochs, verbose=1,
                 early_stopping=False, early_stopping_kwargs=None):
        super().__init__(config=config, n_inputs=n_inputs, n_outputs=2,
                         expected_num_epochs=expected_num_epochs, verbose=verbose,
                         early_stopping=early_stopping,
                         early_stopping_kwargs=early_stopping_kwargs)

    @staticmethod
    def __loss__(y_true, y_pred):
        half = K.constant(0.5, dtype=K.floatx())
        two = K.constant(2.0, dtype=K.floatx())
        threehalf = K.constant(3.0/2.0, dtype=K.floatx())

        mu = y_pred[:, 0]
        mu = K.reshape(mu, [-1, 1])

        scale = y_pred[:, 1]
        scale = K.reshape(scale, [-1, 1])

        tmp_true = K.zeros_like(y_true)
        y_true = tmp_true + y_true

        # Compute logged lh (removed constants)
        help1 = half * K.log(scale)
        help2 = threehalf * K.log(y_true)

        tmp = (y_true/scale)

        help3 = K.pow(tmp - mu, two)
        lower = two * tmp * K.pow(mu, two)
        help3 = help3 / lower

        # add terms (not multiplying them)
        lh = help1 - help2 - help3

        return -lh


class ParamFCNetLognormalFloc(FCNetDistribution):

    def __init__(self, config, n_inputs, expected_num_epochs, verbose=1,
                 early_stopping=False, early_stopping_kwargs=None):
        super().__init__(config=config, n_inputs=n_inputs, n_outputs=2,
                         expected_num_epochs=expected_num_epochs, verbose=verbose,
                         early_stopping=early_stopping,
                         early_stopping_kwargs=early_stopping_kwargs)

    @staticmethod
    def __loss__(y_true, y_pred):
        half = K.constant(0.5, dtype=K.floatx())
        two = K.constant(2, dtype=K.floatx())

        s = y_pred[:, 0]
        s = K.reshape(s, [-1, 1])

        scale = y_pred[:, 1]
        scale = K.reshape(scale, [-1, 1])
        log_scale = K.log(scale)
        log_true = K.log(y_true)

        # Compute logged lh (removed constants)
        help1 = log_true - log_scale
        help1 = half * K.pow(help1 / s, two)

        # add terms (not multiplying them)
        lh = - K.log(s) - log_true - help1

        return -lh


class ParamFCNetExponFloc(FCNetDistribution):

    def __init__(self, config, n_inputs, expected_num_epochs, verbose=1,
                 early_stopping=False, early_stopping_kwargs=None):
        super().__init__(config=config, n_inputs=n_inputs, n_outputs=1,
                         expected_num_epochs=expected_num_epochs, verbose=verbose,
                         early_stopping=early_stopping,
                         early_stopping_kwargs=early_stopping_kwargs)

    @staticmethod
    def __loss__(y_true, y_pred):
        scale = y_pred[:, 0]
        scale = K.reshape(scale, [-1, 1])
        scale = 1 / scale

        log_scale = K.log(scale)

        # Compute logged lh (removed constants)
        lh = log_scale - y_true * scale

        return -lh