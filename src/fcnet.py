import logging
import time

import keras
import ConfigSpace as CS
import numpy as np

from . import util
from . import optimizers as pn_optimizers
from . import lr_schedules as lr_schedules


class FCNetBase(object):

    def __init__(self, config, n_inputs, n_outputs, expected_num_epochs, metrics,
                 verbose=1, early_stopping=False, early_stopping_kwargs=None):
        self.verbose = verbose
        self.config = config
        self.num_epochs = 0
        self.expected_num_epochs = expected_num_epochs
        self.learning_rate_callback = None
        self.es_kwargs = None
        self.metrics = metrics
        self.history = {}
        
        self.logger = logging.getLogger("ParamFCNetBase")

        if type(self.config) is not dict:
            self.conf_dict = self.config.get_dictionary()
        else:
            self.conf_dict = self.config

        opt_name = self.conf_dict['optimizer']
        initial_lr = self.conf_dict[opt_name+'_initial_lr']
        final_lr_fraction = self.conf_dict[opt_name+'_final_lr_fraction']
      
        self.learning_rate_callback = getattr(lr_schedules, self.conf_dict['learning_rate_schedule'])(expected_num_epochs=expected_num_epochs, initial_lr=initial_lr, final_lr_fraction=final_lr_fraction, **config)
        self.callbacks = [keras.callbacks.LearningRateScheduler(self.learning_rate_callback)]

        # Test for NaNs after each batch
        self.callbacks.append(keras.callbacks.TerminateOnNaN())

        if early_stopping:
            self.es_kwargs = dict(monitor='val_loss', patience=4, min_delta=0,
                                  mode='auto')
            if early_stopping_kwargs is not None:
                self.es_kwargs.update(early_stopping_kwargs)
            self.early_stopping_callback = util.EarlyStopping(**self.es_kwargs)
            self.callbacks.append(self.early_stopping_callback)

        self.model = self._build_net(n_inputs, n_outputs)

    def _build_net(self, n_feat, n_outputs):
        model = keras.models.Sequential()

        num_units = np.linspace(self.config['shape_parameter_1'],
                                1-self.config['shape_parameter_1'],
                                self.config['num_layers'])
        num_units *= self.config['average_units_per_layer']/num_units.mean()
        num_units = np.rint(num_units).astype(np.int)

        num_inputs = [n_feat] + num_units[:-1].tolist()

        use_dropout = 'dropout_0' in self.conf_dict.keys()
        use_l2_reg  = 'l2_reg_0'  in self.conf_dict.keys()

        if use_dropout:
            dropout_rates = np.linspace(self.config['dropout_0'],
                                        self.config['dropout_1'],
                                        self.config['num_layers'])

        if use_l2_reg:
            l2_regs = np.linspace(self.config['l2_reg_0'],
                                  self.config['l2_reg_1'],
                                  self.config['num_layers'])
        
        for i in range(self.config['num_layers']):

            if use_l2_reg:
                reg = keras.regularizers.l2(l2_regs[i])
            else: reg = None

            model.add(keras.layers.Dense(units=num_units[i],
                                         input_dim=num_inputs[i],
                                         kernel_regularizer=reg))

            if self.config['batch_normalization']:
                model.add(keras.layers.BatchNormalization())
        
            model.add(keras.layers.Activation(self.config['activation']))
        
            if use_dropout:
                model.add(keras.layers.Dropout(rate=dropout_rates[i])) 

        model.add(keras.layers.Dense(units=n_outputs,
                                     activation=self.config['output_activation']))

        opt = getattr(pn_optimizers, self.config['optimizer'])(self.config)
        
        model.compile(loss=self.config['loss_function'],
                      optimizer=opt(),
                      metrics=self.metrics)
        if self.verbose > 0:
            model.summary()
            
        return model

    def train(self, X_train, y_train, X_valid, y_valid,
              n_epochs, shuffle=True, callbacks=[], time_limit_s=None):

        if self.config['loss_function'] == "categorical_crossentropy"\
                or self.config['loss_function'] == "kullback_leibler_divergence":
            y_train = keras.utils.to_categorical(y_train[:, None],
                                                 num_classes=self.n_classes)
            y_valid = keras.utils.to_categorical(y_valid[:, None],
                                                 num_classes=self.n_classes)

        if getattr(self.model, 'stop_training', False):
            self.logger.debug('Training has been stopped earlier!')
            return None

        callbacks = self.callbacks + callbacks

        # Training

        if time_limit_s is None:
            hist = self.model.fit(X_train, y_train,
                                  epochs=self.num_epochs+n_epochs,
                                  verbose=self.verbose,
                                  batch_size=self.config['batch_size'],
                                  validation_data=(X_valid, y_valid),
                                  callbacks=callbacks,
                                  shuffle=shuffle,
                                  initial_epoch=self.num_epochs)

            self.num_epochs += len(hist.history.get("loss", []))
        
            for k, v in hist.history.items():
                if k in self.history:
                    self.history[k].extend(v)
                else:
                    self.history[k] = v

        else:
            assert time_limit_s > 0, "The time limit has to be positive"
            duration_last_epoch = 0
            final_epoch = self.num_epochs + n_epochs
            used_budget = 0
            start = time.time()

            while (self.num_epochs < final_epoch) and \
                (used_budget + 1.1*duration_last_epoch < time_limit_s):
                hist = self.model.fit(X_train, y_train,
                                      epochs=self.num_epochs+1,
                                      verbose=self.verbose,
                                      batch_size=self.config['batch_size'],
                                      validation_data=(X_valid, y_valid),
                                      callbacks=callbacks,
                                      shuffle=shuffle,
                                      initial_epoch=self.num_epochs)

                self.num_epochs += len(hist.history.get("loss", []))
        
                for k, v in hist.history.items():
                    if k in self.history:
                        self.history[k].extend(v)
                    else:
                        self.history[k] = v

                duration_last_epoch = (time.time()-start) - used_budget
                used_budget += duration_last_epoch 

        return self.history

    def predict(self, X_test):
        return self.model.predict(X_test)

    @staticmethod
    def get_config_space(\
                loss_functions,
                output_activations,
                num_layers_range=[1,2,10], seed=None,
                use_dropout=True, use_l2_regularization=True,
                activations=['relu', 'tanh', 'sigmoid'],
                batch_normalization=[True, False],
                optimizers=['SGD', 'RMSprop', 'Adam'],
                learning_rate_schedules=['ExponentialDecay']):
        cs = CS.ConfigurationSpace(seed)
        
        HPs = [\
            CS.CategoricalHyperparameter('optimizer', optimizers),
            CS.CategoricalHyperparameter("learning_rate_schedule",
                                         learning_rate_schedules),
            CS.UniformIntegerHyperparameter("num_layers",
                                            lower=num_layers_range[0],
                                            default=num_layers_range[1],
                                            upper=num_layers_range[2]),
            CS.UniformIntegerHyperparameter("average_units_per_layer",
                                            lower=16, upper=256, default=16,
                                            log=True),
            CS.UniformFloatHyperparameter('shape_parameter_1', lower=0.01,
                                          upper=0.99, default=0.5),
            CS.UniformIntegerHyperparameter("batch_size", lower=8, upper=256,
                                            default=16, log=True),
            CS.CategoricalHyperparameter("activation", activations),
            CS.CategoricalHyperparameter("output_activation",
                                         output_activations),
            CS.CategoricalHyperparameter("loss_function", loss_functions),
            CS.CategoricalHyperparameter("batch_normalization",
                                         batch_normalization),
        ]

        if use_dropout:
            HPs.append(CS.UniformFloatHyperparameter('dropout_0', lower=0,
                                                     upper=0.5, default=0))
            HPs.append(CS.UniformFloatHyperparameter('dropout_1', lower=0,
                                                     upper=0.5, default=0))

        if use_l2_regularization:
            HPs.append(CS.UniformFloatHyperparameter('l2_reg_0', lower=1e-6,
                                                     upper=1e-2, default=1e-4,
                                                     log=True))
            HPs.append(CS.UniformFloatHyperparameter('l2_reg_1', lower=1e-6,
                                                     upper=1e-2, default=1e-4,
                                                     log=True))

        cs.add_hyperparameters(HPs)

        # get optimizers HPs
        opt_cs_dict = {}
        for opt in optimizers:
            optimizer = getattr(pn_optimizers, opt)
            opt_cs_dict.update(optimizer.get_hyperparameters())

        for k,v in opt_cs_dict.items():
            for hp in v:
                cs.add_hyperparameter(hp)
                cond = CS.EqualsCondition(hp, HPs[0], k)
                cs.add_condition(cond)

        #get learning rate HPs
        lr_cs_dict = {}
        for l in learning_rate_schedules:
            schedule = getattr(lr_schedules, l)
            lr_cs_dict.update(schedule.get_hyperparameters())

        for k,v in lr_cs_dict.items():
            for hp in v:
                cs.add_hyperparameter(hp)
                cond = CS.EqualsCondition(hp, HPs[1], k)
                cs.add_condition(cond)
        return cs