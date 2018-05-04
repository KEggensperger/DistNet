import argparse
from functools import partial
import os
import pickle
import sys

import numpy as np
import scipy
import scipy.stats as scst
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.append("../")
from helper import load_data, preprocess, data_source_release
from src import distnet
from src import fcnet

def main():
    sc_dict = data_source_release.get_sc_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", required=True,
                        choices=("invgauss_distfit.floc",
                                 "invgauss_lbfgs.floc",
                                 "lognormal_distfit.floc",
                                 "lognormal_lbfgs.floc",
                                 "expon_distfit.floc",
                                 "expon_lbfgs.floc",
                                 "invgauss_nn.floc",
                                 "lognormal_nn.floc",
                                 "expon_nn.floc",
                                 "nn.mean", "nn.floc", "rf.mean"))
    parser.add_argument("--scenario", dest="scenario", required=True,
                        choices=sc_dict.keys())
    parser.add_argument("--num_train_samples", dest="num_train_samples",
                        default=100, type=int)
    parser.add_argument("--fold", dest="fold", required=True, type=int)
    parser.add_argument("--save", dest="save", required=True)
    parser.add_argument("--seed", dest="seed", required=False, default=1,
                        type=int)
    parser.add_argument("--wclim", dest="wclim", required=False, default=60*59,
                        type=int)
    parser.add_argument("--neurons", dest="neurons", required=False,
                        default=16, type=int)
    parser.add_argument("--layer", dest="layer", required=False, default=2,
                        type=int)
    parser.add_argument("--epochs", dest="epochs", required=False, default=1000,
                        type=int)
    args = parser.parse_args()
    # 1) Assertions
    assert 0 <= args.fold <= 9

    # 2) Sort out model and task
    model_name, task = args.model.split(".")
    save_path = os.path.join(args.save, "%s.%s.%s.%d.%d.pkl" % (args.scenario,
                                                                task, model_name,
                                                                args.fold,
                                                                args.seed))
    if args.num_train_samples != 100:
        save_path += "_%d" % args.num_train_samples
    add_info = {"task": task, "scenario": args.scenario,
                "fold": args.fold, "model": model_name, "loaded": False,
                "num_train_samples": args.num_train_samples,
                "seed": args.seed}

    # Maybe load configuration from file
    config_fp = os.path.join(args.save, "config",
                             "%s.%s.%s.cfg" % (args.scenario, model_name, task))
    print("CHECKING %s" % config_fp)
    if os.path.exists(config_fp):
        with open(config_fp, "rb") as fh:
            config = pickle.load(fh)
            add_info["loaded"] = True
        print("Successfully loaded config %s" % config_fp)
    else:
        config = None

    # 3) Load data
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()

    runtimes, features, sat_ls = load_data.\
        get_data(scenario=args.scenario, data_dir=data_dir,
                 sc_dict=sc_dict, retrieve=sc_dict[args.scenario]['use'])

    features = np.array(features)
    runtimes = np.array(runtimes)
    
    # Get CV splits
    print(runtimes.shape, features.shape)
    idx = list(range(runtimes.shape[0]))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    cntr = -1
    for train, valid in kf.split(idx):
        # Reset seed for every instance
        np.random.seed(2)
        cntr += 1
        if cntr != args.fold:
            continue

        X_train = features[train, :]
        X_valid = features[valid, :]

        y_train = runtimes[train]
        y_valid = runtimes[valid]

        X_train, X_valid = preprocess.preprocess_features(X_train, X_valid,
                                                          scal="meanstd")

        print("Evaluating %s, %s, %s on %s, %s" % (args.scenario, model_name,
                                                   task, str(X_train.shape),
                                                   str(y_train.shape)))

        X_trn_flat = np.concatenate(
            [[x for i in range(100)] for x in X_train])
        X_vld_flat = np.concatenate(
            [[x for i in range(100)] for x in X_valid])
        y_trn_flat = y_train.flatten().reshape([-1, 1])
        y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Unfold data
        subset_idx = list(range(100))
        if args.num_train_samples != 100:
            print("Cut data down to %d samples with seed %d" %
                  (args.num_train_samples, args.seed))
            rs = np.random.RandomState(args.seed)
            rs.shuffle(subset_idx)
            subset_idx = subset_idx[:args.num_train_samples]
            print(subset_idx)

            # Only shorten data used for training
            X_trn_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_train])
            y_train = y_train[:, subset_idx]
            y_trn_flat = y_train.flatten().reshape([-1, 1])

            X_vld_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_valid])
            y_valid = y_valid[:, subset_idx]
            y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Min/Max Scale runnningtimes
        y_max_ = np.max(y_trn_flat)
        y_min_ = 0

        y_trn_flat = (y_trn_flat - y_min_) / y_max_
        y_vld_flat = (y_vld_flat - y_min_) / y_max_

        y_train = (y_train - y_min_) / y_max_
        y_valid = (y_valid - y_min_) / y_max_

        print("X_train:", X_train.shape)
        print("X_valid:", X_valid.shape)
        print("X_train_flat:", X_trn_flat.shape)
        print("X_valid_flat:", X_vld_flat.shape)

        print()

        print("y_train:", y_train.shape)
        print("y_valid:", y_valid.shape)
        print("y_trn_flat:", y_trn_flat.shape)
        print("y_vld_flat:", y_vld_flat.shape)

        # Model that just predicts mean
        if task == "mean":
            # Construct training data
            import keras.backend as K
            import tensorflow as T
            cfg = T.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
            session = T.Session(config=cfg)
            K.set_session(session)
            y_train_mean = np.mean(y_train, axis=1, keepdims=True)
            y_valid_mean = np.mean(y_valid, axis=1, keepdims=True)
            tra_pred, val_pred = (None, None)
            if model_name == "nn":
                if config is None:
                    config = fcnet.\
                        ParamFCNetRegression.get_config_space().\
                        get_default_configuration()
                print(config)

                model = fcnet.ParamFCNetRegression(
                        config=config, n_feat=X_train.shape[1],
                        n_outputs=1, expected_num_epochs=1000)
                print(model.model.optimizer.__dict__)
                model.train(X_train=X_train, y_train=y_train_mean,
                            X_valid=X_valid, y_valid=y_valid_mean,
                            n_epochs=1000, time_limit_s=60*59)
            elif model_name == "rf":
                model = RandomForestRegressor()
                model.fit(X_train, y_train_mean)

            tra_pred = model.predict(X_train)
            val_pred = model.predict(X_valid)
            dump_res(tra_pred=tra_pred, val_pred=val_pred, save_path=save_path,
                     add_info=add_info)
        elif task == "floc":
            if "nn" in model_name:
                import keras.backend as K
                import tensorflow as T
                cfg = T.ConfigProto(intra_op_parallelism_threads=1,
                                    inter_op_parallelism_threads=1)
                session = T.Session(config=cfg)
                K.set_session(session)
                if model_name == "invgauss_nn":
                    model_hdl = distnet.ParamFCNetInvGaussFloc
                elif model_name == "lognormal_nn":
                    model_hdl = distnet.ParamFCNetLognormalFloc
                elif model_name == "expon_nn":
                    model_hdl = distnet.ParamFCNetExponFloc
                else:
                    raise ValueError(model_name, "is unknown")

                if config is None:
                    config = model_hdl.get_dist_config_space(
                            num_layers_range=[1, 2, 5],
                            #loss_functions=[model_hdl.__loss__],
                            activations=['tanh'],
                            #output_activations=[K.exp],
                            optimizers=['SGD'],
                            learning_rate_schedules=['ExponentialDecay']).\
                        get_default_configuration()
                config["num_layers"] = args.layer
                config["average_units_per_layer"] = args.neurons

                print(config)
                retry = 5
                while retry > 0:
                    model = model_hdl(config=config, n_inputs=X_train.shape[1],
                                      expected_num_epochs=args.epochs,
                                      early_stopping=False, verbose=0)

                    print("Start training")
                    model.train(X_train=X_trn_flat, y_train=y_trn_flat,
                                X_valid=X_trn_flat, y_valid=y_trn_flat,
                                n_epochs=args.epochs, time_limit_s=args.wclim)
                    print("Finished")
                    tra_pred = model.predict(X_train)
                    val_pred = model.predict(X_valid)
                    retry -= 1

                    if np.isfinite(tra_pred).all():
                        retry = -1

            elif "distfit" in model_name:
                if model_name == "lognormal_distfit":
                    dist_hdl = scst.lognorm
                elif model_name == "invgauss_distfit":
                    dist_hdl = scst.invgauss
                elif model_name == "expon_distfit":
                    dist_hdl = scst.expon
                else:
                    raise ValueError("Don't know %s" % model_name)

                tra_pred = list()
                for inst in y_train:
                    assert len(inst) == len(subset_idx)
                    tp = dist_hdl.fit(inst, floc=0)
                    tra_pred.append(tp)

                val_pred = list()
                for inst in y_valid:
                    assert len(inst) == len(subset_idx)
                    tp = dist_hdl.fit(inst, floc=0)
                    val_pred.append(tp)

            elif "lbfgs" in model_name:
                if model_name == "lognormal_lbfgs":
                    dist_hdl = scst.lognorm
                    bounds = [[10e-14, "+inf"], [0, 0], [10e-14, "+inf"]]
                elif model_name == "invgauss_lbfgs":
                    dist_hdl = scst.invgauss
                    bounds = [[10e-14, "+inf"], [0, 0], [10e-14, "+inf"]]
                elif model_name == "expon_lbfgs":
                    dist_hdl = scst.expon
                    bounds = [[0, 0], [10e-14, "+inf"]]
                else:
                    raise ValueError("Don't know %s" % model_name)

                tra_pred = list()
                for inst in y_train:
                    assert len(inst) == len(subset_idx)
                    optimizer = partial(scipy.optimize.fmin_l_bfgs_b,
                                        approx_grad=True,
                                        bounds=bounds)
                    tp = dist_hdl.fit(inst, optimizer=optimizer)[0]
                    tra_pred.append(tp)

                val_pred = list()
                for inst in y_valid:
                    assert len(inst) == len(subset_idx)
                    optimizer = partial(scipy.optimize.fmin_l_bfgs_b,
                                        approx_grad=True,
                                        bounds=bounds)
                    tp = dist_hdl.fit(inst, optimizer=optimizer)[0]
                    val_pred.append(tp)
            else:
                raise NotImplementedError("Don't know %s -> %s, %s" %
                                          (args.model, task, model_name))

            # Cut out zeros
            tra_pred = np.array(tra_pred)
            val_pred = np.array(val_pred)
            if "distfit" in model_name or "lbfgs" in model_name:
                if "lognormal" in model_name or "invgauss" in model_name:
                    assert tra_pred.shape[1] == 3
                    assert val_pred.shape[1] == 3
                    tra_pred = np.hstack([tra_pred[:, 0].reshape([-1, 1]),
                                          tra_pred[:, 2].reshape([-1, 1])])
                    val_pred = np.hstack([val_pred[:, 0].reshape([-1, 1]),
                                          val_pred[:, 2].reshape([-1, 1])])
                elif "expon" in model_name:
                    assert tra_pred.shape[1] == 2
                    assert val_pred.shape[1] == 2
                    tra_pred = tra_pred[:, 1].reshape([-1, 1])
                    val_pred = val_pred[:, 1].reshape([-1, 1])
                else:
                    print("We have a problem")

            dump_res(tra_pred=tra_pred, val_pred=val_pred,
                     save_path=save_path, add_info=add_info)

            # If model was one of scipy.distfits, then also train rf
            if "distfit" in model_name or "lbfgs" in model_name:
                model = RandomForestRegressor(random_state=0)
                model.fit(X_train, tra_pred)
                rf_tra_pred = model.predict(X_train)
                rf_val_pred = model.predict(X_valid)
                save_path_spl = save_path.split('.')
                save_path_new = '.'.join(save_path_spl[:-1]) + ".rf." + \
                                save_path.split('.')[-1]

                add_info["model"] = model_name + ".rf"
                dump_res(save_path=save_path_new, tra_pred=rf_tra_pred,
                         val_pred=rf_val_pred, add_info=add_info)

                model = MultiOutputRegressor(RandomForestRegressor(random_state=0))
                model.fit(X_train, tra_pred)
                rf_tra_pred = model.predict(X_train)
                rf_val_pred = model.predict(X_valid)
                save_path_new = '.'.join(save_path_spl[:-1]) + ".rf2." + \
                                save_path.split('.')[-1]

                add_info["model"] = model_name + ".rf2"
                dump_res(save_path=save_path_new, tra_pred=rf_tra_pred,
                         val_pred=rf_val_pred, add_info=add_info)

        else:
            raise NotImplementedError("Don't know %s -> %s, %s" %
                                      (args.model, task, model_name))


def dump_res(save_path, tra_pred, val_pred, add_info):
    with open(save_path, "wb") as fh:
        pickle.dump([tra_pred, val_pred, add_info], fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped to %s" % save_path)

if __name__ == "__main__":
    main()
