import os

import numpy as np

from . import preprocess


def read_results(data_dir, cutoff=300, runs_per_inst=100, suffix="train"):
    fl_name = "%s/validate-random-%s/validationRunResultLineMatrix-cli-1-" \
              "walltimeworker.csv" % (data_dir, suffix)
    
    if not os.path.exists(fl_name):
        raise ValueError("%s does not exist" % fl_name)
    
    tmp_data = list()
    inst_ls = list()
    sat_ls = list()
    with open(fl_name, 'r') as fl:
        # ignore header
        fl.readline()
        for line in fl:
            line = line.strip().replace('"', '').split(",")
            tmp_data.append(min(cutoff,
                                float(line[3].strip())))
            sat_ls.append(str(line[2]))
            inst_ls.append(line[0])

    # Split per instance
    data = list()
    sat_data = list()
    for inst in range(int(len(tmp_data)/runs_per_inst)):
        data.append(tmp_data[inst*runs_per_inst:(inst+1)*runs_per_inst])
        sat_data.append(sat_ls[inst*runs_per_inst:(inst+1)*runs_per_inst])
    data = np.array(data)
    return data, inst_ls, sat_data


def load_features(fl_name):
    feat_dict = dict()
    with open(fl_name) as fh:
        fh.readline()
        for line in fh:
            line = line.strip().split(",")
            key = line[0]
            val = [float(i) for i in line[1:]]
            feat_dict[key] = val
    return feat_dict


def get_data(scenario, data_dir, sc_dict, retrieve=["SAT", "UNSAT"]):
    data_dir = data_dir + "/" + sc_dict[scenario]["scen"] + "/"
    runtimes, inst_ls, sat_ls = \
        read_results(data_dir=data_dir, cutoff=sc_dict[scenario]["cutoff"],
                               runs_per_inst=100, suffix="train")
    print("Train data loaded")
    try:
        test_runtimes, test_inst_ls, test_sat_ls = \
            read_results(data_dir=data_dir, cutoff=sc_dict[scenario]["cutoff"],
                                   runs_per_inst=100, suffix="test")
        runtimes = np.vstack([runtimes, test_runtimes])
        inst_ls.extend(test_inst_ls)
        sat_ls.extend(test_sat_ls)
        print("Test data loaded")
    except ValueError:
        print("Could not find test data")

    try:
        feat_dict = load_features(sc_dict[scenario]["features"])
    except TypeError:
        print("Features file %s does not exist" % sc_dict[scenario]["features"])
        feat_dict = dict((i, np.random.random_sample(2)) for i in inst_ls[::100])

    print(sc_dict[scenario]["features"])
    features = list()
    for i in inst_ls[::100]:
        features.append(feat_dict[i])

    features = np.array(features)
    runtimes = np.array(runtimes)
    print(features.shape, runtimes.shape)

    runtimes, features, sat_ls = preprocess.remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls,
            status="CRASHED")
    runtimes, features, sat_ls = preprocess.remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls,
            status="TIMEOUT")
    runtimes, features, sat_ls = preprocess.remove_timeouts(
            runningtimes=runtimes, features=features,
            cutoff=sc_dict[scenario]["cutoff"], sat_ls=sat_ls)
    runtimes, features, sat_ls = preprocess.remove_constant_instances(
            runningtimes=runtimes, features=features, sat_ls=sat_ls)

    if "SAT" not in retrieve:
        # remove SAT features
        runtimes, features, sat_ls = preprocess.remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls,
            status="SAT")
    if "UNSAT" not in retrieve:
        # remove SAT features
        runtimes, features, sat_ls = preprocess.remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls,
            status="UNSAT")


    features = preprocess.feature_imputation(features, impute_val=-512,
                                             impute_with="median")
    return runtimes, features, sat_ls