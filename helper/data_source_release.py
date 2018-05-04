import os

def get_data_dir(): return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def get_sc_dict():
    data_dir = get_data_dir()
    sc_dict = {
     "clasp_factoring":
      {"scen": "clasp-3.0.4-p8_rand_factoring",
      "features": "%s/clasp-3.0.4-p8_rand_factoring/features.txt" % data_dir,
      "domain": "sat",
      "use": ('SAT',),
      "cutoff": 5000},
     "saps-CVVAR":
     {"scen": "CP06_CV-VAR",
      "features": "%s/CP06_CV-VAR/features.txt" % data_dir,
      "domain": "sat",
      "use": ('SAT',),
      "cutoff": 60},
     "spear_qcp":
      {"scen": "spear_qcp-hard",
       "features": "%s/spear_qcp-hard/features.txt" % data_dir,
       "domain": "sat",
       "use": ('SAT',),
       "cutoff": 5000},
     "yalsat_qcp":
      {"scen": "yalsat_qcp-hard",
       "features": "%s/yalsat_qcp-hard/features.txt" % data_dir,
       "domain": "sat",
       "use": ('SAT',),
       "cutoff": 5000},
     "spear_swgcp":
      {"scen": "spear_smallworlds",
       "features": "%s/spear_smallworlds/features.txt" % data_dir,
       "domain": "sat",
       "use": ('SAT',),
       "cutoff": 5000},
     "yalsat_swgcp":
      {"scen": "yalsat_smallworlds",
       "features": "%s/yalsat_smallworlds/features.txt" % data_dir,
       "domain": "sat",
       "use": ('SAT',),
       "cutoff": 5000},
      "lpg-zeno":
       {"scen": "lpg-zenotravel",
        "features": "%s/lpg-zenotravel/features.txt" % data_dir,
        "domain": "planning",
        "use": ('SAT', ),
        "cutoff": 300}
    }
    return sc_dict
