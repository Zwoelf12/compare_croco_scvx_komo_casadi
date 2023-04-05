import yaml
from optimization import opt_utils as ou
import numpy as np

probs = ["flip", "obstacle_flight", "simple_flight", "recovery_flight", "loop"]
solver_names = ["KOMO","SCVX","CASADI"]

for p in probs:
    i_g = ou.load_object("data/data_quim/initial_guess_" + p)

    init_x = i_g["init_x"].tolist()
    init_u = i_g["init_u"].tolist()[:-1]
    dat = {'states': init_x,'actions': init_u}

    print(p)

    with open("../scripts/data/data_quim/data_quim_yaml/guess_" + p + ".yaml", mode="wt", encoding="utf-8") as file:
        yaml.dump(dat, file, default_flow_style=None, sort_keys=False)
    

    for s_n in solver_names:

        res = ou.load_object("data/data_quim/"+ p + "_4m_" + s_n)

        x = res.data[:,:13].tolist()
        u = res.data[:,13:].tolist()[:-1]
        cost = {"cost": [float((np.array(u) ** 2).sum())]}
        dat = {"result": [{'states': x,
                      'actions': u}]}
        with open("../scripts/data/data_quim/data_quim_yaml/result_" + p + "_" + s_n + ".yaml", mode="wt", encoding="utf-8") as file:
            yaml.dump(cost, file, default_flow_style=None, sort_keys=False)
            yaml.dump(dat, file, default_flow_style=None, sort_keys=False)