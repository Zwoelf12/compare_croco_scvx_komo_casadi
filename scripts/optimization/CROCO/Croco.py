import subprocess as sp
import os
import yaml
import numpy as np
from optimization import opt_utils as ou

def solve(prob_name, initial_x, initial_u, tdil, max_iter, weight_goal):

    print("weight_goal: ", weight_goal)
    print("max_iter: ", max_iter)

    # write initial guess to file
    init_x = initial_x.tolist()
    init_u = initial_u.tolist()[:-1]
    dat = {'states': init_x,'actions': init_u}

    with open("temp/guess.yaml", mode="wt", encoding="utf-8") as file:
        yaml.dump(dat, file, default_flow_style=None, sort_keys=False)

    # run croco solver
    os.chdir("../../kinodynamic-motion-planning-benchmark/build/")


    if "recovery_flight" in prob_name:
        prob_name = "recovery_flight"
    elif "obstacle_flight" in prob_name:
        prob_name = "obstacle_flight"
    elif "flip" in prob_name:
        prob_name = "flip"
    elif "loop" in prob_name:
        prob_name = "loop"

    env_file = "../benchmark/quadrotor_0/" + prob_name+ ".yaml"
    print(env_file)
    init_guess_file = "../../compare_croco_scvx_komo_casadi/scripts/temp/guess.yaml"  #"../../compare_croco_scvx_komo_casadi/scripts/data/data_quim/data_quim_yaml/guess_low_noise/initial_guess_recovery_flight_1.yaml"#"../data_welf_yaml/guess_recovery_flight.yaml"
    out_file = "../../compare_croco_scvx_komo_casadi/scripts/temp/my_trajectory.yaml" 
    #high noise: 200: 2,5,9,10; 400: 1,3,8; 600: ; 700: 7 ; 800: 4 ; X: 6
    #low noise: X: 10,1

    bashCommand = "./croco_main --welf_format 1 --env_file " + env_file + " --init_guess " + init_guess_file + " --out " + out_file + " --max_iter " + str(max_iter) + " --weight_goal " + str(weight_goal)
    print(bashCommand)
    process = sp.Popen(bashCommand.split(), stdout=sp.PIPE)
    output, error = process.communicate()

    os.chdir("../../compare_croco_scvx_komo_casadi/scripts/")

    # get results from yaml file
    with open("temp/my_trajectory.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    states = np.array(data_loaded["states"])#["states"])
    actions = np.vstack([np.array(data_loaded["actions"]), np.array([np.nan,np.nan,np.nan,np.nan])])   
    time_solver = float(data_loaded["info"].split("=")[2])/1000
    num_iter = int(data_loaded["info"].split(";")[0].split("=")[1])
    print(num_iter)

    solution = ou.OptSolution(states, actions, time = time_solver, num_iter= num_iter, tdil = tdil)

    return solution
