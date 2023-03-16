#all_parameter_search = []
for i in range(nr_of_runs):
    for p in probs:

        prob_setup = Prob_setup(p)
        prob_name = prob_setup.name

        all_alg_pars.append({"KOMO":KOMO_parameter(prob_name),
                            "SCVX":SCVX_parameter(prob_name),
                            "CASADI":CASADI_parameter(prob_name)})
        
        all_prob_names.append("run_" + str(i) + "_" + prob_name)
        all_prob_setups.append(prob_setup)
        all_solver_names.append(solver_names)
        #all_parameter_search.append(False)
