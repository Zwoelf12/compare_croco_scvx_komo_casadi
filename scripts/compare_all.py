from optimization.parallelization import Multiprocess
from optimization_handler import run_optimization
from optimization.problem_setups import Prob_setup
from optimization.algorithm_parameters import KOMO_parameter,SCVX_parameter,CASADI_parameter,CROCO_parameter
from optimization import opt_utils as ou
from visualization.report_visualization import report_all

solver_names = ["KOMO","SCVX","CASADI","CROCO"]
run = False
vis = True

# choose which problems should be solved
nr_of_runs = 30
probs = [2,3,4,5]

# build list of all parameters for run optimization function
all_alg_pars = []
all_prob_names = []
all_prob_setups = []
all_solver_names = []
for i in range(nr_of_runs):
    for p in probs:

        prob_setup = Prob_setup(p)
        prob_name = prob_setup.name

        all_alg_pars.append({"KOMO":KOMO_parameter(prob_name),
                            "SCVX":SCVX_parameter(prob_name),
                            "CASADI":CASADI_parameter(prob_name),
                            "CROCO":CROCO_parameter(prob_name)})
        
        all_prob_names.append("run_" + str(i) + "_" + prob_name)
        all_prob_setups.append(prob_setup)
        all_solver_names.append(solver_names)

# put all arg in one list (order is important)
args = [all_prob_names, all_prob_setups, all_solver_names, all_alg_pars]    

if run:
    process_handler = Multiprocess(run_optimization, args)
    results = process_handler.run_processes()

if vis:
    report_all(all_prob_names, solver_names)
