from optimization.check.check_solution import check_solution
from optimization import opt_utils as ou
from optimization.opt_problem import OptProblem
from optimization.problem_setups import Prob_setup
from physics.multirotor_models import multirotor_full_model_komo_scp
from physics.collision.collisionHandler import collisionHandler
from visualization.report_visualization import report_compare
import yaml
import numpy as np

prob = 3
prob_setup = Prob_setup(prob)
prob_name = prob_setup.name
print("prob name: ", prob_setup.name)

# define optimization problem
optProb = OptProblem()
optProb.prob_name = prob_name

# define robot model
nr_motors = 4
arm_length = 0.046

# define start and end point
optProb.x0 = prob_setup.x0
optProb.xf = prob_setup.xf
optProb.intermediate_states = prob_setup.intermediate_states

# define maximum and minimum flight time and step number for intermediate state
optProb.tf_min = prob_setup.tf_min
optProb.tf_max = prob_setup.tf_max

# define obstacles
optProb.obs = prob_setup.obs

optProb.algorithm = "SCVX"

optProb.robot = multirotor_full_model_komo_scp.Multicopter(nr_motors, arm_length, prob_setup.t2w)

optProb.robot.dt = 1/prob_setup.t_steps_croco
print("dt: ", optProb.robot.dt * optProb.tf_max)

optProb.CHandler = collisionHandler(optProb.robot)
optProb.CHandler.obs_data = optProb.obs

if optProb.obs is not None:
    for obs in optProb.obs:
        optProb.CHandler.addObject(obs.type, obs.shape, obs.pos, obs.quat)

with open("data/data_quim/data_quim_yaml/croco_recovery.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    
states = np.array(data_loaded["states"])#["states"])
actions = np.vstack([np.array(data_loaded["actions"]), np.array([np.nan,np.nan,np.nan,np.nan])])#["actions"]), np.array([np.nan,np.nan,np.nan,np.nan])])

#data_load = ou.load_object("data/flip_4m_SCVX")
#states = data_load.data[:,:13]
#actions = data_load.data[:,13:]

solution = ou.OptSolution(states, actions, tdil = optProb.tf_max)

check, data, int_error, false_success, _ = check_solution(solution, optProb)

pD = ou.Opt_processData()
pD.robot = optProb.robot
pD.x0 = optProb.x0
pD.xf = optProb.xf
pD.t_dil = optProb.tf_max
pD.obs = optProb.obs
pD.data = np.hstack([states,actions])
pD.int_err = int_error

sol = {"croco":pD}

report_compare(sol, ["croco"])