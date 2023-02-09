import numpy as np
from optimization import opt_utils as ou
from optimization.opt_problem_paper_comparision import OptProblem
from physics.multirotor_models.multirotor_double_integrator_paper_comparison import QuadrotorDoubleInt

# define optimization problem
optProb = OptProblem()
optProb.algorithm = "SCVX"

# define robot model
robot_type = "dI"
nr_motors = 4
arm_length = 1e-100
optProb.robot = QuadrotorDoubleInt(arm_length)

# sphere problem
optProb.x0 = np.array([0., 0., 1.,
                       0., 0., 0.])

optProb.xf = np.array([0.2, 2., 1.,
                       0., 0., 0.])

obj1 = ou.Obstacle("sphere", [0.5], [0,1,0.9], [1,0,0,0])


optProb.obs = [obj1]

optProb.tf_min = 2.5
optProb.tf_max = 2.5

# define SCVX Parameter
par = ou.Parameter_scvx()
par.max_num_iter = 15
par.num_time_steps = 30
optProb.robot.dt = 1/(par.num_time_steps-1)
print("dt: {}".format(optProb.robot.dt))

optProb.par = par

print("solving optimization problem ...")
solution = optProb.solve_problem(par)
