#import sys
#import os
#sys.path.insert(0,os.getcwd()+"/scripts/")

from optimization import opt_utils as ou
from optimization.opt_problem import OptProblem
from optimization.check.check_solution import check_solution
from physics.multirotor_models import multirotor_full_model_komo_scp
from physics.multirotor_models import multirotor_full_model_casadi
from visualization.report_visualization import report_compare
from optimization import problem_setups

only_visualize = False
list_of_solvers = ["SCVX","CASADI"]

# choose which problem should be solved
prob = 2
if prob == 1:
    prob_setup = problem_setups.simple_flight_wo_obs()
    prob_name = "simple_flight_wo_obs"
elif prob == 2:
    prob_setup = problem_setups.complex_flight_spheres()
    prob_name = "complex_flight_spheres"
elif prob == 3:
    prob_setup = problem_setups.recovery_flight()
    prob_name = "recovery_flight"

# define optimization problem
optProb = OptProblem()

# define robot model
nr_motors = 4
arm_length = 0.046

if only_visualize == False:

    # define start and end point
    optProb.x0 = prob_setup.x0_fM
    optProb.xf = prob_setup.xf_fM

    # define maximum and minimum flight time
    optProb.tf_min = prob_setup.tf_min
    optProb.tf_max = prob_setup.tf_max

    # define obstacles
    optProb.obs = prob_setup.obs

    # define number of timesteps
    t_steps_scvx = prob_setup.t_steps_scvx
    t_steps_komo = prob_setup.t_steps_komo
    t_steps_casadi = prob_setup.t_steps_casadi

    optProb.initial_x, optProb.initial_u, optProb.initial_p = ou.calc_initial_guess(multirotor_full_model_komo_scp.Multicopter(nr_motors, arm_length, prob_setup.t2w),
                                                                                    t_steps_scvx,
                                                                                    prob_setup.noise,
                                                                                    optProb.xf,
                                                                                    optProb.x0,
                                                                                    optProb.tf_min,
                                                                                    optProb.tf_max)
    if "KOMO" in list_of_solvers:
        ######## solve problem with KOMO #######
        print("lego")
        """
        
        optProb.robot = multirotor_full_model_komo_scp.Multicopter(nr_motors, arm_length, prob_setup.t2w)

        optProb.algorithm = "KOMO"

        # define KOMO Parameter
        par = ou.Parameter_komo()
        par.phases = 1

        par.time_steps_per_phase = t_steps_komo

        par.time_per_phase = optProb.tf_max
        optProb.robot.dt = 1/(par.time_steps_per_phase)

        optProb.par = par

        print("solving optimization problem with KOMO...")
        solution = optProb.solve_problem()
        solution.time_dil = optProb.tf_max

        print("solver time KOMO: {}".format(solution.time))

        print("checking KOMO solution ...")
        check, data, int_error, false_success, _ = check_solution(solution, optProb)

        print("KOMO solution correct?: {}".format(check))
        print("KOMO solution falsly correct?: {}".format(false_success))
        
        ou.save_opt_output(optProb,
                           prob_name,
                           solution,
                           data,
                           real_traj,
                           int_error,
                           int_error_small_dt)
        
        """


    if "SCVX" in list_of_solvers:

        print("lego")

        """

        ######## solve problem with SCVX #######
        optProb.algorithm = "SCVX"

        optProb.robot = multirotor_full_model_komo_scp.Multicopter(nr_motors, arm_length, prob_setup.t2w)

        # define SCVX Parameter
        par = ou.Parameter_scvx()
        par.max_num_iter = 50
        par.num_time_steps = t_steps_scvx
        optProb.robot.dt = 1/(par.num_time_steps-1)
        print("dt SCVX: {}".format(optProb.robot.dt * optProb.tf))
        optProb.par = par

        # solve Problem with scvx
        print("solving optimization problem with SCVX...")
        solution = optProb.solve_problem()

        print("solver time SCVX: {}".format(solution.time))

        print("checking SCVX solution ...")
        check, data, int_error, _, _ = check_solution(solution, optProb)

        print("SCVX solution correct?: {}".format(check))

        ou.save_opt_output(optProb,
                           prob_name,
                           solution,
                           data,
                           real_traj,
                           int_error,
                           int_error_small_dt)
                           
        """


    if "CASADI" in list_of_solvers:

        ######## solve problem with CASADI ########
        optProb.algorithm = "CASADI"

        optProb.robot = multirotor_full_model_casadi.Multicopter(nr_motors, arm_length, prob_setup.t2w)

        # define Casadi Parameter
        par = ou.Parameter_casadi()
        par.num_time_steps = t_steps_casadi
        par.discretization_method = "RK"
        optProb.robot.dt = optProb.tf_max / par.num_time_steps
        print("dt CASADI: {}".format(optProb.robot.dt))
        optProb.par = par

        # solve Problem with casadi
        print("solving optimization problem with CASADI ...")
        solution = optProb.solve_problem()

        print("solver time CASADI: {}".format(solution.time))

        print("checking CASADI solution ...")
        check, data, int_error, _, _ = check_solution(solution, optProb)

        print("CASADI solution correct?: {}".format(check))

        ou.save_opt_output(optProb,
                           prob_name,
                           solution,
                           data,
                           int_error)

"""
sol_scvx, sol_komo = ou.load_opt_output(prob_name, robot_type, nr_motors)

report_compare(sol_scvx.robot,
                  sol_scvx.data,
                  sol_komo.data,
                  sol_scvx.real_traj,
                  sol_komo.real_traj,
                  sol_scvx.obs,
                  sol_scvx.int_err,
                  sol_scvx.int_err_small_dt,
                  sol_komo.int_err,
                  sol_komo.int_err_small_dt,
                  sol_scvx.x0, sol_scvx.xf, sol_scvx.t_dil)
"""




