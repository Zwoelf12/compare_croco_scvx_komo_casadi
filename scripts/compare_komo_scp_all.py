from optimization import opt_utils as ou
from optimization.opt_problem import OptProblem
from scripts.optimization.check.check_solution import check_solution
from physics.multirotor_models.multirotor_double_integrator import QuadrotorDoubleInt
from physics.multirotor_models.multirotor_full_model import QuadrotorAutograd
from visualization.report_visualization import report_noise
from optimization import problem_setups
import matplotlib.pyplot as plt

only_visualize = True

# choose which problems should be solved
probs = [3]
path = "data/data_scvx_komo_comparision/"  # path for normal comparison
#path = "data/data_scvx_komo_comparision/prone_to_noise/" # path for tests to evaluate how prone the algorithms are to noise

# define optimization problem
optProb = OptProblem()

# define robot model
nTrials = 15
robot_type = "fM"
nr_motors = [8, 6, 4]
t2w_vec = [1.5]
# define noise on initial guess
noise_vec = [0.01]

num_Itr = len(probs)*len(nr_motors)*len(t2w_vec)*len(noise_vec)
itr = 0
for noise in noise_vec:
    for prob in probs:
        for t2w_fac in t2w_vec:
            for nrM in nr_motors:
                itr += 1
                print("Test {} of {}".format(itr, num_Itr))

                if prob == 1:
                    prob_setup = problem_setups.simple_flight_wo_obs()
                    prob_name = "simple_flight_wo_obs"
                elif prob == 2:
                    prob_setup = problem_setups.complex_flight_spheres()
                    prob_name = "complex_flight_spheres"
                elif prob == 3:
                    prob_setup = problem_setups.recovery_flight()
                    prob_name = "recovery_flight"

                if only_visualize == False:

                    if robot_type == "dI":
                        arm_length = 1e-4
                        optProb.robot = QuadrotorDoubleInt(arm_length)

                        # define start and end point
                        optProb.x0 = prob_setup.x0_dI
                        optProb.xf = prob_setup.xf_dI

                    elif robot_type == "fM":
                        arm_length = 0.046
                        optProb.robot = QuadrotorAutograd(nrM, arm_length, t2w_fac)

                        # define start and end point
                        optProb.x0 = prob_setup.x0_fM
                        optProb.xf = prob_setup.xf_fM

                    # set thrust to weight ratio
                    optProb.robot.t2w = t2w_fac

                    # define maximum and minimum flight time
                    optProb.tf_min = prob_setup.tf_min
                    optProb.tf_max = prob_setup.tf_max

                    # define obstacles
                    optProb.obs = prob_setup.obs

                    # define number of timesteps
                    t_steps_scvx = prob_setup.t_steps_scvx  # dI simple flight 30 ,full model only z 30
                    t_steps_komo = prob_setup.t_steps_komo  # dI simple flight 30 ,full model only z 30

                    # data komo
                    data_list_komo = []
                    time_list_komo = []
                    check_list_komo = []
                    check_spec_list_komo = []
                    constr_viol_list = []

                    # data scvx
                    data_list_scvx = []
                    time_list_scvx = []
                    cvx_time_list_scvx = []
                    cvxpy_time_list_scvx = []
                    check_list_scvx = []
                    itr_list_scvx = []
                    int_error_komo_list = []
                    int_error_scvx_list = []
                    int_error_komo_small_step_size_list = []
                    int_error_scvx_small_step_size_list = []

                    # initial guess data
                    initial_guess_list = []

                    for i in range(nTrials):

                        print("Trial: {} of {}".format(i, nTrials))

                        # calculate initial guess
                        optProb.initial_x, optProb.initial_u, optProb.initial_p = ou.calc_initial_guess(optProb.robot,
                                                                                                        t_steps_scvx,
                                                                                                        noise,
                                                                                                        optProb.xf,
                                                                                                        optProb.x0,
                                                                                                        optProb.tf_min,
                                                                                                        optProb.tf_max)

                        initial_guess_list.append([optProb.initial_x, optProb.initial_u, optProb.initial_p])

                        ######## solve problem with KOMO #######
                        optProb.algorithm = "KOMO"

                        # define KOMO Parameter
                        par = ou.Parameter_KOMO()
                        par.phases = 1

                        if robot_type == "dI":
                            par.time_steps_per_phase = t_steps_komo - 1
                        elif robot_type == "fM":
                            par.time_steps_per_phase = t_steps_komo

                        par.time_per_phase = optProb.tf_max
                        optProb.robot.dt = 1 / (par.time_steps_per_phase)

                        optProb.par = par

                        # solve the optimization problem
                        print("solving optimization problem with KOMO...")
                        solution_komo = optProb.solve_problem()
                        solution_komo.time_dil = optProb.tf_max

                        # check solution for feasibility with respect to the original constraints
                        print("checking KOMO solution ...")
                        check, data_komo, real_traj_komo, int_error_komo, int_error_komo_small_step_size, check_spec, constraint_viol = check_solution(solution_komo, optProb)

                        print("KOMO solution correct?: {}".format(check))

                        # append data
                        data_list_komo.append(data_komo)
                        time_list_komo.append(solution_komo.time)
                        check_list_komo.append(check)
                        check_spec_list_komo.append(check_spec)
                        int_error_komo_list.append(int_error_komo)
                        int_error_komo_small_step_size_list.append(int_error_komo_small_step_size)
                        constr_viol_list.append(constraint_viol)

                        ######## solve problem with SCVX #######
                        optProb.algorithm = "SCVX"

                        # define SCVX Parameter
                        par = ou.Parameter_scvx()
                        par.tf_min = optProb.tf_min
                        par.tf_max = optProb.tf_max

                        par.max_num_iter = 50
                        par.num_time_steps = t_steps_scvx
                        optProb.robot.dt = 1/(par.num_time_steps-1)
                        print("dt SCVX: {}".format(optProb.robot.dt))

                        optProb.par = par

                        # solve the optimization problem
                        print("solving optimization problem with SCVX...")
                        solution_scvx = optProb.solve_problem()

                        print("solver time SCVX: {}".format(solution_scvx.time))

                        # check solution for feasibility with respect to the original constraints
                        print("checking SCVX solution ...")
                        check, data_scvx, real_traj_scvx, int_error_scvx, int_error_scvx_small_step_size, _, _ = check_solution(solution_scvx,optProb)

                        print("SCVX solution correct?: {}".format(check))

                        # append data
                        data_list_scvx.append(data_scvx)
                        time_list_scvx.append(solution_scvx.time)
                        cvx_time_list_scvx.append(solution_scvx.time_cvx_solver)
                        cvxpy_time_list_scvx.append(solution_scvx.time_cvxpy)
                        check_list_scvx.append(check)
                        itr_list_scvx.append(solution_scvx.num_iter)
                        int_error_scvx_list.append(int_error_scvx)
                        int_error_scvx_small_step_size_list.append(int_error_scvx_small_step_size)

                    collection_komo = {
                        "data": data_list_komo,
                        "time": time_list_komo,
                        "check": check_list_komo,
                        "check_spec": check_spec_list_komo,
                        "init_guess": initial_guess_list,
                        "int_error": int_error_komo_list,
                        "int_error_small_step_size": int_error_komo_small_step_size_list,
                        "constr_viol": constr_viol_list
                    }

                    collection_scvx = {
                        "data": data_list_scvx,
                        "time": time_list_scvx,
                        "cvx_time": cvx_time_list_scvx,
                        "cvxpy_time": cvxpy_time_list_scvx,
                        "check": check_list_scvx,
                        "num_itr": itr_list_scvx,
                        "init_guess": initial_guess_list,
                        "int_error": int_error_scvx_list,
                        "int_error_small_step_size": int_error_scvx_small_step_size_list
                    }

                    ou.save_object(path + prob_name + "_" + robot_type + "_nrM" + str(nrM) + "_t2w" + str(optProb.robot.t2w) + "_noise" + str(noise) + "_adaptiveWeights_False2_komo", collection_komo)
                    ou.save_object(path + prob_name + "_" + robot_type + "_nrM" + str(nrM) + "_t2w" + str(optProb.robot.t2w) + "_noise" + str(noise) + "_adaptiveWeights_False2_scvx", collection_scvx)

for prob in probs:

    if prob == 1:
        prob_setup = problem_setups.simple_flight_wo_obs()
        prob_name = "simple_flight_wo_obs"
    elif prob == 2:
        prob_setup = problem_setups.complex_flight_spheres()
        prob_name = "complex_flight_spheres"
    elif prob == 3:
        prob_setup = problem_setups.recovery_flight()
        prob_name = "recovery_flight"

    report_noise(prob_name, path, nr_motors, t2w_vec, noise_vec, robot_type, prob_setup)

plt.show()

