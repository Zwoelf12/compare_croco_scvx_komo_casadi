from casadi import *
from casadi.tools import *
import optimization.opt_utils as ou
import pylab as pl
import time
import subprocess

def update_c_code(robot,discretization_method):

    if discretization_method == "RK":

        f = robot.step_RK
        x = SX.sym('x',robot.max_x.shape[0])
        u = SX.sym('u',robot.nrMotors)

        F = Function('f',[x,u],[f(x,u)])

        file_name_c = "step_RK.c"
        file_name_so = "step_RK.so"
        F.generate(file_name_c)

        cmd = "gcc -fPIC -shared {} -o {}".format(file_name_c, file_name_so)
        print("compiling c code")
        subprocess.run(cmd.split())
        print("compiling done")

    elif discretization_method == "euler":

        f = robot.step_euler

        x = SX.sym('x', robot.max_x.shape[0])
        u = SX.sym('u', robot.nrMotors)

        F = Function('f', [x, u], [f(x, u)])

        file_name_c = "step_euler.c"
        file_name_so = "step_euler.so"
        F.generate(file_name_c)

        cmd = "gcc -fPIC -shared {} -o {}".format(file_name_c, file_name_so)
        print("compiling c code")
        subprocess.run(cmd.split())
        print("compiling done")

    else:
        raise NameError("unknown discretization method")



def solve(robot, x0, xf, intermediate_states, t_final, obs, initial_x, initial_u, num_timesteps, discretization_method, use_c_code, alg_par):

    opt = casadi.Opti() # generate optimization problem

    ## define states and control
    nMotors = len(robot.min_u)
    X = opt.variable(13, num_timesteps) # optimization variables for state trajectory
    U = opt.variable(nMotors, num_timesteps-1) # optimization variables for control trajectory

    ## set initial guess
    opt.set_initial(X, initial_x.T)
    opt.set_initial(U, initial_u[:-1, :].T)

    ## chose discretization
    if discretization_method == "RK":
        f = robot.step_RK
        if use_c_code:
            f = external("f", "step_RK.so")
    elif discretization_method == "euler":
        f = robot.step_euler
        if use_c_code:
            f = external("f", "step_euler.so")
    else:
        raise NameError("unknown discretization method")

    ## define constraints
    # initial, final and intermediate constraints
    opt.subject_to(X[:,0] == x0) # initial
    opt.subject_to(X[:,-1] == xf) # final

    if intermediate_states is not None: # intermediate
        for i_s in intermediate_states:
            if "pos" in i_s.type:
                opt.subject_to(X[:3, i_s.timing] == i_s.value[:3])
            if "vel" in i_s.type:
                opt.subject_to(X[3:6, i_s.timing] == i_s.value[3:6])
            if "quat" in i_s.type:
                opt.subject_to(X[6:10, i_s.timing] == i_s.value[6:10])
            if "rot_vel" in i_s.type:
                opt.subject_to(X[10:, i_s.timing] == i_s.value[10:])

    for t in range(num_timesteps-1):

        # state and input constraints
        opt.subject_to(robot.min_x <= X[:, t])
        opt.subject_to(X[:, t] <= robot.max_x)
        opt.subject_to(robot.min_u <= U[:, t])
        opt.subject_to(U[:, t] <= robot.max_u)

        # collision constraints
        robot_position = X[:3, t]  # positions
        if obs:
            for obstacle in obs:
                if obstacle.type == "sphere":
                    obs_center = obstacle.pos
                    obs_radius = obstacle.shape[0]
                    opt.subject_to(norm_2(robot_position-obs_center) > obs_radius + robot.arm_length)

        # dynamic constraints
        x_next = f(X[:,t],U[:,t])
        opt.subject_to(X[:,t+1] == x_next)

    ## define objective
    opt.minimize(sumsqr(U)) # minimze energy

    ## solve optimization problem
    p_opt = {}
    s_opt = {"acceptable_constr_viol_tol":alg_par.acceptable_constr_viol_tol,
             "acceptable_tol":alg_par.acceptable_tol,
             "acceptable_iter":alg_par.acceptable_iter}
    
    opt.solver('ipopt',p_opt,s_opt)

    try:
        t_start_solver = time.time()
        sol = opt.solve()
        t_end_solver = time.time()

    except RuntimeError:
        print("Solver failed: Converged to a point of local infeasibility")
        # debugging
        #index = 33
        #opt.debug.value(X)
        #opt.debug.value(X,opt.initial())
        #opt.debug.show_infeasibilities()
        #opt.debug.x_describe(index)
        #opt.debug.g_describe(index)
    except:
        print("something went wrong")

    states = np.array(sol.value(X).T)
    actions = np.array(sol.value(U).T)

    solution = ou.OptSolution(states, actions, time = t_end_solver - t_start_solver, tdil = t_final, num_iter=sol.stats()["iter_count"], hessian_evals=sol.stats()["iter_count"])

    """
    ## plot Jacobian

    pl.figure()
    pl.spy(sol.value(jacobian(opt.g,opt.x)))
    pl.figure()
    pl.spy(sol.value(hessian(opt.f+dot(opt.lam_g,opt.g),opt.x)[0]))
    pl.show()
    """

    return solution

