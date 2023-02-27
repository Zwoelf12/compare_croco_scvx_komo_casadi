from casadi import *
from casadi.tools import *
import optimization.opt_utils as ou
import pylab as pl
import time

def solve(robot, x0, xf, xm, xm_timing, t_final, obs, initial_x, initial_u, num_timesteps, discretization_method):

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
    elif discretization_method == "euler":
        f = robot.step_euler
    else:
        raise NameError("unknown discretization method")

    ## define constraints
    # initial, final and intermediate constraints
    opt.subject_to(X[:,0] == x0) # initial
    opt.subject_to(X[:,-1] == xf) # final
    if xm is not None and xm_timing:
        opt.subject_to(X[6:10,xm_timing] == xm[6:10]) # apply intermediate constraints only to the orientation

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
    opt.solver('ipopt')

    try:
        t_start_solver = time.time()
        sol = opt.solve()
        t_end_solver = time.time()

    except RuntimeError:
        print("##### DEBUGGING #####")
        # debugging
        #index = 33
        #opt.debug.value(X)
        #opt.debug.value(X,opt.initial())
        #opt.debug.show_infeasibilities()
        #opt.debug.x_describe(index)
        #opt.debug.g_describe(index)

    states = np.array(sol.value(X).T)
    actions = np.array(sol.value(U).T)

    solution = ou.OptSolution(states, actions, time = t_end_solver - t_start_solver, tdil = t_final)

    """
    ## plot Jacobian

    pl.figure()
    pl.spy(sol.value(jacobian(opt.g,opt.x)))
    pl.figure()
    pl.spy(sol.value(hessian(opt.f+dot(opt.lam_g,opt.g),opt.x)[0]))
    pl.show()
    """

    return solution

