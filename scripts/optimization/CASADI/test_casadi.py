from casadi import *
import sys
import os
sys.path.insert(0,os.getcwd()+"/scripts/")
from casadi.tools import *
from physics.multirotor_models import multirotor_full_casadi
from physics.multirotor_models import multirotor_full_model
from optimization.problem_setups import simple_flight_wo_obs
import optimization.opt_utils as ou
import numpy as np
from jax import jacfwd, jit, config
import matplotlib.pyplot as plt

def test_jacobians_casadi():
    # define robot
    nM = 4
    arm_length = 0.046

    prob = simple_flight_wo_obs()
    N = prob.t_steps_casadi # number of control points
    x0 = prob.x0_fM
    xf = prob.xf_fM
    tf = prob.tf_max
    noise = prob.noise

    # get jacobians of via casadi
    robot_cas = multirotor_full_casadi.QuadrotorAutograd(nM, arm_length)
    robot_cas.dt = tf/N
    robot_cas.t_dil_casadi = tf

    f_cas = robot_cas.step_euler

    x = SX.sym("x", 13)
    u = SX.sym("u", 4)

    jac_dyn_x_cas = Function("jac_dyn_x", [x, u], [jacobian(f_cas(x, u), x)])
    jac_dyn_u_cas = Function("jac_dyn_u", [x, u], [jacobian(f_cas(x, u), u)])

    # get jacobians via jax
    robot_jax = multirotor_full_model.QuadrotorAutograd(nM, arm_length)
    robot_jax.dt = tf/N

    f_jax = robot_jax.step

    jac_dyn_x_jax = jit(jacfwd(f_jax, 0))
    jac_dyn_u_jax = jit(jacfwd(f_jax, 1))

    # get initial guess to evaluate jacobians
    initial_x, initial_u, _ = ou.calc_initial_guess(robot_cas, N, noise, xf, x0, tf, tf)

    # compare jacobians
    flags = []
    for t in range(N-1):

        j_x_cas = np.array(jac_dyn_x_cas(initial_x[t,:].T, initial_u[t,:].T))
        j_u_cas = np.array(jac_dyn_u_cas(initial_x[t, :].T, initial_u[t, :].T))

        j_x_jax = np.array(jac_dyn_x_jax(initial_x[t,:],initial_u[t,:], tf))
        j_u_jax = np.array(jac_dyn_u_jax(initial_x[t, :], initial_u[t, :], tf))

        flags.append(np.allclose(j_x_jax, j_x_cas))
        flags.append(np.allclose(j_u_jax, j_u_cas))


    assert np.all(flags)


def test_casadi_optimization():

    N = 100  # number of control intervals

    opti = Opti()  # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(2, N + 1)  # state trajectory
    pos = X[0, :]
    speed = X[1, :]
    U = opti.variable(1, N)  # control trajectory (throttle)
    T = opti.variable()  # final time

    # ---- objective          ---------
    opti.minimize(T)  # race in minimal time

    # ---- dynamic constraints --------
    f = lambda x, u: vertcat(x[1], u - x[1])  # dx/dt = f(x,u)

    dt = T / N  # length of a control interval
    for k in range(N):  # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # ---- path constraints -----------
    limit = lambda pos: 1 - sin(2 * pi * pos) / 2
    opti.subject_to(speed <= limit(pos))  # track speed limit
    opti.subject_to(opti.bounded(0, U, 1))  # control is limited

    # ---- boundary conditions --------
    opti.subject_to(pos[0] == 0)  # start at position 0 ...
    opti.subject_to(speed[0] == 0)  # ... from stand-still
    opti.subject_to(pos[-1] == 1)  # finish line at position 1

    # ---- misc. constraints  ----------
    opti.subject_to(T >= 0)  # Time must be positive

    # ---- initial values for solver ---
    opti.set_initial(speed, 1)
    opti.set_initial(T, 1)

    # ---- solve NLP              ------
    opti.solver("ipopt")  # set numerical backend
    sol = opti.solve()  # actual solve

    assert opti.stats()["success"]
