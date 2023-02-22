import yaml
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)

import sys
import os
sys.path.append(os.getcwd())
from motionplanningutils import CollisionChecker
from optimization.CROCO.croco_models import *

#from scp import SCP
from scripts.deprecated.robots import Quadrotor
import inspect

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from optimization.opt_utils import OptSolution

def croc_problem_from_env(D):

    cro = OCP_abstract()

    #robot = Quadrotor(free_time=False, semi_implicit=False, normalize=False)

    robot = D["robot"]
    D["min_u"] = np.array(robot.min_u)
    D["max_u"] = np.array(robot.max_u)
    D["min_x"] = np.array(robot.min_x)
    D["max_x"] = np.array(robot.max_x)

    cro = OCP_quadrotor(**D, use_jit=False)

    return cro


def sample_at_dt(X, U):

    # the third component of U is dt scaling

    dts = np.array([0] + [u[-1] for u in U])
    dts_accumulated = np.cumsum(dts)
    print("dts", dts)
    print("dts_accumulated", dts_accumulated)
    print("total time", dts_accumulated[-1])

    query_t = np.arange(np.ceil(dts_accumulated[-1]) + 1)

    Xnew = []
    Unew = []

    for i in range(len(X[0])):
        x_a = np.array([x[i] for x in X])
        x_a_new = np.interp(query_t, dts_accumulated, x_a)

        plt.plot(dts_accumulated, x_a, 'o-', label="dat")
        plt.plot(query_t, x_a_new, 'o-', label="query")
        plt.legend()
        plt.show()

        Xnew.append(x_a_new)

    for i in range(len(U[0]) - 1):
        u_a = np.array([u[i] for u in U])
        u_a_new = np.interp(query_t[:-1], dts_accumulated[:-1], u_a)

        plt.plot(dts_accumulated[:-1], u_a, 'o-', label="dat")
        plt.plot(query_t[:-1], u_a_new, 'o-', label="query")
        plt.legend()
        plt.show()

        Unew.append(u_a_new)

    _Xnew = np.stack(Xnew)  # num_coordinates x num_datapoints
    _Unew = np.stack(Unew)  # num_coordinates x num_datapoints

    print("plotting all")
    for i, x in enumerate(_Xnew):
        plt.plot(query_t, x, 'o-', label=f"x{i}")

    for i, u in enumerate(_Unew):
        plt.plot(query_t[:-1], u, 'o-', label=f"u{i}")

    plt.legend()
    plt.show()

    _Xnew = np.transpose(_Xnew)  # num_datapoints x num_coordinates
    _Unew = np.transpose(_Unew)  # num_datapoints x num_coordinates

    return _Xnew, _Unew


def run_croco(filename_env, filename_initial_guess, robot, visualize, free_time=False):

    # TODO: add option to use trivial inital guess (either
    # u = udefault (zeros or hover for quadrotor) and fixed x0
    # or u = default and linear interpolation between x0 and goal for x.
    with open(filename_env) as f:
        env = yaml.safe_load(f)

    with open(filename_initial_guess) as f:
        guess = yaml.safe_load(f)

    robot_node = env["robots"][0]

    # define initial and final state
    # format: (13,), (13,)
    goal = np.array(robot_node["goal"])
    print("shape xf: ", goal.shape)
    x0 = np.array(robot_node["start"])
    print("shape x0: ", x0.shape)

    vis = None

    # use C++ robot model to define collision model (spherical approximation)
    cc = CollisionChecker()
    cc.load(filename_env)

    obs = []
    for obstacle in env["environment"]["obstacles"]:
        obs.append([obstacle["center"], obstacle["size"]])

    # define initial guess
    # format: X: (50,13), U: (49,4)
    X = guess["result"][0]["states"]
    print("shape inital X: ", np.array(X).shape)
    U = guess["result"][0]["actions"]
    print("shape initial U: ", np.array(U).shape)

    assert len(X) == len(U) + 1
    Xl = [np.array(x) for x in X]
    Xl[0] = np.array(x0)  # make sure intial state is equal to x0
    Ul = [np.array(u) for u in U]

    print(f"*** XL *** n= { len(Xl)}")
    _ = [print(x) for x in Xl]
    print(f"*** UL *** n ={len(Ul)}")
    _ = [print(u) for u in Ul]

    if vis is None:
        def plot_fn(xs, us):
            crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
            plt.show()

    if visualize:
        crocoddyl.plotOCSolution(Xl, Ul, figIndex=1, show=False)
        plt.show()

    # define number of timesteps
    # Format: 49
    T = len(guess["result"][0]["states"]) - \
        1  # number of steps = num of actions
    name = env["robots"][0]["type"]

    # GENERATE THE PROBLEM IN CROCODYLE
    print(len(Xl))
    print(len(Ul))

    robot_node = env["robots"][0]
    x0 = np.array(robot_node["start"])
    goal = np.array(robot_node["goal"])
    D = {"goal": goal, "x0": x0, "cc": cc, "T": T, "free_time": free_time,
         "name": name, "robot": robot}

    D["col"] = False if not obs else True

    problem = croc_problem_from_env(D)
    problem.recompute_init_guess(Xl, Ul)
    solver = crocoddyl.SolverBoxFDDP
    ddp = solver(problem.problem)
    ddp.th_acceptNegStep = .3
    ddp.th_stop = 1e-2
    ddp.setCallbacks([crocoddyl.CallbackLogger(),
                      crocoddyl.CallbackVerbose()])

    reguralize_wrt_init_guess = True
    if reguralize_wrt_init_guess:
        aa = 0
        for i in problem.featss:
            for jj in i.list_feats:
                if jj.fn.name == "regx":
                    jj.fn.ref = np.copy(Xl[aa])
                    aa += 1

    auglag_solver(
        ddp,
        Xl,
        Ul,
        problem,
        np.zeros(3),
        visualize,
        plot_fn,
        max_it_ddp=20,
        max_it=5)

    X = [x.tolist() for x in ddp.xs]
    U = [x.tolist() for x in ddp.us]

    reoptimize_fixed_time = False
    if reoptimize_fixed_time:
        _Xnew, _Unew = sample_at_dt(X, U)

        xs_rollout = []

        xs_rollout.append(x0)
        x = np.copy(x0)
        model_ = problem.running_model[0]
        model_.robot.free_time = False
        for u in _Unew:
            xnew = model_.robot.step(x, u)
            x = np.copy(xnew)
            xs_rollout.append(x)

        print("no we areeee doing this")

        plot_rollouts = False
        if plot_rollouts:

            print("plotting rollouts")
            for i in range(len(_Unew[0])):
                us = [u[i] for u in _Unew]
                plt.plot(us, 'o-', label=f"u{i}")

            for i in range(len(xs_rollout[0])):
                xs = [x[i] for x in xs_rollout]
                plt.plot(xs, 'o-', label=f"x{i}-rollout")

            for i in range(len(_Xnew[0])):
                xs = [x[i] for x in _Xnew]
                plt.plot(xs, 'o-', label=f"x{i}")

            plt.legend()
            plt.show()

        Xl = [np.array(x) for x in _Xnew.tolist()]
        Ul = [np.array(u) for u in _Unew.tolist()]

        D["free_time"] = False
        D["T"] = _Unew.shape[0]
        problem = croc_problem_from_env(D)
        problem.recompute_init_guess(Xl, Ul)
        solver = crocoddyl.SolverBoxFDDP
        ddp = solver(problem.problem)
        ddp.th_acceptNegStep = .3
        ddp.th_stop = 1e-2
        ddp.setCallbacks([crocoddyl.CallbackLogger(),
                          crocoddyl.CallbackVerbose()])

        reguralize_wrt_init_guess = True
        if reguralize_wrt_init_guess:
            aa = 0
            for i in problem.featss:
                for jj in i.list_feats:
                    if jj.fn.name == "regx":
                        jj.fn.ref = np.copy(Xl[aa])
                        aa += 1

        auglag_solver(
            ddp,
            Xl,
            Ul,
            problem,
            np.zeros(3),
            visualize,
            plot_fn,
            max_it_ddp=20,
            max_it=5)
        X = [x.tolist() for x in ddp.xs]
        U = [x.tolist() for x in ddp.us]

    problem.normalize(X, U)

    solution = OptSolution(X,U)

    return solution