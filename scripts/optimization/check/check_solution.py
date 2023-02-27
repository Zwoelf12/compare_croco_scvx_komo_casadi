import numpy as np
import rowan
import copy

def check_solution(solution, optProb):

    states = solution.states
    actions = solution.actions
    t_dil = solution.time_dil

    robot = optProb.robot

    def check_array(a, b, msg):
        quat_diff = rowan.geometry.distance(a[6:10], b[6:10])
        success = np.allclose(a[:6], b[:6], rtol=0.01, atol=1e-3)
        success &= np.allclose(a[10:], b[10:], rtol=0.01, atol=0.8)
        success &= quat_diff <= 1e-3
        if not success:
            print("{} \n Is: {} \n Should: {} \n Delta: {}".format(msg, a, b, a - b))
        return success

    success = True
    if states.shape[1] != len(robot.min_x):
        print("Wrong state dimension!")
        success = False
    if actions.shape[1] != len(robot.min_u):
        print("Wrong action dimension!")
        success = False

    if robot.nrMotors != 2 and robot.nrMotors != 3:
        success &= check_array(states[0], optProb.x0, "start state")
        success &= check_array(states[-1], optProb.xf, "end state")
        if optProb.xm is not None and optProb.xm_timing:
            success &= check_array(states[optProb.xm_timing, 6:10], optProb.xm[6:10], "intermediate state")
    else:
        success &= check_array(states[0], optProb.x0, "start state")
        success &= check_array(states[-1,:6], optProb.xf[:6], "end state")
        if optProb.xm is not None and optProb.xm_timing:
            success &= check_array(states[optProb.xm_timing, 6:10], optProb.xm[6:10], "intermediate state")

    # dynamics
    int_err = []
    T = states.shape[0]

    print("propagating data ...")
    for t in range(T - 1):
        # calculate integration error
        if optProb.algorithm == "SCVX":
            state_desired_big_grid = robot.step(states[t], actions[t], t_dil)
        elif optProb.algorithm == "CASADI":
            if optProb.par.discretization_method == "RK":
                state_desired_big_grid = robot.step_RK(states[t], actions[t])
            elif optProb.par.discretization_method == "euler":
                state_desired_big_grid = np.array(robot.step_euler(states[t], actions[t])).squeeze()
            else:
                print("unknown discretization scheme")
        else:
            state_desired_big_grid = robot.step_KOMO(states[t], states[t+1], actions[t], t_dil)

        success &= check_array(states[t + 1], state_desired_big_grid, "Wrong dynamics at t={}".format(t + 1))
        quat_diff = rowan.geometry.sym_distance(states[t+1, 6:10], state_desired_big_grid[6:10])
        err = np.array(states[t + 1] - state_desired_big_grid)
        err[6:10] = quat_diff
        int_err.append(err)

    # state limits
    for t in range(T):
        if (states[t,:10] > robot.max_x[:10] + 1e-2).any() or (states[t,:10] < robot.min_x[:10] - 1e-2).any():
            print("State outside bounds at t={} ({})".format(t, states[t]))
            success = False

    # action limits
    for t in range(T - 1):
        if (actions[t] > robot.max_u + 1e-3).any() or (actions[t] < robot.min_u - 1e-3).any():
            print("Action outside bounds at t={} ({})".format(t, actions[t]))
            success = False

    # collisions
    #for t in range(T):
    #    dist, pRob, pObs = optProb.CHandler.calcDistance_broadphase(states[t])
    #    if dist < -0.03:  # allow up to 3cm violation
    #        print("Collision at t={} ({})".format(t, dist))
    #        success = False

    # check if constraint violations in KOMO are reasonable (not really necessary?)
    false_success = False
    if optProb.algorithm == "KOMO":
        if solution.constr_viol > 1 and success:
            success = False
            false_success = True

    actions_nan = np.empty(robot.nrMotors)
    actions_nan[:] = np.nan
    data = np.hstack((states, np.vstack((actions, actions_nan))))

    return success, data, np.array(int_err), false_success, solution.constr_viol

