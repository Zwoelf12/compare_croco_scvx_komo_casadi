import numpy as np
import rowan
import copy

def check_solution(solution, optProb):

    states = solution.states
    actions = solution.actions
    t_dil = solution.time_dil

    robot = optProb.robot

    check_dt = 0.2 # substepsize with which every state is checked
    robot_small_dt = copy.deepcopy(robot)
    robot_small_dt.dt = check_dt

    def check_array(a, b, msg, check_quat_diff = False):
        if check_quat_diff:
            quat_diff = rowan.geometry.distance(a[6:10], b[6:10])
            success = np.allclose(a[:6], b[:6], rtol=0.01, atol=1e-3)
            success &= np.allclose(a[10:], b[10:], rtol=0.01, atol=0.8)
            success &= quat_diff <= 1e-3
        else:
            success = np.allclose(a, b, rtol=0.01, atol=1e-3)
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

    if robot.type == "dI":
        success &= check_array(states[0], optProb.x0, "start state")
        success &= check_array(states[-1], optProb.xf, "end state")
    else:
        if robot.nrMotors != 2 and robot.nrMotors != 3:
            success &= check_array(states[0], optProb.x0, "start state", True)
            success &= check_array(states[-1], optProb.xf, "end state", True)
        else:
            success &= check_array(states[0], optProb.x0, "start state")
            success &= check_array(states[-1,:6], optProb.xf[:6], "end state")

    # dynamics
    int_err = []
    real_traj = [np.array(optProb.x0)]
    T = states.shape[0]

    # check with finer stepsize
    int_err_small_grid = []

    print("propagating data ...")
    for t in range(T - 1):
        # calculate integration error
        if optProb.algorithm == "SCVX":
            state_desired_big_grid = robot.step(states[t], actions[t], t_dil)
        else:
            state_desired_big_grid = robot.step_KOMO(states[t], states[t+1], actions[t], t_dil)

        state_desired_small_grid = calc_desired_state(robot_small_dt, states[t], actions[t], t_dil/(T-1))

        if robot.type == "dI":
            success &= check_array(states[t + 1], state_desired_big_grid, "Wrong dynamics at t={}".format(t + 1))
            int_err.append(states[t + 1] - state_desired_big_grid)
            int_err_small_grid.append(states[t + 1] - state_desired_small_grid)
        else:
            success &= check_array(states[t + 1], state_desired_big_grid, "Wrong dynamics at t={}".format(t + 1), True)
            quat_diff = rowan.geometry.sym_distance(states[t+1, 6:10], state_desired_big_grid[6:10])
            err = np.array(states[t + 1] - state_desired_big_grid)
            err[6:10] = quat_diff
            int_err.append(err)

            quat_diff_small_grid = rowan.geometry.sym_distance(states[t + 1, 6:10], state_desired_small_grid[6:10])
            err_small_grid = np.array(states[t + 1] - state_desired_small_grid)
            err_small_grid[6:10] = quat_diff_small_grid
            int_err_small_grid.append(err_small_grid)

        # calculate real trajectory
        real_traj.append(robot.step(real_traj[-1], actions[t], t_dil))

    # state limits
    for t in range(T):
        if robot.type == "dI" or (robot.nrMotors != 2 and robot.nrMotors != 3):
            if (states[t] > robot.max_x + 1e-2).any() or (states[t] < robot.min_x - 1e-2).any():
                print("State outside bounds at t={} ({})".format(t, states[t]))
                success = False
        else:
            if (states[t,:10] > robot.max_x[:10] + 1e-2).any() or (states[t,:10] < robot.min_x[:10] - 1e-2).any():
                print("State outside bounds at t={} ({})".format(t, states[t]))
                success = False

    # action limits
    for t in range(T - 1):
        if (actions[t] > robot.max_u + 1e-3).any() or (actions[t] < robot.min_u - 1e-3).any():
            print("Action outside bounds at t={} ({})".format(t, actions[t]))
            success = False

    # collisions
    for t in range(T):
        dist, pRob, pObs = optProb.CHandler.calcDistance_broadphase(states[t])
        if dist < -0.03:  # allow up to 3cm violation
            print("Collision at t={} ({})".format(t, dist))
            success = False

    # check if constraint violations in KOMO are reasonable (not really necessary?)
    false_success = False
    if optProb.algorithm == "KOMO":
        if solution.constr_viol > 1 and success:
            success = False
            false_success = True

    data = np.hstack((states, actions))

    return success, data, np.array(real_traj), np.array(int_err), np.array(int_err_small_grid), false_success, solution.constr_viol

def calc_desired_state(robot, state, action, t_dil):
    num_timesteps = int(np.ceil(1/robot.dt))

    state_now = state
    for t in range(num_timesteps):
        state_now = robot.step(state_now, action, t_dil)

    return state_now

