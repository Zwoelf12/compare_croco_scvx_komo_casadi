import numpy as np
import rowan
import copy

def check_solution(solution, optProb):

    states = solution.states
    actions = solution.actions
    t_dil = solution.time_dil

    robot = optProb.robot

    def check_array(a, b, msg):
        quat_diff = rowan.geometry.sym_distance(a[6:10], b[6:10])
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

        if optProb.intermediate_states is not None:
            for i_s in optProb.intermediate_states:
                if "pos" in i_s.type:
                    if not np.allclose(states[i_s.timing, :3], i_s.value[:3], rtol=0.01, atol=1e-3):
                        success &= False
                        print("intermediate state pos is false: \n should: {} \n is: {}".format(i_s.value[:3],states[i_s.timing, :3]))
                if "vel" in i_s.type:
                    if not np.allclose(states[i_s.timing, 3:6], i_s.value[3:6], rtol=0.01, atol=1e-3):
                        success &= False
                        print("intermediate state vel is false: \n should: {} \n is: {}".format(i_s.value[3:6],states[i_s.timing, 3:6]))
                if "quat" in i_s.type:
                    quat_diff = rowan.geometry.sym_distance(states[i_s.timing, 6:10], i_s.value[6:10])
                    if quat_diff > 1e-3:
                        success &= False
                        print("intermediate state quat is false: \n should: {} \n is: {}".format(i_s.value[6:10],states[i_s.timing, 6:10]))
                if "rot_vel" in i_s.type:
                    if not np.allclose(states[i_s.timing, 10:], i_s.value[10:], rtol=0.01, atol=1e-3):
                        success &= False
                        print("intermediate state rot vel is false: \n should: {} \n is: {}".format(i_s.value[10:],states[i_s.timing, 10:]))
    else:
        success &= check_array(states[0], optProb.x0, "start state")
        success &= check_array(states[-1,:6], optProb.xf[:6], "end state")

        if optProb.intermediate_states is not None:
            for i_s in optProb.intermediate_states:
                if i_s.type == "pos":
                    success &= check_array(states[i_s.timing, :3], i_s.value[:3], "intermediate state pos")
                if i_s.type == "vel":
                    success &= check_array(states[i_s.timing, 3:6], i_s.value[3:6], "intermediate state vel")
                #if i_s.type == "quat":
                #    success &= check_array(states[i_s.timing, 6:10], i_s.value[6:10], "intermediate state quat")
                #if i_s.type == "rot_vel":
                #    success &= check_array(states[i_s.timing, 10:], i_s.value[10:], "intermediate state rot_vel")

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
        elif optProb.algorithm == "CROCO":
            state_desired_big_grid = robot.step(states[t], actions[t], t_dil)
        else:
            state_desired_big_grid = robot.step_KOMO(states[t], states[t+1], actions[t], t_dil)

        success &= check_array(states[t + 1], state_desired_big_grid, "Wrong dynamics at t={}".format(t + 1))
        quat_diff = rowan.geometry.sym_distance(states[t+1, 6:10], state_desired_big_grid[6:10])
        err = np.array(states[t + 1] - state_desired_big_grid)
        err[6:10] = quat_diff
        int_err.append(err)

    # state limits
    if optProb.algorithm == "CROCO":
        for t in range(T):
            if (np.isnan(states[t])).any():
                success = False
            else:
                if (states[t,3:] > robot.max_x[3:] + 1e-2).any() or (states[t,3:] < robot.min_x[3:] - 1e-2).any():
                    print("State outside bounds at t={} ({})".format(t, states[t]))
                    success = False
    else:
        for t in range(T):
            if (np.isnan(states[t])).any():
                success = False
            else:
                if optProb.algorithm == "KOMO":
                    tol = 1e-1
                else:
                    tol = 1e-2
                if (states[t] > robot.max_x + tol).any() or (states[t] < robot.min_x - tol).any():
                    print("State outside bounds at t={} ({})".format(t, states[t]))
                    success = False

    # action limits
    for t in range(T - 1):
        if (np.isnan(actions[t])).any():
                success = False
        else:
            if (actions[t] > robot.max_u + 1e-3).any() or (actions[t] < robot.min_u - 1e-3).any():
                print("Action outside bounds at t={} ({})".format(t, actions[t]))
                success = False

    # collisions
    for t in range(T):
        if (np.isnan(states[t])).any():
            success = False
        else:
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

    if optProb.algorithm == "CASADI":
        actions_nan = np.empty(robot.nrMotors)
        actions_nan[:] = np.nan
        actions = np.vstack((actions, actions_nan))

    data = np.hstack((states, actions))

    return success, data, np.array(int_err), false_success, solution.constr_viol

