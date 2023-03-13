import sys
sys.path += ['../build/deps/rai']
import libry as ry
import numpy as np
from physics.multirotor_models import multirotor_flex_komo

def fullModel_flight(phases,
                  timeStepspP,
                  timepP,
                  x0,
                  xf,
                  intermediate_states,
                  obstacles,
                  robot,
                  alg_par,
                  initial_x,
                  initial_u):

    C = multirotor_flex_komo.build_copter(robot.nrMotors, robot.arm_length)

    C.addObject(name='target',
                parent='world',
                shape=ry.ST.ssBox,
                pos=xf[:3],
                quat=xf[6:10],
                size=[.05, .05, .05, .0002],
                color=[0., 1., 0.])
    C.frame("target").setContact(False)

    multiCopter = C.frame("drone")

    # set inital state
    multiCopter.setPosition(x0[:3])
    multiCopter.setQuaternion(x0[6:10])

    J = np.array([robot.J[0], 0, 0,
                  0, robot.J[1], 0,
                  0, 0, robot.J[2]])

    multiCopter.setMass(robot.mass)  # to make sure that all methods use the same mass
    multiCopter.setInertia(J, robot.mass)

    thrust_to_torque = robot.t2t
    max_force_per_motor = robot.max_u[0]

    komo = C.komo(phases, timeStepspP, timepP, 2, True)

    # add obstacles
    obs_num = 1
    if obstacles is not None:
        for obs in obstacles:
            if obs.type == "box":
                C.addObject(name="obs" + str(obs_num),
                            parent='world',
                            shape=ry.ST.ssBox,
                            pos=obs.pos,
                            size=obs.shape + [0],
                            color=[0.8, 0.8, 0.8])
            elif obs.type == "sphere":
                C.addObject(name="obs" + str(obs_num),
                            parent='world',
                            shape=ry.ST.sphere,
                            pos=obs.pos,
                            size=obs.shape,
                            color=[0.8, 0.8, 0.8])
            elif obs.type == "cylinder":
                C.addObject(name="obs" + str(obs_num),
                            parent='world',
                            shape=ry.ST.cylinder,
                            pos=obs.pos,
                            size=[obs.shape[1],obs.shape[0]],
                            color=[0.8, 0.8, 0.8])
            C.frame("obs" + str(obs_num)).setContact(True)
            obs_num += 1

    # add forces as optimization variables
    # timepP/timeStepspP works only if there is no time optimization komo.tau that has to be made accessible in python
    for m in range(robot.nrMotors):

        if np.mod(m+1,2) == 1:
            C.addForce("world", "m" + str(m+1), -max_force_per_motor * (timepP / timeStepspP), 0., -thrust_to_torque)

        else:
            C.addForce("world", "m" + str(m+1), -max_force_per_motor * (timepP / timeStepspP), 0., thrust_to_torque)

    #print("input + state dims: ", C.getJointDimension())

    komo.setModel(C, True)

    # avoid collisions
    komo.addObjective(times=[], feature=ry.FS.accumulatedCollisions, frames=[], type=ry.OT.eq, scale=[1e1])

    # avoid extreme quaternion normalization
    komo.addSquaredQuaternionNorms([], 1)

    # reach target position in the end
    komo.addObjective(times=[1.], feature=ry.FS.positionDiff, frames=['drone', 'target'], type=ry.OT.eq,
                      scale=[1e2])

    # reach intermediate states
    if intermediate_states is not None:
        for i_s in intermediate_states:
            if "pos" in i_s.type:
                komo.addObjective(times=[(i_s.timing+1)/timeStepspP], feature=ry.FS.position, frames=['drone'], type=ry.OT.eq,
                                  scale=[1e2], target=i_s.value[:3])

            if robot.nrMotors != 2 and robot.nrMotors != 3:
                if "quat" in i_s.type:
                    komo.addObjective(times=[(i_s.timing+1)/timeStepspP], feature=ry.FS.quaternion, frames=['drone'], type=ry.OT.eq,
                                      scale=[1e2], target=i_s.value[6:10])
            
            if "vel" in i_s.type:
                komo.addObjective(times=[(i_s.timing+1)/timeStepspP], feature=ry.FS.position, frames=['drone'], type=ry.OT.eq, order=1,
                                      scale=[1e2, 1e2, 1e2], target=i_s.value[3:6])


    if robot.nrMotors != 2 and robot.nrMotors != 3:

        # reach desired orientation
        komo.addObjective(times=[1.], feature=ry.FS.quaternion, frames=['drone'], type=ry.OT.eq,
                          scale=[1e2], target=xf[6:10])

        # final velocity at the end
        komo.addObjective(times=[1.], feature=ry.FS.qItself, frames=["drone"], type=ry.OT.eq, scale=[1e2],
                          target=[], order=1)

    else:
        print("correct const")
        # final velocity at the end
        komo.addObjective(times=[1.], feature=ry.FS.position, frames=['drone'], type=ry.OT.eq, target=xf[3:6], order=1,scale=[1e1, 1e1, 1e1])

    # initial pose at the beginning (has also to be set to t=1 since the newton euler equations can only be defined from t=2)
    komo.addObjective(times=[0.], feature=ry.FS.pose, frames=["drone"], type=ry.OT.eq, scale=[1e2],
                      target=list(x0[:3])+list(x0[6:10]), order=0, deltaFromStep=+0, deltaToStep=+1)

    # initial velocity at the beginning
    komo.addObjective(times=[0.], feature=ry.FS.position, frames=['drone'], type=ry.OT.eq, target=x0[3:6], order=1,
                      scale=[1e2, 1e2, 1e2], deltaFromStep=+0, deltaToStep=+1)

    komo.addObjective(times=[0.], feature=ry.FS.angularVel, frames=['drone'], type=ry.OT.eq, target=x0[10:], order=1,
                      scale=[1e2, 1e2, 1e2], deltaFromStep=+0, deltaToStep=+1)

    #komo.addObjective(times=[0.], feature=ry.FS.pose, frames=["drone"], type=ry.OT.eq, scale=[1e2],
    #                  target=[], order=1, deltaFromStep=+0, deltaToStep=+1)

    # follow dynamic equations
    komo.addObjective(times=[], feature=ry.FS.NewtonEuler, frames=["drone"], type=ry.OT.eq, scale=[alg_par.weight_dynamics], target=[],
                      order=2, deltaFromStep=+2, deltaToStep=-1)

    # add costs on input
    for m in range(robot.nrMotors):
        komo.addObjective(times=[], feature=ry.FS.fex_Force, frames=['world', "m" + str(m+1)], type=ry.OT.sos,
                          scale=[alg_par.weight_input], target=[], order=0, deltaFromStep=+0, deltaToStep=+0)

        #komo.addObjective(times=[], feature=ry.FS.fex_Force, frames=['world', "m" + str(m+1)], type=ry.OT.sos,
        #                  scale=[1e1], target=[], order=1, deltaFromStep=+0, deltaToStep=+0)

        #komo.addObjective(times=[], feature=ry.FS.fex_Force, frames=['world', "m" + str(m+1)], type=ry.OT.sos,
        #                  scale=[1e-1], target=[], order=2, deltaFromStep=+0, deltaToStep=+0)
                          
    # add limits to forces which where defined when the forces where added
    komo.addObjective(times=[], feature=ry.FS.qLimits, frames=['world'], type=ry.OT.ineq,
                      scale=[1e1])

    # dont leave the flyable volume
    komo.addObjective(times=[], feature=ry.FS.position, frames=['drone'], type=ry.OT.ineq, target=robot.max_x[:3],
                      scale=[1, 1, 1])

    komo.addObjective(times=[], feature=ry.FS.position, frames=['drone'], type=ry.OT.ineq, target=robot.min_x[:3],
                      scale=[-1, -1, -1])

    # dont exceed orientation limits
    komo.addObjective(times=[], feature=ry.FS.quaternion, frames=['drone'], type=ry.OT.ineq, target=robot.max_x[6:10],
                      scale=[1, 1, 1, 1])

    komo.addObjective(times=[], feature=ry.FS.quaternion, frames=['drone'], type=ry.OT.ineq, target=robot.min_x[6:10],
                      scale=[-1, -1, -1, -1])

    # dont exceed rotational and linear velocities
    komo.addObjective(times=[], feature=ry.FS.position, frames=['drone'], type=ry.OT.ineq, target=robot.max_x[3:6],
                      scale=[1, 1, 1], order=1)

    komo.addObjective(times=[], feature=ry.FS.position, frames=['drone'], type=ry.OT.ineq, target=robot.min_x[3:6],
                      scale=[-1, -1, -1], order=1)

    if robot.nrMotors != 2 and robot.nrMotors != 3:
        komo.addObjective(times=[], feature=ry.FS.angularVel, frames=['drone'], type=ry.OT.ineq, target=robot.max_x[10:],
                          scale=[1, 1, 1])

        komo.addObjective(times=[], feature=ry.FS.angularVel, frames=['drone'], type=ry.OT.ineq, target=robot.min_x[10:],
                          scale=[-1, -1, -1])

    way_points = []
    for t in range(timeStepspP):
        way_points.append(list(initial_x[t, :3]) + list(initial_x[t, 6:10]) + list(initial_u[t]))

    komo.initWithWaypoints(way_points, timeStepspP, True)

    return C, komo
