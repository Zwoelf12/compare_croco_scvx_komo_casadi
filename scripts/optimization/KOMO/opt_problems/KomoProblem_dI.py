import sys
sys.path += ['deps/rai']
import libry as ry
import numpy as np

def doubleIntegrator_flight(phases,
               timeStepspP,
               timepP,
               x0,
               xf,
               obstacles,
               robot,
               initial_x):

    C = ry.Config()

    # define setup
    C.addFile('./physics/multirotor_models/double_integrator.g')
    C.addObject(name='target',
                parent='world',
                shape=ry.ST.ssBox,
                pos=xf[:3],
                size=[.05, .05, .05, .0002],
                color=[0., 1., 0.])
    C.frame("target").setContact(False)

    QC1 = C.frame("quadCopter1")
    QC1.setPosition(x0[:3])

    # set up optimization problem
    komo = C.komo(phases, timeStepspP, timepP, 2, True)

    # add obstacles
    if obstacles is not None:
        obs_num = 1
        for obs in obstacles:
            if obs.type == "box":
                C.addObject(name="obs"+str(obs_num),
                            parent='world',
                            shape=ry.ST.ssBox,
                            pos=obs.pos,
                            size=obs.shape+[0],
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
            C.frame("obs"+str(obs_num)).setContact(True)
            obs_num += 1

    komo.setModel(C, True)
    #### set constraints ####

    # avoid collisions
    komo.addObjective(times=[], feature=ry.FS.accumulatedCollisions, frames=[], type=ry.OT.eq, scale=[2])

    # avoid extreme quaternion normalization
    komo.addSquaredQuaternionNorms([], 3.)

    # initial condition velocity
    komo.addObjective(times=[0], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.eq, order=1,
                      target=x0[3:6],
                      scale=[1, 1, 1])

    # reach target position in the end
    komo.addObjective(times=[1.], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.eq, target=xf[:3],
                      scale=[1e2,1e2,1e2])

    # target velocity at goal
    komo.addObjective(times=[1.], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.eq, order=1,
                      target=xf[3:6],
                      scale=[1e2, 1e2, 1e2])

    # obey vel and acc constraints
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, order=1,
                      target=robot.max_x[3:6], scale=[1, 1, 1])
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, order=1,
                      target=robot.min_x[3:6], scale=[-1, -1, -1])
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, order=2,
                      target=robot.max_u, scale=[1, 1, 1])
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, order=2,
                      target=robot.min_u, scale=[-1, -1, -1])

    # dont crash in to the ground or leave the flyable volume
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, target=robot.max_x[:3],
                      scale=[1, 1, 1])

    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.ineq, target=robot.min_x[:3],
                      scale=[-1, -1, -1])


    #### set cost ####
    komo.addObjective(times=[], feature=ry.FS.position, frames=['quadCopter1'], type=ry.OT.sos, order=2,
                      target=[0, 0, 0], scale=[1, 1, 1])


    ### initialize solution ###
    way_points = []
    for t in range(timeStepspP):
        way_points.append(list(initial_x[t,:3]) + [1, 0, 0, 0])

    komo.initWithWaypoints(way_points, timeStepspP, False)

    return C,komo
