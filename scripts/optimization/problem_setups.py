import numpy as np
from optimization import opt_utils as ou

class intermediate_state():

    def __init__(self, type=None, value=None, timing=None):
        self.type = type # list of which part of the states should be equal (pos, vel, ...)
        self.value = value # actual value of the state part
        self.timing = timing # intermediate time in time step number (e.g. at step 50 of 100)

class Prob_setup():

    x0 = None
    xf = None
    intermediate_states = None

    tf_min = None
    tf_max = None

    t_steps_scvx = None
    t_steps_komo = None
    t_steps_croco = None
    t_steps_casadi = None


    obs = None

    t2w = 1.4

    noise = 0.01

def simple_flight_wo_obs():

    setup = Prob_setup()

    setup.x0 = np.array([0., 0., 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.],
                         dtype=np.float64)

    setup.xf = np.array([0., 1., 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.], dtype=np.float64)

    setup.tf_min = 2.1
    setup.tf_max = 2.1

    setup.t_steps_scvx = 30
    setup.t_steps_komo = 30
    setup.t_steps_croco = 30
    setup.t_steps_casadi = 30

    setup.noise = 0.01

    setup.t2w = 1.4

    return setup

def complex_flight_spheres():

    setup = Prob_setup()

    setup.x0 = np.array([0.1, -1.3, 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.],
                         dtype=np.float64)  # from -0.5 to 0.5 and 0.5 hight diff works for full model # 0.71,0.71 for 90°

    setup.xf = np.array([-0.1, 1.3, 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.], dtype=np.float64)

    setup.tf_min = 2.7
    setup.tf_max = 2.7

    setup.t_steps_scvx = 30
    setup.t_steps_komo = 30
    setup.t_steps_croco = 30
    setup.t_steps_casadi = 30

    obs1 = ou.Obstacle("sphere", [0.4], [0., -0.7, 1.], [1, 0, 0, 0])
    obs2 = ou.Obstacle("sphere", [0.4], [1., 0., 1.], [1, 0, 0, 0])
    obs3 = ou.Obstacle("sphere", [0.4], [-1., 0., 1.], [1, 0, 0, 0])
    obs4 = ou.Obstacle("sphere", [0.4], [0., 0., 1.7], [1, 0, 0, 0])
    obs5 = ou.Obstacle("sphere", [0.4], [0., 0., .3], [1, 0, 0, 0])
    obs6 = ou.Obstacle("sphere", [0.4], [0., 0.7, 1.], [1, 0, 0, 0])
    setup.obs = [obs1, obs2, obs3, obs4, obs5, obs6]

    setup.noise = 0.05
    setup.t2w = 1.4

    return setup

def recovery_flight():

    setup = Prob_setup()

    setup.x0 = np.array([0., 0., 1.,
                         0., 0., 0.,
                         0.0436194, -0.9990482, 0., 0.,
                         0., 0., 0.],
                         dtype=np.float64)

    setup.xf = np.array([0., 0.15 , 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.], dtype=np.float64)

    setup.tf_min = 1.8
    setup.tf_max = 1.8

    setup.t_steps_scvx = 100
    setup.t_steps_komo = 100
    setup.t_steps_croco = 100
    setup.t_steps_casadi = 100

    setup.noise = 0.01

    setup.t2w = 1.4

    return setup


def flip():

    setup = Prob_setup()

    setup.x0 = np.array([0., 0., 1.,
                         0., 0., 0.,
                         1, 0, 0., 0.,
                         0., 0., 0.],
                         dtype=np.float64)

    setup.xf = np.array([0., 1.5, 1.,
                         0., 0., 0.,
                         1., 0., 0., 0.,
                         0., 0., 0.], dtype=np.float64)

    xm_1 = intermediate_state(["quat"],
                              np.array([0., 0.725, 1.,
                                        0., 0., 0.,
                                        0., 0., -1., 0.,
                                        0., 0., 0.], dtype=np.float64),
                              30)

    xm_2 = intermediate_state(["pos"],
                              np.array([0., 1., 1.5,
                                        0., 0., 0.,
                                        .71, 0., -.71, 0.,
                                        0., 0., 0.], dtype=np.float64),
                              20)

    xm_3 = intermediate_state(["pos"],
                              np.array([0., 0.5, 1.5,
                                        0., 0., 0.,
                                        .71, 0., .71, 0.,
                                        0., 0., 0.], dtype=np.float64),
                              40)

    setup.intermediate_states = [xm_1,xm_2,xm_3]

    setup.t_steps_scvx = 60
    setup.t_steps_komo = 60
    setup.t_steps_croco = 60
    setup.t_steps_casadi = 60

    setup.tf_min = 2.7
    setup.tf_max = 2.7

    setup.noise = 0.01

    setup.t2w = 1.4

    return setup
    
    
    
