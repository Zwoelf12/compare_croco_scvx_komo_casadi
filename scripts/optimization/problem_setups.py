import numpy as np
from optimization import opt_utils as ou

class intermediate_state():

    def __init__(self, type=None, value=None, timing=None):
        self.type = type # list of which part of the states should be equal (pos, vel, ...)
        self.value = value # actual value of the state part
        self.timing = timing # intermediate time in time step number (e.g. at step 50 of 100)

class Prob_setup():

    def __init__(self, prob_nr):
    
        self.name = None

        self.x0 = None
        self.xf = None
        self.intermediate_states = None

        self.tf_min = None
        self.tf_max = None

        self.t_steps_scvx = None
        self.t_steps_komo = None
        self.t_steps_croco = None
        self.t_steps_casadi = None

        self.obs = None

        self.t2w = 1.4

        self.noise = 0.01

        if prob_nr == 1:
            self.simple_flight()
        elif prob_nr == 2:
            self.obstacle_flight()
        elif prob_nr == 3:
            self.recovery_flight()
        elif prob_nr == 4:
            self.flip()
        else:
            print("unknown problem")


    def simple_flight(self):

        self.name = "simple_flight"

        self.x0 = np.array([0., 0., 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.],
                            dtype=np.float64)

        self.xf = np.array([0., 1., 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.], dtype=np.float64)

        self.tf_min = 2.1
        self.tf_max = 2.1

        self.t_steps_scvx = 30
        self.t_steps_komo = 30
        self.t_steps_croco = 30
        self.t_steps_casadi = 30

        self.noise = 0.01

        self.t2w = 1.4

    def obstacle_flight(self):

        self.name = "obstacle_flight"

        self.x0 = np.array([-0.1, -1.3, 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.],
                            dtype=np.float64) 

        self.xf = np.array([-0.1, 1.3, 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.], dtype=np.float64)

        self.tf_min = 2.7
        self.tf_max = 2.7

        self.t_steps_scvx = 100
        self.t_steps_komo = 100
        self.t_steps_croco = 100
        self.t_steps_casadi = 100

        
        obs1 = ou.Obstacle("sphere", [0.3], [-0.8, 0.4, 0.2], [1, 0, 0, 0])
        obs2 = ou.Obstacle("sphere", [0.3], [0., 0.6, 0.5], [1, 0, 0, 0])
        obs3 = ou.Obstacle("sphere", [0.3], [0.8, -0.3, 0.2], [1, 0, 0, 0])

        obs4 = ou.Obstacle("sphere", [0.3], [-0.5, 0.2, 1.1], [1, 0, 0, 0])
        obs5 = ou.Obstacle("sphere", [0.3], [0., -0.4, 1.], [1, 0, 0, 0])
        obs6 = ou.Obstacle("sphere", [0.3], [0.5, 1., 0.8], [1, 0, 0, 0])

        obs7 = ou.Obstacle("sphere", [0.3], [-0.8, -0.8, 1.1], [1, 0, 0, 0])
        obs8 = ou.Obstacle("sphere", [0.3], [0., 0.4, 1.5], [1, 0, 0, 0])
        obs9 = ou.Obstacle("sphere", [0.3], [0.8, 0., 1.8], [1, 0, 0, 0])

        #obs7 = ou.Obstacle("sphere", [0.3], [-0.3, -0.45, 0.8], [1, 0, 0, 0])

        self.obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9]

        self.noise = 0.05
        self.t2w = 1.4

    def recovery_flight(self):

        self.name = "recovery_flight"

        self.x0 = np.array([0., 0., 1.,
                            0., 0., 0.,
                            0.0436194, -0.9990482, 0., 0.,
                            0., 0., 0.],
                            dtype=np.float64)

        self.xf = np.array([0., 0.15 , 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.], dtype=np.float64)

        self.tf_min = 1.8
        self.tf_max = 1.8

        self.t_steps_scvx = 100
        self.t_steps_komo = 100
        self.t_steps_croco = 100
        self.t_steps_casadi = 100

        self.noise = 0.01

        self.t2w = 1.4


    def flip(self):

        self.name = "flip"

        self.x0 = np.array([0., 0, 1.,
                            0., 0., 0.,
                            1, 0, 0., 0.,
                            0., 0., 0.],
                            dtype=np.float64)

        self.xf = np.array([0., 1.4, 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.], dtype=np.float64)

        xm_1 = intermediate_state(["pos"],
                                np.array([0., 1., 1.5,
                                            0., 0., 1.6,
                                            1., 0., 0., 0.,
                                            0., 0., 0.], dtype=np.float64),
                                45)

        xm_2 = intermediate_state(["quat"],
                                np.array([0., 0.725, 1.5,
                                            0., -2, 0.,
                                            0., 1., 0., 0.,#.71, .71, 0., 0.,
                                            0., 0., 0.], dtype=np.float64),
                                50)

        xm_3 = intermediate_state(["pos"],
                                np.array([0., 0.5, 1.5,
                                            0., 0., -1.6,
                                            0., 1., 0., 0.,
                                            0., 0., 0.], dtype=np.float64),
                                55)

        self.intermediate_states = [xm_2]#[xm_1,xm_2,xm_3]

        self.t_steps_scvx = 100
        self.t_steps_komo = 100
        self.t_steps_croco = 100
        self.t_steps_casadi = 100

        self.tf_min = 2.7
        self.tf_max = 2.7

        self.noise = 0.01

        self.t2w = 1.4
    
    
    
