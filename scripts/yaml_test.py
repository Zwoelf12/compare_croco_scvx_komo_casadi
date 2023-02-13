import yaml
import numpy as np
from optimization.opt_utils import gen_yaml_files
from physics.multirotor_models.multirotor_full_model import QuadrotorAutograd
from optimization import opt_utils as ou

r = QuadrotorAutograd(4,0.1)

obs1 = ou.Obstacle("sphere", [0.4], [0., -0.7, 1.], [1, 0, 0, 0])
obs2 = ou.Obstacle("sphere", [0.4], [1., 0., 1.], [1, 0, 0, 0])
obs3 = ou.Obstacle("sphere", [0.4], [-1., 0., 1.], [1, 0, 0, 0])
obs4 = ou.Obstacle("sphere", [0.4], [0., 0., 1.7], [1, 0, 0, 0])
obs5 = ou.Obstacle("sphere", [0.4], [0., 0., .3], [1, 0, 0, 0])
obs6 = ou.Obstacle("sphere", [0.4], [0., 0.7, 1.], [1, 0, 0, 0])
obs = [obs1, obs2, obs3, obs4, obs5, obs6]

gen_yaml_files(np.zeros([50,13]),np.zeros([50,4]),obs,np.zeros(13),np.zeros(13),r)

