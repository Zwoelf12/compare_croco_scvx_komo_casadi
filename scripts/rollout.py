from physics.multirotor_models.multirotor_full_model_komo_scp import Multicopter
import numpy as np
import yaml

robot = Multicopter(4,0.046, 1.4)
t_steps = 10
tf = 0.5
robot.dt = 1/t_steps

states = [np.array([0,0,0,
                   0,0,0,
                   1,0,0,0,
                   0,0,0])]

action = np.array([0.08,0.08,0.1,0.1])
acts = [np.array([0.08,0.08,0.1,0.1])]
for t in range(10):
    states.append(np.array(robot.step(states[t], action, tf)))
    acts.append(action)

setup = {'x0': states[0].tolist(), 't_steps': 10, 'dt': robot.dt*tf}
dat = {'states': np.array(states).tolist(),'actions': np.array(acts).tolist()}

print(dat)

with open("../scripts/rollout.yaml", mode="wt", encoding="utf-8") as file:
        yaml.dump(setup, file, default_flow_style=None, sort_keys=False)
        yaml.dump(dat, file, default_flow_style=None, sort_keys=False)

