import jax.numpy as np # differentiate native python
# import numpy as np
from jax import lax # additional operations in python

class QuadrotorDoubleInt():

	def __init__(self, arm_length):
		self.arm_length = arm_length
		self.type = "dI"
		self.dt = None

		self.min_u = np.array([-20.09, -20.09, 0.3])
		self.max_u = np.array([20.09, 20.09, 23.2])

		self.min_sig = .6
		self.max_sig = 23.2

		self.min_x = np.array([-10, -10, -10,
							   -30, -30, -30], dtype=np.float32)

		self.max_x = np.array([10, 10, 10,
							   30, 30, 30], dtype=np.float32)

		self.g_vec = 9.81 * np.array([0, 0, 1])
		self.theta_max = np.pi/3 # 60°

	def step(self, state, actions, t_dil):
		# compute next state
		acc = actions - self.g_vec
		pos_next = state[0:3] + state[3:6] * self.dt * t_dil
		vel_next = state[3:6] + acc * self.dt * t_dil

		return np.concatenate((pos_next, vel_next))

	def step_implicit(self, state, actions):
		# compute next state
		acc = actions - self.g_vec
		vel_next = state[3:6] + acc * self.dt
		pos_next = state[0:3] + vel_next * self.dt

		return np.concatenate((pos_next, vel_next))

