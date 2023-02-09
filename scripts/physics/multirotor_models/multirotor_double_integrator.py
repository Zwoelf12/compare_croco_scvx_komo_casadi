import jax.numpy as np # differentiate native python
# import numpy as np
from jax import lax # additional operations in python

class QuadrotorDoubleInt():

	def __init__(self, arm_length, t2w = 1.4):
		self.arm_length = arm_length
		self.type = "dI"
		self.dt = None
		self.t2w = t2w

		self.min_u = -np.ones(3)*t2w*9.81
		self.max_u = np.ones(3)*t2w*9.81

		self.min_x = np.array([-1.5,-1.5,0,
							   -3,-3,-3],dtype=np.float64)

		self.max_x = np.array([1.5,1.5,2,
							   3,3,3],dtype=np.float64)

		self.g_vec = 9.81 * np.array([0, 0, 1])

	def step(self, state, actions, t_dil):
		# compute next state
		dt = self.dt * t_dil
		acc = actions  # - self.g_vec
		vel_next = state[3:6] + acc * dt
		pos_next = state[0:3] + state[3:6] * dt
		#pos_next = state[0:3] + vel_next * dt

		return np.concatenate((pos_next, vel_next))

	def step_KOMO(self, state_tm1, state_t, actions, t_dil):
		# compute next state
		dt = self.dt * t_dil
		acc = actions
		vel_next = state_tm1[3:6] + acc * dt
		pos_next = state_tm1[0:3] + vel_next * dt

		return np.concatenate((pos_next, vel_next))

