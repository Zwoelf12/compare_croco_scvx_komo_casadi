import jax.numpy as np # differentiate native python
# import numpy as np
from jax import lax # additional operations in python
# import rowan

# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
	return np.concatenate((
		np.array([q1[0] * q2[0] - np.sum(q1[1:4]*q2[1:4])]),
		q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
	return np.concatenate((q[0:1],-q[1:4]))

def qrotate(q, v):
	if v.shape[0] == 3:
		quat_v = np.concatenate((np.array([0]), v))
		return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]
	else:
		quat_v = v
		return qmultiply(q, qmultiply(quat_v, qconjugate(q)))

def qexp_regular_norm(q):
	e = np.exp(q[0])
	norm = np.linalg.norm(q[1:4])
	result_v = e * q[1:4] / norm * np.sin(norm)
	result_w = e * np.cos(norm)
	return np.concatenate((np.array([result_w]), result_v))

def qexp_zero_norm(q):
	e = np.exp(q[0])
	result_v = np.zeros(3)
	result_w = e
	return np.concatenate((np.array([result_w]), result_v))

def qexp(q):
	return lax.cond(np.allclose(q[1:4], 0), q, qexp_zero_norm, q, qexp_regular_norm)

def qintegrate(q, v, dt):
	quat_v = np.concatenate((np.array([0]), v*dt/2))
	return qmultiply(qexp(quat_v), q) # uses the rotational velocity in the inertial frame to calculate the new q

def qnormalize(q):
	return q / np.linalg.norm(q)

def qinverse(q):
	return np.concatenate((q[0:1], -q[1:4]))*(1/(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]))

def calc_rot_vel(q_t,q_tm1,dt):
	#return 2*(qmultiply((q_t - q_tm1) / dt, qinverse(q_t)))[1:]  # from quaternion derivative solved for omega: w=2*(q(t)-q(t-1))*q⁻¹
	q_dot = (q_t - q_tm1) / dt
	return 2 * (qmultiply(qinverse(q_tm1), q_dot))[1:]

class Multicopter():

	def __init__(self, nrMotors, arm_length, t2w = 1.4):
		self.type = "fM"
		self.nrMotors = nrMotors
		self.t2w = t2w
		self.dt = None
		self.t_dil_croco = None
		self.t_dil_casadi = None
		self.g = 9.81  # not signed

		self.min_x = np.array([-2.5, -2.5, 0,
							  -3, -3, -3,
							  -1.001, -1.001, -1.001, -1.001,
							   -35*0.7, -35*0.7, -35*0.7],dtype=np.float32)

		self.max_x = np.array([2.5, 2.5, 3,
							   3, 3, 3,
							    1.001, 1.001, 1.001, 1.001,
							   35*0.7, 35*0.7, 35*0.7], dtype=np.float32)

		# parameters (Crazyflie 2.0 quadrotor)
		self.mass = 0.034

		self.min_u = np.ones(nrMotors) * 0
		self.max_u = np.ones(nrMotors) * (self.t2w * self.mass * self.g)/self.nrMotors

		self.arm_length = arm_length

		# Note: we assume here that our control is forces
		self.t2t = 0.006 # 0.006 # thrust-to-torque ratio

		if nrMotors == 2:
			# describes the lever of the force to the axes
			arm_x = [-np.sin(np.pi / 4) * arm_length, np.sin(np.pi / 4) * arm_length]
			arm_y = [-np.cos(np.pi / 4) * arm_length, np.cos(np.pi / 4) * arm_length]
			self.B0 = np.array([
				[1, 1],
				[arm_x[0], arm_x[1]],
				[arm_y[0], arm_y[1]],
				[-self.t2t, self.t2t]
			])

		elif nrMotors == 3:
			# describes the lever of the force to the axes
			arm_x = [-arm_length, np.sin(np.pi/6)*arm_length, np.sin(np.pi/6)*arm_length]
			arm_y = [0, np.cos(np.pi/6)*arm_length, -np.cos(np.pi/6)*arm_length]
			self.B0 = np.array([
				[1, 1, 1],
				[arm_x[0], arm_x[1], arm_x[2]],
				[arm_y[0], arm_y[1], arm_y[2]],
				[-self.t2t, self.t2t, -self.t2t]
			])

		elif nrMotors == 4:
			# describes the lever of the force to the axes
			arm_x = [-np.sin(np.pi / 4) * arm_length, -np.sin(np.pi / 4) * arm_length, np.sin(np.pi / 4) * arm_length, np.sin(np.pi / 4) * arm_length]
			arm_y = [-np.cos(np.pi / 4) * arm_length, np.cos(np.pi / 4) * arm_length, np.cos(np.pi / 4) * arm_length, -np.cos(np.pi / 4) * arm_length]
			self.B0 = np.array([
				[1, 1, 1, 1],
				[arm_x[0], arm_x[1], arm_x[2], arm_x[3]],
				[arm_y[0], arm_y[1], arm_y[2], arm_y[3]],
				[-self.t2t, self.t2t, -self.t2t, self.t2t]
				])

		elif nrMotors == 6:
			# describes the lever of the force to the axes
			arm_x = [-np.sin(np.pi / 3) * arm_length, -np.sin(np.pi / 3) * arm_length, 0, np.sin(np.pi / 3) * arm_length, np.sin(np.pi / 3) * arm_length, 0]
			arm_y = [-np.cos(np.pi / 3) * arm_length, +np.cos(np.pi / 3) * arm_length, arm_length, np.cos(np.pi / 3) * arm_length, -np.cos(np.pi / 3) * arm_length, -arm_length]

			self.B0 = np.array([
				[1, 1, 1, 1, 1, 1],
				[arm_x[0], arm_x[1], arm_x[2], arm_x[3], arm_x[4], arm_x[5]],
				[arm_y[0], arm_y[1], arm_y[2], arm_y[3], arm_y[4], arm_y[5]],
				[-self.t2t, self.t2t, -self.t2t, self.t2t, -self.t2t, self.t2t]
			])

		elif nrMotors == 8:
			# describes the lever of the force to the axes
			arm_x = [-np.sin(np.pi / 4) * arm_length, -arm_length, -np.sin(np.pi / 4) * arm_length, 0,np.sin(np.pi / 4) * arm_length, arm_length, np.sin(np.pi / 4) * arm_length, 0]
			arm_y = [-np.cos(np.pi / 4) * arm_length, 0, +np.cos(np.pi / 4) * arm_length, arm_length,np.cos(np.pi / 4) * arm_length, 0, -np.cos(np.pi / 4) * arm_length, -arm_length]

			self.B0 = np.array([
				[1, 1, 1, 1, 1, 1, 1, 1],
				[arm_x[0], arm_x[1], arm_x[2], arm_x[3], arm_x[4], arm_x[5], arm_x[6], arm_x[7]],
				[arm_y[0], arm_y[1], arm_y[2], arm_y[3], arm_y[4], arm_y[5], arm_y[6], arm_y[7]],
				[-self.t2t, self.t2t, -self.t2t, self.t2t, -self.t2t, self.t2t, -self.t2t, self.t2t]
			])

		# approximate Inertia of the quadcopter
		mi = (self.mass / nrMotors) * 0.45
		J_11 = 0 #(16.571710e-6 / m0) / (r0 ** 2)
		J_22 = 0 #(16.655602e-6 / m0) / (r0 ** 2)
		J_33 = 0 #(29.261652e-6 / m0) / (r0 ** 2)

		for m in range(nrMotors):
			J_11 += mi*(arm_x[m]**2)
			J_22 += mi*(arm_y[m]**2)
			J_33 += mi*(self.arm_length**2)

		self.J = np.array([J_11, J_22, J_33])

		if self.J.shape == (3, 3):
			self.inv_J = np.linalg.pinv(self.J) # full matrix -> pseudo inverse
		else:
			self.inv_J = 1 / self.J # diagonal matrix -> division

	# to integrate the dynamics, see
	# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
	# https://arxiv.org/pdf/1604.08139.pdf

	def step(self, state, force, t_dil):
		# compute next state
		pos = state[:3]
		vel = state[3:6]
		q = state[6:10]
		omega = state[10:]
		dt = self.dt * t_dil

		eta = np.dot(self.B0, force)
		f_u = np.array([0, 0, eta[0]])
		tau_u = np.array([eta[1], eta[2], eta[3]])

		## dynamics ##

		# velocity
		# v(t) = v(t-1) + (g + R(f(t))/m)*dt = v(t-1) + a(t)*dt
		acc = (np.array([0, 0, -self.g]) + qrotate(q, f_u) / self.mass)
		vel_next = vel + acc * dt

		# positions
		# x(t) = x(t-1) + v(t)*dt
		pos_next = pos + vel * dt

		# rotational velocity
		# w(t) = w(t-1) + J⁻¹(Tau(t-1) - w(t-1)xJw(t-1))*dt = w(t-1) + w_dot(t-1)
		omega_dot = (self.inv_J * (tau_u - np.cross(omega, self.J * omega)))
		omega_next = omega + omega_dot * dt

		# quaternions
		# q(t) = q(t-1) + 0.5*w(t)q(t-1)
		q_next = qnormalize(qintegrate(q, qrotate(q, omega), dt))

		return np.concatenate((pos_next, vel_next, q_next, omega_next))

	def step_KOMO(self, state_tm1, state_t, force, t_dil):
		# compute next state
		q_t = state_t[6:10] # rotation of the body frame in the inertial frame
		omega_t = state_t[10:] # rotational velocity in the inertial frame
		pos_t = state_t[:3] # position in the inertial frame
		vel_t = state_t[3:6] # velocity in the inertial frame

		q_tm1 = state_tm1[6:10]
		omega_tm1 = state_tm1[10:]
		pos_tm1 = state_tm1[:3]
		vel_tm1 = state_tm1[3:6]

		dt = self.dt*t_dil

		eta = np.dot(self.B0, force)
		f_u = np.array([0, 0, eta[0]])
		tau_u = np.array([eta[1], eta[2], eta[3]])

		## dynamics ##
		# positions
		# x(t) = x(t-1) + v(t)*dt
		pos_next = pos_tm1 + vel_t * dt

		# lin velocities
		# v(t) = v(t-1) + (g + q(t)(f(t)/m)q⁻¹(t))*dt = v(t-1) + a(t)*dt
		acc_t_I = (np.array([0, 0, -self.g]) + qrotate(q_tm1, f_u) / self.mass)
		vel_next = vel_tm1 + acc_t_I * dt

		# quaternions
		# q_dot(t) = 0.5*q(t)*omega(t)
		q_dot_t = 0.5 * qmultiply(q_t, np.concatenate((np.array([0]), omega_t)))
		q_next = qnormalize(q_tm1 + q_dot_t * dt)

		# rotational velocities
		# w(t) = w(t-1) + J⁻¹(Tau(t) - w(t)xJw(t))*dt = w(t-1) + w_dot(t)*dt
		omega_dot_t = self.inv_J * (tau_u - np.cross(omega_tm1,self.J * omega_tm1))
		#omega_dot_t = self.inv_J * tau_u
		omega_next = omega_tm1 + omega_dot_t * dt

		return np.concatenate((pos_next, vel_next, q_next, omega_next))