import jax.numpy as np
from jax import lax


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))

# Quaternion routines adapted from rowan to use autograd


def qmultiply(q1, q2):
    return np.concatenate((
        np.array([q1[0] * q2[0] - np.sum(q1[1:4] * q2[1:4])]),
        q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))


def qconjugate(q):
    return np.concatenate((q[0:1], -q[1:4]))


def qrotate(q, v):
    quat_v = np.concatenate((np.array([0]), v))
    return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]


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
    return lax.cond(np.allclose(q[1:4], 0), q,
                    qexp_zero_norm, q, qexp_regular_norm)


def qintegrate(q, v, dt):
    quat_v = np.concatenate((np.array([0]), v * dt / 2))
    return qmultiply(qexp(quat_v), q)


def qnormalize(q):
    return q / np.linalg.norm(q)


class RobotSingleIntegratorN:

    margin = 0.

    def __init__(
            self,
            dim=2,
            u_min=-np.inf,
            u_max=np.inf,
            free_time=False,
            semi_implicit=True):

        self.action_desc = [f"v_i [m/s] dim{dim}"]
        self.dim = dim
        self.min_u = np.array([u_min for _ in range(self.dim)])
        self.max_u = np.array([u_max for _ in range(self.dim)])

        self.state_desc = [f"x_i [m] dim{dim}"]
        self.min_x = np.array([u_min for _ in range(self.dim)])
        self.max_x = np.array([u_max for _ in range(self.dim)])

        self.dt = 0.1
        self.is2D = dim == 2

        self.free_time = free_time
        self.semi_implicit = semi_implicit

    def valid_state(self, state):
        return (state >= self.min_x - self.margin).all() and \
            (state <= self.max_x + self.margin).all()

    def step(self, state, action):

        if self.free_time:
            v = action[:-1]
            dt_scaling = action[-1]
        else:
            v = action
            dt_scaling = 1.

        state_next = state + self.dt * dt_scaling * v
        return state_next


class RobotDoubleIntegratorN:

    margin = 0.

    def __init__(
            self,
            dim=2,
            u_min=-np.inf,
            u_max=np.inf,
            free_time=False,
            semi_implicit=True):

        self.action_desc = [f"a_i [m/s^2] dim:{dim}"]
        self.dim = dim
        self.min_u = np.array([u_min for _ in range(self.dim)])
        self.max_u = np.array([u_max for _ in range(self.dim)])

        self.state_desc = [f"x_i [m] v_i[m/s] dim:{dim}"]
        self.min_x = np.array([u_min for _ in range(2 * self.dim)])
        self.max_x = np.array([u_max for _ in range(2 * self.dim)])

        self.dt = 0.1
        self.is2D = dim == 2

        self.free_time = free_time
        self.semi_implicit = semi_implicit

    def valid_state(self, state):
        return (state >= self.min_x - self.margin).all() and \
            (state <= self.max_x + self.margin).all()

    def step(self, state, action):

        if self.free_time:
            a = action[:-1]
            dt_scaling = action[-1]
        else:
            a = action
            dt_scaling = 1.

        pos = state[:self.dim] + self.dt * dt_scaling * state[self.dim:]
        vel = state[self.dim:] + self.dt * dt_scaling * a
        return np.concatenate((pos, vel))


class RobotUnicycleFirstOrder:

    margin = 0.

    def __init__(
            self,
            v_min=-np.inf,
            v_max=np.inf,
            w_min=-np.inf,
            w_max=np.inf,
            free_time=False,
            semi_implicit=True,
            normalize=True):

        self.action_desc = ["v [m/s]", "w [rad/s]"]
        self.min_u = np.array([v_min, w_min])
        self.max_u = np.array([v_max, w_max])

        self.state_desc = ["x [m]", "y [m]", "yaw [rad]"]
        self.min_x = np.array([-np.inf, -np.inf, -np.pi])
        self.max_x = np.array([np.inf, np.inf, np.pi])

        self.dt = 0.1
        self.is2D = True

        self.free_time = free_time
        self.semi_implicit = semi_implicit
        self.normalize = normalize

    def valid_state(self, state):
        return (state >= self.min_x - self.margin).all() and \
            (state <= self.max_x + self.margin).all()

    def step(self, state, action):
        x, y, yaw = state

        if self.free_time:
            v, w, dt_scaling = action
        else:
            v, w = action
            dt_scaling = 1.

        yaw_next = yaw + w * self.dt * dt_scaling
        if self.normalize:
            yaw_next = (yaw_next + np.pi) % (2 * np.pi) - np.pi

        yaw_eval = yaw_next if self.semi_implicit else yaw
        x_next = x + v * np.cos(yaw_eval) * self.dt * dt_scaling
        y_next = y + v * np.sin(yaw_eval) * self.dt * dt_scaling

        state_next = np.array([x_next, y_next, yaw_next])
        return state_next

    # def step(self, state, action):
    #     x, y, yaw = state
    #     v, w = action

    #     yaw_next = yaw + w * self.dt
    #     x_next = x + v * np.cos(yaw) * self.dt
    #     y_next = y + v * np.sin(yaw) * self.dt
    #     # normalize yaw between -pi and pi

    #     state_next = np.array([x_next, y_next, yaw_next])
    #     return state_next


class RobotUnicycleSecondOrder:

    margin = 0.

    def __init__(
            self,
            v_limit=np.inf,
            w_limit=np.inf,
            a_limit=np.inf,
            w_dot_limit=np.inf,
            free_time=False,
            semi_implicit=True,
            normalize=True):
        self.action_desc = ["a [m^2/s]", "w_dot [rad^2/s]"]
        self.min_u = np.array([-a_limit, -w_dot_limit])
        self.max_u = np.array([a_limit, w_dot_limit])

        self.state_desc = [
            "x [m]",
            "y [m]",
            "yaw [rad]",
            "v [m/s]",
            "w [rad/s]"]
        self.min_x = np.array([-np.inf, -np.inf, -np.pi, -v_limit, -w_limit])
        self.max_x = np.array([np.inf, np.inf, np.pi, v_limit, w_limit])

        self.dt = 0.1
        self.is2D = True

        self.free_time = free_time
        self.semi_implicit = semi_implicit
        self.normalize = normalize

    def valid_state(self, state):
        return (state >= self.min_x - self.margin).all() and \
            (state <= self.max_x + self.margin).all()

    def step(self, state, action):

        x, y, yaw, v, w = state

        if self.free_time:
            a, w_dot, dt_scaling = action
        else:
            a, w_dot = action
            dt_scaling = 1.

        dt = self.dt * dt_scaling

        v_next = v + a * dt
        w_next = w + w_dot * dt
        w_eval = w_next if self.semi_implicit else w
        yaw_next = yaw + w_eval * dt
        if self.normalize:
            yaw_next = (yaw_next + np.pi) % (2 * np.pi) - np.pi
        yaw_eval = yaw_next if self.semi_implicit else yaw
        v_eval = v_next if self.semi_implicit else v
        x_next = x + v_eval * np.cos(yaw_eval) * dt
        y_next = y + v_eval * np.sin(yaw_eval) * dt

        state_next = np.array([x_next, y_next, yaw_next, v_next, w_next])
        return state_next


# LaValle book, Equation 13.19
class RobotCarFirstOrderWithTrailers:

    def __init__(self,
                 v_min=-np.inf,
                 v_max=np.inf,
                 phi_min=-np.pi,
                 phi_max=np.pi,
                 L=.25,
                 hitch_lengths=[.5],
                 free_time=False,
                 semi_implicit=True,
                 normalize=True):

        self.action_desc = ["v [m/s]", "steering angle [rad]"]
        self.min_u = np.array([v_min, phi_min])
        self.max_u = np.array([v_max, phi_max])

        self.state_desc = ["x [m]", "y [m]", "yaw [rad]"]
        min_x = [-np.inf, -np.inf, -np.pi]
        max_x = [np.inf, np.inf, np.pi]
        for k in range(len(hitch_lengths)):
            self.state_desc.append("trailer{} yaw [rad]".format(k + 1))
            min_x.append(-np.pi)
            max_x.append(np.pi)
        self.min_x = np.array(min_x)
        self.max_x = np.array(max_x)

        self.L = L
        self.hitch_lengths = hitch_lengths
        self.dt = 0.1
        self.is2D = True

        self.free_time = free_time
        self.semi_implicit = semi_implicit
        self.normalize = normalize

    def valid_state(self, state):
        # check if theta0 and theta1 have a reasonable relative angle
        tol = 1e-3
        dangle = diff_angle(state[2], state[3])
        print(state, self.min_x, self.max_x)
        print(np.absolute(dangle))
        return (state >= self.min_x - tol).all() and \
            (state <= self.max_x + tol).all() and \
            np.absolute(dangle) <= np.pi / 4 + tol

    def step(self, state, action):
        """"
        x_dot = v * cos (theta_0)
        y_dot = v * sin (theta_0)
        theta_0_dot = v / L * tan(phi)
        theta_1_dot = v / hitch_lengths[0] * sin(theta_0 - theta_1)
        ...
        """
        x, y, yaw = state[:3]

        if self.free_time:
            v, phi, dt_scaling = action
        else:
            v, phi = action
            dt_scaling = 1.0

        dt = self.dt * dt_scaling

        yaw_next = yaw + v / self.L * np.tan(phi) * dt
        # normalize yaw between -pi and pi
        if self.normalize:
            yaw_next = normalize_angle(yaw_next)
        yaw_eval = yaw_next if self.semi_implicit else yaw
        x_next = x + v * np.cos(yaw_eval) * dt
        y_next = y + v * np.sin(yaw_eval) * dt

        state_next_list = [x_next, y_next, yaw_next]

        for i, d in enumerate(self.hitch_lengths):
            theta_dot = v / d
            for j in range(0, i):
                theta_dot *= np.cos(state[2 + j] - state[3 + j])
            theta_dot *= np.sin(state[2 + i] - state[3 + i])

            theta = state[3 + i]
            theta_next = theta + theta_dot * dt
            if self.normalize:
                theta_next = normalize_angle(theta_next)
            state_next_list.append(theta_next)

        state_next = np.array(state_next_list)
        return state_next


class Quadrotor:

    def __init__(self, free_time=False, semi_implicit=True, normalize=True):
        self.action_desc = ["f1 [N]", "f2 [N]", "f3 [N]", "f4 [N]"]
        self.min_u = np.zeros(4)
        self.max_u = np.ones(4) * 12.0 / 1000.0 * 9.81
        max_v = 2
        max_omega = 4

        self.state_desc = [
            "x [m]", "y [m]", "z [m]",
            "qx", "qy", "qz", "qw",
            "vx [m/s]", "vy [m/s]", "vz [m/s]",
            "w_x", "w_y", "w_z"]
        min_x = [-np.inf, -np.inf, -np.inf,
                 -1.001, -1.001, -1.001, -1.001,
                 -max_v, -max_v, -max_v,
                 -max_omega, -max_omega, -max_omega]
        max_x = [np.inf, np.inf, np.inf,
                 1.001, 1.001, 1.001, 1.001,
                 max_v, max_v, max_v,
                 max_omega, max_omega, max_omega]
        self.min_x = np.array(min_x)
        self.max_x = np.array(max_x)

        # parameters (Crazyflie 2.0 quadrotor)
        self.mass = 0.034  # kg
        # self.J = np.array([
        #   [16.56,0.83,0.71],
        #   [0.83,16.66,1.8],
        #   [0.72,1.8,29.26]
        #   ]) * 1e-6  # kg m^2
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # Note: we assume here that our control is forces
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.g = 9.81  # not signed

        if self.J.shape == (3, 3):
            # full matrix -> pseudo inverse
            self.inv_J = np.linalg.pinv(self.J)
        else:
            self.inv_J = 1 / self.J  # diagonal matrix -> division

        self.dt = 0.01
        self.is2D = False

        self.free_time = free_time
        self.semi_implicit = semi_implicit
        self.normalize = normalize

    def valid_state(self, state):
        out = (state >= self.min_x).all() and \
            (state <= self.max_x).all()
        if not out:
            print(state >= self.min_x)
            print(state <= self.max_x)
        return out

    def step(self, state, _action):
        # compute next state
        state = np.asarray(state)

        if self.free_time:
            action = _action[:-1]
            dt_scaling = _action[-1]
        else:
            action = _action
            dt_scaling = 1.

        dt = self.dt * dt_scaling

        action = np.asarray(action)
        q = np.concatenate((state[6:7], state[3:6]))
        q = qnormalize(q)
        omega = state[10:]

        eta = np.dot(self.B0, action)

        f_u = np.array([0, 0, eta[0]])
        tau_u = np.array([eta[1], eta[2], eta[3]])

        # dynamics
        # dot{p} = v
        pos_next = state[0:3] + state[7:10] * dt
        # mv = mg + R f_u
        vel_next = state[7:10] + (np.array([0, 0, -self.g]) +
                                  qrotate(q, f_u) / self.mass) * dt

        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
        # https://arxiv.org/pdf/1604.08139.pdf
        q_next = qnormalize(qintegrate(q, omega, dt))

        # mJ = Jw x w + tau_u
        omega_next = state[10:] + (self.inv_J * \
                                   (np.cross(self.J * omega, omega) + tau_u)) * dt

        return np.concatenate(
            (pos_next, q_next[1:4], q_next[0:1], vel_next, omega_next))


def create_robot(robot_type):
    if robot_type == "unicycle_first_order_0":
        return RobotUnicycleFirstOrder(-0.5, 0.5, -0.5, 0.5)
    elif robot_type == "unicycle_first_order_1":
        return RobotUnicycleFirstOrder(0.25, 0.5, -0.5, 0.5)
    elif robot_type == "unicycle_first_order_2":
        return RobotUnicycleFirstOrder(0.25, 0.5, -0.25, 0.5)
    elif robot_type == "unicycle_second_order_0":
        return RobotUnicycleSecondOrder(0.5, 0.5, 0.25, 0.25)
    elif robot_type == "car_first_order_0":
        return RobotCarFirstOrderWithTrailers(-0.1,
                                              0.5, -np.pi / 3, np.pi / 3, 0.25, [])
    elif robot_type == "car_first_order_with_1_trailers_0":
        return RobotCarFirstOrderWithTrailers(-0.1,
                                              0.5, -np.pi / 3, np.pi / 3, 0.25, [0.5])
    elif robot_type == "quadrotor_0":
        return Quadrotor()
    else:
        raise Exception("Unknown robot type {}!".format(robot_type))


if __name__ == '__main__':
    r = create_robot("quadrotor_0")

    # state = np.array([1, 1, 0.999959, -0.00603708, 0.00740579, -
    #                  0.000374226, 0.999954, 0, 0, -0.00819401, -2.41403, 2.96305, - 0.149642])
    # action = np.array([0.0755318, 0.0915615, 0.0796134, 0.0853427])

    eta = np.dot(r.B0, np.array([r.max_u[0], 0, r.max_u[0], 0]))

    f_u = np.array([0, 0, eta[0]])
    tau_u = np.array([eta[1], eta[2], eta[3]])

    print(f_u, tau_u)

# ==========================
# Compound state [
# Compound state [
# RealVectorState [1 1 0.999959]
# SO3State [-0.00603708 0.00740579 -0.000374226 0.999954]
# ]
# RealVectorState [0 0 -0.00819401]
# RealVectorState [-2.41403 2.96305 -0.149642]
# ]
# RealVectorControl [0.0755318 0.0915615 0.0796134 0.0853427]
#  apply for 0.01
# Compound state [
# Compound state [
# RealVectorState [1 1 0.999877]
# SO3State [-0.0181048 0.0222181 -0.00112223 0.999589]
# ]
# RealVectorState [0.0014469 0.00117859 -0.00865024]
# RealVectorState [-2.45169 3.16259 -0.10482]
# ]

    # import matplotlib.pyplot as plt

    # position = np.arange(-1.2, 0.6, 0.05)
    # f = - 0.0025 * np.cos(3 * position)
    # print(np.min(f), np.max(f))
    # fig, ax = plt.subplots()
    # ax.plot(position, f)
    # plt.show()

    # x = np.array([-0.5, 0.0])

    # robot = ContinuousMountainCarAutograd()
    # u = [robot.max_u]

    # states = [x]
    # for _ in range(100):
    #   x = robot.step(x, u)
    #   states.append(x)

    # states = np.array(states)

    # fig, axs = plt.subplots(3, 1, sharex='all')
    # axs[0].plot(states[:,0])
    # axs[0].set_ylabel("position [m]")
    # axs[1].plot(states[:,1])
    # axs[1].set_ylabel("velocity [m/s]")

    # ext_force = 0.0025 * np.cos(3 * states[:,0])
    # axs[2].plot(ext_force)
    # axs[2].set_ylabel("external force [N]")
    # plt.show()
