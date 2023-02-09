import cvxpy as cp
import jax.numpy as jnp  # Thinly-wrapped numpy
import numpy as np
from jax import jacfwd, jit, config
config.update("jax_enable_x64", True)
import time
from optimization.opt_utils import OptSolution
import matplotlib.pyplot as plt
from visualization.geom_visualization import draw_obs
from visualization.visualize_paper_comparision_V2 import plot_comparision

class SCvx():
  def __init__(self, robot):
    self.robot = robot

    # user defined parameter
    self.lam = 30 # weight for slack in cost
    self.alp = 0  # weight for time in cost
    self.bet = 1 # weight for input in cost
    self.gam = 0.  # weight ratio between input and time penalty
    self.E = np.eye(self.robot.min_x.shape[0])
    # when to update trust region
    self.roh0 = 0.
    self.roh1 = 0.1
    self.roh2 = 0.7
    # growth and shrink parameter for trust region
    self.bgr = 2
    self.bsh = 2
    # initial trust region and minimal and maximal trust region radius
    self.eta = 1 #3 for dI -> has to be high for the full model to make firs iteration feasible
    self.eta0 = 1e-3
    self.eta1 = 10
    # stopping criterion
    self.eps = -0.1
    self.eps_r = -0.1
    # slack weight update factor
    self.lam_fac = 1.0 #5 # 1 # sometimes necessary to not get stuck when using cost on input
    # norm order for trust region
    self.q = "inf"
    # tolerance for nu before solver tries to solve the problem without slack
    if self.robot.type == "dI":
      self.tol = 1e-7
    elif self.robot.type == "fM":
      self.tol = 1e-7

    # build A & B Matrices and f
    self.constructA = jit(jacfwd(robot.step, 0))
    self.constructB = jit(jacfwd(robot.step, 1))
    self.constructF = jit(jacfwd(robot.step, 2))
    self.step = jit(robot.step)

    self.collAvoidance = "paper"

  def solve(self, x0, xf, tf_min, tf_max,
    T = 50,
    num_iterations = 10,
    CHandler=None):

    time_all_s = time.time()

    self.tf_min = tf_min
    self.tf_max = tf_max

    # variable scaling to make SCvx scale invariant
    # determine scaling matrices
    self.Su = jnp.diag(self.robot.max_u - self.robot.min_u)
    self.Ssig = self.robot.max_sig - self.robot.min_sig
    self.Sp = self.tf_max - self.tf_min
    # determine offset
    self.cu = self.robot.min_u
    self.csig = self.robot.min_sig
    self.cp = self.tf_min

    self.Sx = np.eye(6, 6)
    self.cx = np.zeros(6)

    stateDim = x0.shape[0]
    actionDim = self.robot.min_u.shape[0]
    if CHandler.obs_data:
      nObstacles = len(CHandler.obs_data)
    else:
      nObstacles = 0

    initial_guess = self.calc_initial_guess(x0, xf, T, stateDim, actionDim, nObstacles)

    xprev = initial_guess.x
    uprev = initial_guess.u
    nuprev = initial_guess.nu
    sigprev = initial_guess.sig
    nusprev = initial_guess.nus
    nuICprev = initial_guess.nuIC
    nuTCprev = initial_guess.nuTC
    pprev = initial_guess.p
    Pprev = initial_guess.P
    Pfprev = initial_guess.Pf
    dx_lqprev = initial_guess.dx_lq
    du_lqprev = initial_guess.du_lq
    dp_lqprev = initial_guess.dp_lq

    prevNLCost = None
    lin_cost = []
    nonlin_cost = []
    stop_flag = False
    for itr in range(num_iterations):

      start_time = time.time() # start timing

      x = cp.Variable((T, stateDim))
      u = cp.Variable((T, actionDim))
      nu = cp.Variable((T - 1, stateDim))
      if nObstacles == 0:
        nus = cp.Variable((T, 1))
      else:
        nus = cp.Variable((T, nObstacles))
      nuIC = cp.Variable(stateDim)
      nuTC = cp.Variable(stateDim)
      sig = cp.Variable(T)
      p = cp.Variable(1)
      P = cp.Variable(T)
      Pf = cp.Variable(2)
      dx_lq = cp.Variable(T)
      du_lq = cp.Variable(T)
      dp_lq = cp.Variable(1)

      # set initial guess for warm start
      x.value = xprev
      u.value = uprev
      sig.value = sigprev
      p.value = pprev
      nu.value = nuprev
      nus.value = nusprev
      nuIC.value = nuICprev
      nuTC.value = nuTCprev
      P.value = Pprev
      Pf.value = Pfprev
      dx_lq.value = dx_lqprev
      du_lq.value = du_lqprev
      dp_lq.value = dp_lqprev

      # scale problem
      x_scal, u_scal, sig_scal, p_scal = self.scale_prob(x, u, sig, p)
      x_prev_scal, u_prev_scal, sig_prev_scal, p_prev_scal = self.scale_prob(xprev, uprev, sigprev, pprev)

      if itr==0:
        X_, U_, SIG_, P_, = [x_prev_scal], \
                            [u_prev_scal], \
                            [sig_prev_scal], \
                            [p_prev_scal]

        NU_, NUS_, NUIC_, NUTC_ = [nuprev], \
                                  [nusprev], \
                                  [nuICprev], \
                                  [nuTCprev]

        ETA_, ROH_ = [self.eta], []

      # define objective
      cp_objective = self.define_objective(x_scal, u_scal, sig_scal, p_scal, nu, nus, nuIC, nuTC, P, Pf, T)

      # define constraints
      constraints = self.define_constraints(x_scal, u_scal, sig_scal, p_scal,
                                            nu, nus, nuIC, nuTC, P, Pf,
                                            dx_lq, du_lq, dp_lq,
                                            x_prev_scal, u_prev_scal, p_prev_scal,
                                            x0, xf, T, CHandler)

      # define problem
      prob = cp.Problem(cp_objective, constraints)

      # The optimal objective value is returned by `prob.solve()`.
      try:
        start_time_sol = time.time()
        result = prob.solve(verbose=False, warm_start=False, solver=cp.ECOS_BB, feastol=1e-15, abstol=1e-15, reltol=1e-15)  # , BarQCPConvTol=1e-9, reoptimize=True)
        end_time_sol = time.time()
      except cp.error.SolverError:
        print("Warning: Solver failed!")
        return X_, U_, nu.value, float('inf')
      except KeyError:
        print("Warning BarQCPConvTol too big?")
        return X_, U_, nu.value, float('inf')

      if prob.value == np.inf:
        print("infeasible problem")
        return X_, U_, nu.value, float('inf')

      # execute update policy
      if lin_cost:
        lin_cost_prev = lin_cost[-1]
      else:
        lin_cost_prev = None

      # stopping criterion
      if prevNLCost != None:
        absCrit = np.linalg.norm(p.value - pprev, 1) + np.max(np.linalg.norm(x.value - xprev, 1, 1))
        relCrit = np.abs(prevNLCost - prob.value) / np.abs(prevNLCost)
        if relCrit < self.eps_r or absCrit < self.eps:
          stop_flag = True

      xprev, uprev, sigprev, pprev, nuprev, nusprev, nuICprev, nuTCprev, Pprev, Pfprev, dx_lqprev, du_lqprev, dp_lqprev, nl_cost, l_cost, roh = self.update_prob(x.value, u.value, sig.value, p.value,
                                                                                                                                           nu.value, nus.value, nuIC.value, nuTC.value, P.value, Pf.value,
                                                                                                                                           dx_lq.value, du_lq.value, dp_lq.value,
                                                                                                                                           T, xf, x0, prob,
                                                                                                                                           xprev, uprev, sigprev, pprev,
                                                                                                                                           nuprev, nusprev, nuICprev, nuTCprev, Pprev, Pfprev,
                                                                                                                                           dx_lqprev, du_lqprev, dp_lqprev,
                                                                                                                                           CHandler,lin_cost_prev)

      x_scal, u_scal, sig_scal, p_scal = self.scale_prob(x.value, u.value, sig.value, p.value)

      X_.append(x_scal)
      U_.append(u_scal)
      SIG_.append(sig_scal)
      P_.append(p_scal)
      NU_.append(nu.value)
      NUS_.append(nus.value)
      NUIC_.append(nuIC.value)
      NUTC_.append(nuTC.value)
      lin_cost.append(l_cost)
      nonlin_cost.append(nl_cost)
      ETA_.append(self.eta)
      ROH_.append(roh)

      end_time = time.time()

      print("________________________________________")
      print("Iteration: ", itr + 1)
      print("nu max: ", np.max(np.abs(nu.value)))
      print("nus max: ", np.max(np.abs(nus.value)))
      print("nuIC max: ", np.max(np.abs(nuIC.value)))
      print("nuTC max: ", np.max(np.abs(nuTC.value)))
      print("trajectory is executed in: {} s".format(self.redim_p(p.value)))
      print("iteration time: ", np.round(end_time - start_time, 4))
      print("solver time: ", np.round(end_time_sol - start_time_sol, 4))
      print("________________________________________")

      # execute stopping crit
      if stop_flag == True:
        break

      if itr == num_iterations - 1:
        print("reached max num of iterations")

      # update previous optimal value
      prevNLCost = nl_cost

    time_all_e = time.time()

    plot_comparision(np.array(X_),np.array(U_),np.array(SIG_),np.array(P_),
                     np.array(NU_),np.array(NUS_),np.array(NUIC_),np.array(NUTC_),np.array(lin_cost),np.array(nonlin_cost),
                     np.array(ETA_), np.array(ROH_), T, num_iterations,stateDim,actionDim,x0,xf,CHandler)

    states = np.array(X_[-1])
    actions = np.vstack((np.array(U_[-1]), np.zeros(self.robot.min_u.shape[0])*np.nan))

    solution = OptSolution(states, actions, time_all_e - time_all_s, nu.value, prob.value, itr+1, p.value)
    return solution

###############################################################################################################

  def scale_prob(self, x, u, sig, p):

    # scale states and inputs
    x_scal = (self.Sx @ x.T + self.cx[:, np.newaxis]).T
    u_scal = (self.Su @ u.T + self.cu[:, np.newaxis]).T

    sig_scal = self.Ssig * sig + self.csig
    p_scal = self.Sp * p + self.cp

    return x_scal, u_scal, sig_scal, p_scal

  def redim_x(self,x):
    return self.Sx @ x + self.cx

  def redim_u(self,u):
    return self.Su @ u + self.cu

  def redim_sig(self,sig):
    return self.Ssig * sig + self.csig

  def redim_p(self,p):
    return self.Sp * p + self.cp

  def nondim_x(self,xp):
    return (np.linalg.inv(self.Sx) @ (xp - self.cx).T).T

  def nondim_u(self,up):
    return (np.linalg.inv(self.Su) @ (up - self.cu).T).T

  def nondim_sig(self,sigp):
    if self.Ssig == 0:
      return self.csig
    else:
      return (1/self.Ssig) * (sigp - self.csig)

  def nondim_p(self,pp):
    if self.Sp == 0:
      return self.cp
    else:
      return (1/self.Sp) * (pp - self.cp)

  def define_objective(self, x, u, sig, p, nu, nus, nuIC, nuTC, P, Pf, T):

    inputs = self.robot.dt*cp.sum((sig/9.81)**2) # penalty on inputs
    slack_cost = cp.sum(self.lam * self.robot.dt * P) + cp.sum(self.lam * Pf) # penalty for slack

    time = (p/self.tf_max)**2

    cp_objective = cp.Minimize(slack_cost + (1-self.gam)*inputs + self.gam*time)

    return cp_objective

  def define_constraints(self, x, u, sig, p,
                         nu, nus, nuIC, nuTC, P, Pf,
                         dx_lq, du_lq, dp_lq,
                         xprev, uprev, pprev,
                         x0, xf, T, CHandler):

    # initial and final constraints
    constraints = [
      x[0] == x0 + nuIC, # initial state constraint
      x[T-1] == xf + nuTC # final state constraint
    ]

    # add constraints on helper variables (p) for trust region calculation
    constraints.append(
      cp.norm(self.nondim_p(p) - self.nondim_p(pprev), self.q) <= dp_lq
    )

    for t in range(0, T):
      # constraints on time dilation
      constraints.extend([
        p >= self.tf_min,
        p <= self.tf_max
      ])

      # add constraints on helper variables for trust region calculation
      constraints.append(
        cp.norm(self.nondim_x(x[t]) - self.nondim_x(xprev[t]), self.q) <= dx_lq[t]
      )

      # add constraints on helper variables for trust region calculation
      constraints.append(
        cp.norm(self.nondim_u(u[t]) - self.nondim_u(uprev[t]), self.q) <= du_lq[t]
      )

      # trust region constraints
      constraints.append(
        dx_lq[t] + du_lq[t] + dp_lq <= self.eta
      )

      # bounds on sig
      constraints.extend([
        self.robot.min_sig <= sig[t],
        sig[t] <= self.robot.max_sig
      ])

      constraints.append(
        cp.norm(u[t], 2) <= sig[t]
      )

      constraints.append(
        sig[t] * np.cos(self.robot.theta_max) <= np.array([0, 0, 1]).T @ u[t]
      )

      if self.collAvoidance == "paper":
        H1 = np.diag([2,2,2])
        c1 = np.array([0,1,0.9])
        C1 = -(H1.T @ H1 @ (xprev[t,:3]-c1))/np.linalg.norm(H1 @ (xprev[t,:3]-c1), 2)
        s_prev1 = 1 - np.linalg.norm(H1 @ (xprev[t,:3]-c1))

        constraints.append(
          s_prev1 + C1 @ (x[t, :3] - xprev[t, :3]) <= nus[t, 0]
        )
      else:


        # Obstacle constraints
        if CHandler.obs:

          dist_tilde, pRob, pObs = CHandler.calcDistance(xprev[t])

          # only add constraints if the robot was near an Obstacle in the last iteration
          for i in range(len(dist_tilde)):

            if dist_tilde[i] > 0:
              dist = pRob[i] - pObs[i]
            else:
              dist = pObs[i] - pRob[i]

            norm_dist = np.linalg.norm(dist, 2)

            if norm_dist > 0:  # if distance is greater 0 add constraints
              dist_hat = dist / norm_dist

              constraints.append(
                dist_tilde[i] + dist_hat[:3].T @ (x[t, :3] - xprev[t, :3]) >= nus[t, i]
              )

            else:
              # if distance is 0 and direction information can not be obtained deflate the robot to obtain the direction
              arm_length = self.robot.arm_length
              CHandler.redim_robot(arm_length*0.9)

              dist_tilde_defl, pRob_defl, pObs_defl = CHandler.calcDistance(xprev[t - 1])
              # inflate robot to correct size
              CHandler.redim_robot(arm_length)

              if dist_tilde_defl[i] > 0:
                dist_defl = pRob_defl[i] - pObs_defl[i]
              else:
                dist_defl = pObs_defl[i] - pRob_defl[i]

              norm_dist_defl = np.linalg.norm(dist_defl)

              if norm_dist_defl > 0:
                dist_hat_defl = dist_defl / norm_dist_defl

                constraints.append(
                  dist_hat_defl[:3].T @ (x[t, :3] - xprev[t, :3]) >= nus[t, i]
                )

    for t in range(0, T - 1):
      # dynamic constraints
      xbar = xprev[t]
      ubar = uprev[t]
      pbar = pprev

      A = self.constructA(xbar, ubar, pbar)
      B = self.constructB(xbar, ubar, pbar)
      F = self.constructF(xbar, ubar, pbar)

      # relaxed constraint (it can also a matrix E be chosen and E*nu
      # be added but then nu has tu be a vector and the pair A,E has to be controllable)
      constraints.append(
        x[t + 1] == self.step(xbar, ubar, pbar) + A @ (x[t] - xbar) + B @ (u[t] - ubar) + F @ (p - pbar) + self.E @ nu[t]
      )

      # add constraints on helper variable for cost calculation (dyn and obst slack)
      constraints.append(
        cp.norm(self.E @ nu[t], 1) + cp.norm(nus[t],1) <= P[t]
      )

    # add constraints on helper variable for cost calculation (obst slack) for last time step
    constraints.append(
      cp.norm(nus[T-1], 1) <= P[T-1]
    )

    # add constraints on helper variable for cost calculation (initial condition slack)
    constraints.append(
      cp.norm(nuIC, 1) <= Pf[0]
    )

    # add constraints on helper variable for cost calculation (terminal condition slack)
    constraints.append(
      cp.norm(nuTC, 1) <= Pf[1]
    )

    #print("nr of constraints: ", len(constraints))
    return constraints

  def update_prob(self, x, u, sig, p, nu, nus, nuIC, nuTC, P, Pf, dx_lq, du_lq, dp_lq,
                  T, xf, x0, prob, xprev, uprev, sigprev, pprev,
                  nuprev, nusprev, nuICprev, nuTCprev, Pprev, Pfprev,
                  dx_lqprev, du_lqprev, dp_lqprev, CHandler, lin_cost_prev):

    # calc nonlinear augmented costfunction
    cost_aug_nl_prev = self.calc_aug_nl_cost(xprev, uprev, sigprev, pprev, x0, xf, T, CHandler)
    cost_aug_nl = self.calc_aug_nl_cost(x, u, sig, p, x0, xf, T, CHandler)

    # use update function only if previous problem value exists and the unconstraint problem is not solved
    # calculate accuracy of the convex prediction of the nonconvex problem
    roh = (cost_aug_nl_prev - cost_aug_nl) / (cost_aug_nl_prev - prob.value)

    # apply update rule
    if roh < self.roh0:  # when accuracy is lower then minimal accuracy stay at same point and only make trust radius smaller
      self.eta = jnp.max(np.array([self.eta0, self.eta / self.bsh]))
    else:
      xprev = np.array(x)
      uprev = np.array(u)
      nuprev = np.array(nu)
      nusprev = np.array(nus)
      nuICprev = np.array(nuIC)
      nuTCprev = np.array(nuTC)
      sigprev = np.array(sig)
      pprev = np.array(p)
      Pprev = np.array(P)
      Pfprev = np.array(Pf)
      dx_lqprev = np.array(dx_lq)
      du_lqprev = np.array(du_lq)
      dp_lqprev = np.array(dp_lq)

      if self.roh0 <= roh and roh < self.roh1:
        self.eta = jnp.max(np.array([self.eta0, self.eta / self.bsh]))
      elif self.roh1 <= roh and roh < self.roh2:
        self.eta = self.eta
      elif self.roh2 <= roh:
        self.eta = jnp.min(np.array([self.eta1, self.eta * self.bgr]))

    self.lam = self.lam_fac * self.lam

    return xprev, uprev, sigprev, pprev, nuprev, nusprev, nuICprev, nuTCprev, Pprev, Pfprev, dx_lqprev, du_lqprev, dp_lqprev, cost_aug_nl, prob.value, roh

  def calc_aug_nl_cost(self, x, u, sig, p, x0, xf, T, CHandler):
    # calculate terms of the nonlinear augmented costfunction

    # initial and final constraint violation
    ic_viol = self.redim_x(x[0]) - x0
    tc_viol = self.redim_x(x[-1]) - xf

    dc_viol = np.zeros((T - 1, len(x0)))
    if CHandler.obs_data:
      oc_viol = np.zeros((T, len(CHandler.obs_data)))
    else:
      oc_viol = np.zeros((T, 1))

    H1 = np.diag([2, 2, 2])
    c1 = np.array([0, 1, 0.9])

    P = np.zeros(T)
    for t in range(T - 1):
      # calculate difference between the dynamics of the solution found by the solver and the nonlinear dynamics
      delta = self.redim_x(x[t + 1]) - self.step(self.redim_x(x[t]), self.redim_u(u[t]), self.redim_p(p))
      dc_viol[t] = delta

      if self.collAvoidance == "paper":
        s_prev1 = np.max([1 - np.linalg.norm(H1 @ (self.redim_x(x[t])[:3] - c1)),0])
        oc_viol[t, 0] = np.max(s_prev1,0)

      else:
        # calculate violation of Obstacle constraints
        if CHandler.obs_data:
          dist, _, _ = CHandler.calcDistance(self.redim_x(x[t]))
          for i in range(len(dist)):
            # only take distance to colliding Obstacle
            if dist[i] < 0:
              oc_viol[t, i] = np.abs(dist[i])
            else:
              oc_viol[t, i] = 0

      P[t] = (np.linalg.norm(dc_viol[t],1) + np.linalg.norm(oc_viol[t],1))

    # add constraint violation of last state
    # calculate violation of Obstacle constraints

    if self.collAvoidance == "paper":
      s_prev1 = np.max([1 - np.linalg.norm(H1 @ (self.redim_x(x[T-1])[:3] - c1)), 0])
      oc_viol[T-1, 0] = np.max(s_prev1, 0)

    else:
      if CHandler.obs_data:
        dist, _, _ = CHandler.calcDistance(self.redim_x(x[T-1]))
        for i in range(len(dist)):
          # only take distance to colliding Obstacle
          if dist[i] < 0:
            oc_viol[T-1, i] = np.abs(dist[i])
          else:
            oc_viol[T-1, i] = 0

    P[T-1] = np.linalg.norm(oc_viol[T-1], 1)
    Pf = np.array([np.linalg.norm(ic_viol, 1), np.linalg.norm(tc_viol, 1)])

    inputs = self.robot.dt*np.sum((self.redim_sig(sig)/9.81)**2)
    time = (self.redim_p(p) / self.tf_max) ** 2

    # the different terms have to be added according to the defined costfunction !!
    cost = np.sum(self.lam * self.robot.dt * P) + np.sum(self.lam * Pf) + (1 - self.gam) * inputs + self.gam * time

    return np.sum(np.array(cost))

  def calc_initial_guess(self, x0, xf, T, stateDim, actionDim, nObstacles):

    class initial_guess():
      def __init__(self):
        self.x = None
        self.u = None
        self.sig = None
        self.nu = None
        self.nus = None
        self.nuIC = None
        self.nuTC = None
        self.p = None
        self.P = None
        self.Pf = None
        self.dx_lq = None
        self.du_lq = None
        self.dp_lq = None

    # set up initial trajectory
    initial_x = np.zeros((T, stateDim))
    initial_u = np.zeros((T, actionDim))

    # use interpolation between start and end point as initial guess
    initial_x[:, 0:3] = np.linspace(x0[0:3], xf[0:3], T)

    initial_u[:, 2] += 9.81  # add gravity compensation to acc in z direction

    # add noise
    initial_x[1:] += np.random.normal(0, 0.000, initial_x.shape)[1:]
    initial_u += np.random.normal(0, 0.0000, initial_u.shape)

    i_g = initial_guess()
    i_g.x = np.array(self.nondim_x(initial_x))
    i_g.u = np.array(self.nondim_u(initial_u))

    acc_mag = np.linalg.norm(initial_u[0],2)
    i_g.sig = np.array(self.nondim_sig(np.random.normal(acc_mag, 0.0, (T))))

    i_g.p = np.array(self.nondim_p(np.random.normal((self.tf_min + self.tf_max) / 2, 0.001, 1))).reshape((1))

    i_g.nu = np.random.normal(0, 1e-12, (T - 1, stateDim))
    if nObstacles == 0:
      i_g.nus = np.random.normal(0, 1e-12, (T, 1))
    else:
      i_g.nus = np.random.normal(0, 1e-12, (T, nObstacles))
    i_g.nuIC = np.random.normal(0, 1e-12, (stateDim))
    i_g.nuTC = np.random.normal(0, 1e-12, (stateDim))
    i_g.P = np.random.normal(0,1e-12,(T))
    i_g.Pf = np.random.normal(0,1e-12,(2))
    i_g.dx_lq = np.random.normal(0,1e-12,(T))
    i_g.du_lq = np.random.normal(0,1e-12,(T))
    i_g.dp_lp = np.random.normal(0, 1e-12, (1))

    return i_g

  def vis_process(self, itr, T, X, x0, xf, CHandler):
    plt.figure()
    ax = plt.axes(projection='3d')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, itr + 2)))
    leg = []
    for i in range(itr + 2):
      c = next(color)
      ax.plot3D(np.array(X[i])[:, 0], np.array(X[i])[:, 1], np.array(X[i])[:, 2], color=c, marker='o', markersize=3)
      leg.append("it: " + str(i + 1))

    ax.scatter3D(x0[0], x0[1], x0[2], color='k')
    ax.scatter3D(xf[0], xf[1], xf[2], color='k')
    if CHandler.obs_data:
      for obs in CHandler.obs_data:
        draw_obs(obs.type, obs.shape, obs.pos, ax)

    plt.legend(leg)
    ax.set_xlim([self.robot.min_x[0], self.robot.max_x[0]])
    ax.set_ylim([self.robot.min_x[1], self.robot.max_x[1]])
    ax.set_zlim([self.robot.min_x[2], self.robot.max_x[2]])

    if CHandler.obs_data:
      plt.figure()
      ax = plt.axes(projection='3d')
      for t in range(T):
        dist, pRob, pObs = CHandler.calcDistance_broadphase(np.array(X[-2])[t, :])
        dist2, pRob2, pObs2 = CHandler.calcDistance_broadphase(np.array(X[-1])[t, :])

        if dist > 0:
          dist_ = pRob - pObs
        else:
          dist_ = pObs - pRob

        norm_dist = np.linalg.norm(dist_)

        if norm_dist > 0:
          dist_hat = dist_ / norm_dist

        if dist2 < -0.03:  # allow up to 3cm violation
          print("Collision at t={} ({})".format(t, dist2))
          dist_approx = dist + dist_hat[:3].T @ (np.array(X[-1])[t, :3] - np.array(X[-2])[t, :3])
          print("should be positive: {}".format(dist_approx))

          ax.scatter3D(np.array(X[-1])[t, 0], np.array(X[-1])[t, 1], np.array(X[-1])[t, 2], color='r', marker='*')
          ax.scatter3D(np.array(X[-2])[t, 0], np.array(X[-2])[t, 1], np.array(X[-2])[t, 2], color='b', marker='*')
          # plot distance of point in previous iteration to Obstacle
          ax.plot3D([pRob[0], pObs[0]], [pRob[1], pObs[1]], [pRob[2], pObs[2]], color="r")  # , marker='o', markersize=3)
           # plot distance of point in this iteration
          ax.plot3D([pRob2[0], pObs2[0]], [pRob2[1], pObs2[1]], [pRob2[2], pObs2[2]],
                    color="b")  # , marker='o', markersize=3)
          # plot approximated distance
          if dist2 > 0:
            dist2_ = pRob2 - pObs2
          else:
            dist2_ = pObs2 - pRob2

          print("distance at point t: {}".format(dist))
          print("full distance at point t+1: {}".format(dist_))

          print("distance at point t+1: {}".format(dist2))
          print("full distance at point t+1: {}".format(dist2_))
          print("approximated distance at t+1: {}".format(dist_approx))

          # print("approximated distance at t+1: {}".format(dist_approx))

          norm_dist2 = np.linalg.norm(dist2_)

          if norm_dist2 > 0:
            dist_hat2 = dist2_ / norm_dist2

            ax.plot3D([pRob2[0], pRob2[0] + (dist_hat2 * dist_approx)[0]],
                      [pRob2[1], pRob2[1] + (dist_hat2 * dist_approx)[1]],
                      [pRob2[2], pRob2[2] + (dist_hat2 * dist_approx)[2]], color="g")  # , marker='o', markersize=3)

      if CHandler.obs_data:
        for obs in CHandler.obs_data:
          draw_obs(obs.type, obs.shape, obs.pos, ax)
      ax.set_xlim([self.robot.min_x[0], self.robot.max_x[0]])
      ax.set_ylim([self.robot.min_x[1], self.robot.max_x[1]])
      ax.set_zlim([self.robot.min_x[2], self.robot.max_x[2]])

      X_opt = X[-1]
      X_dist_to_opt = np.array([np.linalg.norm(Xi - X_opt,2) for Xi in X[1:]])
      plt.figure()
      plt.plot(X_dist_to_opt/np.linalg.norm(X_opt,2), marker = 'o', color = 'k')
      plt.yscale('log')
      plt.legend(["distance to final solution"])

    plt.show()
