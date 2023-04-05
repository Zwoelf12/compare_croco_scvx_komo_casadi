import cvxpy as cp
import jax.numpy as jnp  # Thinly-wrapped numpy
import numpy as np
from jax import jacfwd, jit, config
config.update("jax_enable_x64", True)
import time
from optimization.opt_utils import OptSolution, calc_initial_guess

class SCvx():
  def __init__(self, robot,alg_par):

    self.useScaling = True
    self.useHelperVars = True
    self.useQuatNormalization = True
    self.useEarlyStopping = False
    self.useAdaptiveWeights = False
    
    self.solver = "ecos"

    self.robot = robot
    # inflate robot to avoid corner cutting
    self.robot.arm_length = robot.arm_length

    # user defined parameter
    self.lam = alg_par.lam # weight 3 slack in cost
    self.alp = alg_par.alp  # weight for time in cost
    self.bet = alg_par.bet # weight for input in cost
    self.gam = alg_par.gam  # weight ratio between input and time penalty
    self.adapWeightsFac = alg_par.adapWeightsFac # determines by how much the slack cost is decreased and the problem cost is increased when slack is in a reasonable range
    self.weightsFac = alg_par.weightsFac
    self.E = alg_par.E #weight matrix for dynamic slack
    # when to update trust region
    self.roh0 = alg_par.roh0
    self.roh1 = alg_par.roh1
    self.roh2 = alg_par.roh2
    # growth and shrink parameter for trust region
    self.bgr = alg_par.bgr
    self.bsh = alg_par.bsh
    # initial trust region and minimal and maximal trust region radius
    self.eta = alg_par.eta
    self.eta0 = alg_par.eta0
    self.eta1 = alg_par.eta1
    # stopping criterion
    self.eps = alg_par.eps
    self.eps_t = alg_par.eps_t
    # norm order for trust region
    self.q = alg_par.q

    # build A & B Matrices and f
    self.constructA = jit(jacfwd(robot.step, 0))
    self.constructB = jit(jacfwd(robot.step, 1))
    self.constructF = jit(jacfwd(robot.step, 2))
    self.step = jit(robot.step)

  def solve(self, x0, xf, intermediate_states,
    tf_min, tf_max, initial_x,
    initial_u, initial_p, T = 50,
    num_iterations = 10,
    CHandler=None):

    time_all_s = time.time()

    stateDim = x0.shape[0]
    actionDim = self.robot.min_u.shape[0]

    self.tf_min = tf_min
    self.tf_max = tf_max

    # variable scaling to make SCvx scale invariant
    # determine scaling matrices
    if self.useScaling:
      self.Sx = np.diag(self.robot.max_x - self.robot.min_x)
      self.Su = np.diag(self.robot.max_u - self.robot.min_u)
      self.Sp = self.tf_max - self.tf_min
      # determine offset
      self.cx = self.robot.min_x
      self.cu = self.robot.min_u
      self.cp = self.tf_min

    else:
      self.Sx = np.eye(stateDim)
      self.Su = np.eye(actionDim)
      self.Sp = 1
      # determine offset
      self.cx = np.zeros(stateDim)
      self.cu = np.zeros(actionDim)
      self.cp = 0

    if CHandler.obs_data:
      nObstacles = len(CHandler.obs_data)
    else:
      nObstacles = 0

    initial_guess = self.get_initial_guess(initial_x, initial_u, initial_p, intermediate_states, T, nObstacles)

    xprev = initial_guess.x
    uprev = initial_guess.u
    nuprev = initial_guess.nu
    nusprev = initial_guess.nus
    nuICprev = initial_guess.nuIC
    nuTCprev = initial_guess.nuTC
    nuMCprev = initial_guess.nuMC
    pprev = initial_guess.p
    Pprev = initial_guess.P
    Pfprev = initial_guess.Pf
    dx_lqprev = initial_guess.dx_lq
    du_lqprev = initial_guess.du_lq
    dp_lqprev = initial_guess.dp_lq

    prevNLCost = None
    trajCrit = np.inf
    lin_cost = []
    nonlin_cost = []
    stop_flag = False
    interface_time = []
    cvx_solver_time = []
    n_convex_iterations = []
    for itr in range(num_iterations):

      start_time = time.time() # start timing

      x = cp.Variable((T, stateDim))
      u = cp.Variable((T, actionDim))
      p = cp.Variable(1)
      nu = cp.Variable((T - 1, stateDim))
      if nObstacles == 0:
        nus = cp.Variable((T, 1))
      else:
        nus = cp.Variable((T, nObstacles))
      nuIC = cp.Variable(stateDim)
      nuTC = cp.Variable(stateDim)
      if intermediate_states is not None:
        nuMC = cp.Variable(stateDim)
        Pf = cp.Variable(3)
      else:
        nuMC = None
        Pf = cp.Variable(2)
      P = cp.Variable(T)
      dx_lq = cp.Variable(T)
      du_lq = cp.Variable(T)
      dp_lq = cp.Variable(1)

      # set initial guess for warm start
      x.value = xprev
      u.value = uprev
      p.value = pprev
      nu.value = nuprev
      nus.value = nusprev
      nuIC.value = nuICprev
      nuTC.value = nuTCprev
      if intermediate_states is not None:
        nuMC.value = nuMCprev
      P.value = Pprev
      Pf.value = Pfprev
      dx_lq.value = dx_lqprev
      du_lq.value = du_lqprev
      dp_lq.value = dp_lqprev

      # scale problem
      x_ph, u_ph, p_ph = self.scale_prob(x, u, p)
      x_prev_ph, u_prev_ph, p_prev_ph = self.scale_prob(xprev, uprev, pprev)

      if itr == 0:
        X_, U_, P_, = [x_prev_ph], \
                      [u_prev_ph], \
                      [p_prev_ph]

        NU_, NUS_, NUIC_, NUTC_, NUMC_ = [nuprev], \
                                         [nusprev], \
                                         [nuICprev], \
                                         [nuTCprev], \
                                         [nuMCprev]

        prevNLCost = self.calc_aug_nl_cost(xprev, uprev, pprev, x0, xf, intermediate_states, T, CHandler)

      # define objective
      cp_objective = self.define_objective(u_ph, p_ph, P, Pf, nu, nus, nuIC, nuTC, nuMC, intermediate_states)

      # define constraints
      constraints = self.define_constraints(x_ph, u_ph, p_ph,
                                            nu, nus, nuIC, nuTC, nuMC, P, Pf,
                                            dx_lq, du_lq, dp_lq,
                                            x_prev_ph, u_prev_ph, p_prev_ph,
                                            x0, xf, intermediate_states, T, CHandler)

      # define problem
      prob = cp.Problem(cp_objective, constraints)

      start_time_sol_interface = time.time()
      try:
        if self.solver == "gurobi":
          if self.useEarlyStopping and trajCrit >= self.eps_t*100:
            result = prob.solve(verbose=True, warm_start=True, solver=cp.GUROBI, BarQCPConvTol=1e-4, BarConvTol=1e-4)#, SolutionLimit = 1, OptimalityTol = 1e-2)
          else:
            result = prob.solve(verbose=False, warm_start=True, solver=cp.GUROBI, BarQCPConvTol=1e-10)
        elif self.solver == "ecos":
          result = prob.solve(verbose=False, warm_start=True, solver=cp.ECOS_BB)

        # calculate solver time and time spend in the cvxpy interface
        end_time_sol_interface = time.time()
        time_convex_solver = prob.solver_stats.solve_time
        time_interface = end_time_sol_interface - start_time_sol_interface - time_convex_solver

        interface_time.append(time_interface)
        cvx_solver_time.append(time_convex_solver)
        n_convex_iterations.append(prob.solver_stats.num_iters)

        prob_value = prob.value

      except:
        print("infeasible sub-problem - cvx solver failed")
        interface_time.append(np.NaN)
        cvx_solver_time.append(np.NaN)
        time_all_e = np.inf

        prob_value = np.inf

        states = np.array(X_[-1])
        actions = np.vstack((np.array(U_[-1])))
        actions[-1, :] = float("nan")
        time_dilation = self.redim_p(p.value)

        solution = OptSolution(states, actions, time_all_e - time_all_s, nu.value, prob_value, itr + 1, time_dilation, sum(interface_time), sum(cvx_solver_time))
        return solution
      
      # calculate nonlinear augmented cost
      nlCost = self.calc_aug_nl_cost(x.value, u.value, p.value, x0, xf, intermediate_states, T, CHandler)
      # stopping criterion
      trajCrit = np.linalg.norm(p.value - pprev, 1) + np.max(np.linalg.norm(x.value - xprev, 1, 1)) # criterion following Malyuta et. al. looks only at trajectory change
      costCrit = np.abs(prevNLCost - prob_value) / np.abs(prevNLCost) # criterion with respect to optimal value

      print("problem value: ", prob_value)
      print("nonlinear cost: ", nlCost)
      print("part 1 tajCrit: ", np.linalg.norm(p.value - pprev, 1))
      print("part 2 tajCrit: ", np.max(np.linalg.norm(x.value - xprev, 1, 1)))
      print("costCrit: ", costCrit)


      if costCrit < self.eps or trajCrit < self.eps_t:
        stop_flag = True

      if intermediate_states is not None:
        nuMC_value = nuMC.value
      else:
        nuMC_value = np.zeros(stateDim)

      # update rule
      xprev, uprev, pprev, nuprev, nusprev, nuICprev, nuTCprev, nuMCprev, Pprev, Pfprev, dx_lqprev, du_lqprev, dp_lqprev, roh = self.update_prob(x.value, u.value, p.value,
                                                                                                                                       nu.value, nus.value, nuIC.value, nuTC.value, nuMC_value, P.value, Pf.value,
                                                                                                                                       dx_lq.value, du_lq.value, dp_lq.value, prob_value,
                                                                                                                                       xprev, uprev, pprev,
                                                                                                                                       nuprev, nusprev, nuICprev, nuTCprev, nuMCprev, Pprev, Pfprev,
                                                                                                                                       dx_lqprev, du_lqprev, dp_lqprev,
                                                                                                                                       nlCost, prevNLCost)

      # add obtained trajectory
      x_ph, u_ph, p_ph = self.scale_prob(x.value, u.value, p.value)

      X_.append(x_ph)
      U_.append(u_ph)
      P_.append(p_ph)
      NU_.append(nu.value)
      NUS_.append(nus.value)
      NUIC_.append(nuIC.value)
      NUTC_.append(nuTC.value)
      NUMC_.append(nuMC_value)
      print("intermediate constraints violations True: ", np.linalg.norm(nuMC_value, 1))
      lin_cost.append(prob_value)
      nonlin_cost.append(nlCost)

      # update previous nonlinear cost
      prevNLCost = nonlin_cost[-1]

      end_time = time.time()

      print("________________________________________")
      print("iteration: ", itr + 1)
      print("max dyn constraint violation: ", np.max(np.abs(nu.value)))
      print("max collision constraint violation: ", np.max(np.abs(nus.value)))
      print("max init constraint violation: ", np.max(np.abs(nuIC.value)))
      print("max terminal constraint violation: ", np.max(np.abs(nuTC.value)))
      print("max intermediate constraint violation: ", np.max(np.abs(nuMC_value)))
      print("trajectory is executed in: {} s".format(self.redim_p(p.value)))
      print("time for iteration: ", np.round(end_time - start_time, 4))
      print("time to solve convex subproblem: ", np.round(end_time_sol_interface - start_time_sol_interface, 4))
      print("________________________________________")

      if stop_flag == True:
        break

      if itr == num_iterations - 1:
        print("reached max num of iterations")

    time_all_e = time.time()

    states = np.array(X_[-1])
    actions = np.vstack((np.array(U_[-1])))
    actions[-1,:] = float("nan")
    time_dilation = self.redim_p(p.value)

    solution = OptSolution(states, actions, time_all_e - time_all_s, nu.value, prob_value, itr + 1, time_dilation, sum(interface_time), sum(cvx_solver_time), num_cvx_iter = sum(n_convex_iterations), hessian_evals = sum(n_convex_iterations))
    return solution

###################################################################################################################################################################################

  def scale_prob(self, x, u, p):

    # scale states and inputs
    x_ph = (self.Sx @ x.T + self.cx[:, np.newaxis]).T
    u_ph = (self.Su @ u.T + self.cu[:, np.newaxis]).T

    p_ph = self.Sp * p + self.cp

    return x_ph, u_ph, p_ph

  def redim_x(self,x):
    return self.Sx @ x + self.cx

  def redim_u(self,u):
    return self.Su @ u + self.cu

  def redim_p(self,p):
    return self.Sp * p + self.cp

  def nondim_x(self,xp):
    return (np.linalg.inv(self.Sx) @ (xp - self.cx).T).T

  def nondim_u(self,up):
    return (np.linalg.inv(self.Su) @ (up - self.cu).T).T

  def nondim_p(self,pp):
    if self.Sp == 0:
      return self.cp
    else:
      return (1/self.Sp) * (pp - self.cp)


###################################################################################################################################################################################

  def define_objective(self, u, p, P, Pf, nu, nus, nuIC, nuTC, nuMC, intermediate_states):

    inputs = self.robot.dt * cp.sum(u**2) # penalty on inputs

    if self.useHelperVars:
      runningSlackCost = cp.sum(self.lam * self.robot.dt * P)
      terminalSlackCost = cp.sum(self.lam * Pf)
      slackCost = runningSlackCost + terminalSlackCost # penalty for slack
    else:
      runningSlackCost = self.lam * self.robot.dt * (cp.norm(self.E @ nu.T, 1) + cp.norm(nus, 1))
      terminalSlackCost = self.lam * (cp.norm(nuIC,1) + cp.norm(nuTC,1))
      if intermediate_states is not None:
        terminalSlackCost += self.lam * cp.norm(nuMC,1)
      slackCost = runningSlackCost + terminalSlackCost # penalty for slack

    time = (p/self.tf_max)**2 # penalty on time dilation

    probCost = self.bet*(1-self.gam)*inputs + self.gam*time

    cp_objective = cp.Minimize(slackCost * (1 / self.weightsFac) + probCost)

    return cp_objective

  ###################################################################################################################################################################################
  
  def calc_aug_nl_cost(self, x, u, p, x0, xf, intermediate_states, T, CHandler):
    # calculate terms of the nonlinear augmented costfunction
    dc_viol = np.zeros((T - 1, len(x0)))
    if CHandler.obs_data:
      oc_viol = np.zeros((T, len(CHandler.obs_data)))
    else:
      oc_viol = np.zeros((T, 1))

    x_ph, u_ph, p_ph = self.scale_prob(x,u,p)

    # initial and final constraint violation
    ic_viol = x_ph[0] - x0
    tc_viol = x_ph[-1] - xf

    # calculate intermediate constraints violations
    violations = []
    if intermediate_states is not None:
        for i_s in intermediate_states:
          state_viol = np.zeros(len(x0))

          if "pos" in i_s.type:
            state_viol[:3] = x_ph[i_s.timing,:3] - i_s.value[:3]
          if "vel" in i_s.type:
            state_viol[3:6] = x_ph[i_s.timing,3:6] - i_s.value[3:6]
          if "quat" in i_s.type:
            state_viol[6:10] = x_ph[i_s.timing,6:10] - i_s.value[6:10]
          if "rot_vel" in i_s.type:
            state_viol[10:] = x_ph[i_s.timing,10:] - i_s.value[10:]

          violations.append(state_viol)

          mc_viol = np.max(np.vstack(violations), axis = 0)

        print("intermediate constraints violations augmented: ", np.linalg.norm(mc_viol, 1))


    P = np.zeros(T)
    for t in range(T - 1):
      # calculate difference between the dynamics of the solution found by the solver and the nonlinear dynamics
      delta = x_ph[t + 1] - self.step(x_ph[t], u_ph[t], p_ph)
      dc_viol[t] = delta

      # calculate violation of Obstacle constraints
      if CHandler.obs_data:
        dist, _, _ = CHandler.calcDistance(x_ph[t])
        for i in range(len(dist)):
          # only take distance to colliding Obstacle
          if dist[i] < 0:
            oc_viol[t, i] = np.abs(dist[i])
          else:
            oc_viol[t, i] = 0

      P[t] = (np.linalg.norm(dc_viol[t],1) + np.linalg.norm(oc_viol[t],1))

    # calculate violation of Obstacle constraints
    if CHandler.obs_data:
      dist, _, _ = CHandler.calcDistance(x_ph[T-1])
      for i in range(len(dist)):
        # only take distance to colliding Obstacle
        if dist[i] < 0:
          oc_viol[T-1, i] = np.abs(dist[i])
        else:
          oc_viol[T-1, i] = 0

    P[T-1] = np.linalg.norm(oc_viol[T-1], 1)


    Pf = np.array([np.linalg.norm(ic_viol, 1), np.linalg.norm(tc_viol, 1)])
    if intermediate_states is not None:
      np.append(Pf,np.linalg.norm(mc_viol, 1))

    inputs = self.robot.dt * np.sum(u_ph ** 2)

    if self.useHelperVars:
      runningSlackCost = np.sum(self.lam * self.robot.dt * P)
      terminalSlackCost = np.sum(self.lam * Pf)
      slackCost = runningSlackCost + terminalSlackCost
    else:
      runningSlackCost = self.lam * self.robot.dt * (np.linalg.norm(self.E @ dc_viol.T, 1) + np.linalg.norm(oc_viol, 1))
      terminalSlackCost = self.lam * (np.linalg.norm(ic_viol, 1) + np.linalg.norm(tc_viol, 1))
      if intermediate_states is not None:
        terminalSlackCost += self.lam * np.linalg.norm(mc_viol, 1)
      slackCost = runningSlackCost + terminalSlackCost  # penalty for slack

    time = (p_ph / self.tf_max) ** 2

    probCost = self.bet*(1 - self.gam) * inputs + self.gam * time

    # the different terms have to be added according to the defined costfunction !!
    cost = slackCost * (1 / self.weightsFac) + probCost

    return np.sum(np.array(cost))

###################################################################################################################################################################################

  def define_constraints(self, x, u, p,
                         nu, nus, nuIC, nuTC, nuMC, P, Pf,
                         dx_lq, du_lq, dp_lq,
                         xprev, uprev, pprev,
                         x0, xf, intermediate_states, T, CHandler):

    # initial, final and intermediate constraints
    if self.robot.type == "dI" or (self.robot.nrMotors != 2 and self.robot.nrMotors != 3):
      constraints = [
        x[0] == x0 + nuIC, # initial state constraint
        x[T-1] == xf + nuTC # final state constraint
      ]

      if intermediate_states is not None:  # intermediate
        for i_s in intermediate_states:
          if "pos" in i_s.type:
            constraints.append(x[i_s.timing,:3] == i_s.value[:3] + nuMC[:3])
          if "vel" in i_s.type:
            constraints.append(x[i_s.timing,3:6] == i_s.value[3:6] + nuMC[3:6])
          if "quat" in i_s.type:
            constraints.append(x[i_s.timing,6:10] == i_s.value[6:10] + nuMC[6:10])
          if "rot_vel" in i_s.type:
            constraints.append(x[i_s.timing,10:] == i_s.value[10:] + nuMC[10:])

    else:
      constraints = [
        x[0] == x0 + nuIC,  # initial state constraint
        x[T-1, :6] == xf[:6] + nuTC[:6]  # final state constraint
      ]

    # add constraints on helper variables (p) for trust region calculation
    if self.useHelperVars:
      constraints.append(
        cp.norm(self.nondim_p(p) - self.nondim_p(pprev), self.q) <= dp_lq
      )

    # constraints on time dilation
    constraints.extend([
      p >= self.tf_min,
      p <= self.tf_max
    ])

    for t in range(0, T):

      if self.useHelperVars:
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

      else:
        # trust region constraints
        constraints.append(
          cp.norm(self.nondim_x(x[t]) - self.nondim_x(xprev[t]), self.q) +
          cp.norm(self.nondim_u(u[t]) - self.nondim_u(uprev[t]), self.q) +
          cp.norm(self.nondim_p(p) - self.nondim_p(pprev), self.q) <= self.eta
        )

      # input constraints
      constraints.extend([
        u[t] >= self.robot.min_u,
        u[t] <= self.robot.max_u
      ])

      # state constraints
      if self.robot.type == "dI" or (self.robot.nrMotors != 2 and self.robot.nrMotors != 3):
        
        constraints.extend([
          x[t] >= self.robot.min_x,
          x[t] <= self.robot.max_x
        ])
      else:
        constraints.extend([
          x[t, :10] >= self.robot.min_x[:10],
          x[t, :10] <= self.robot.max_x[:10]
        ])

      # quaternion normalization (seems to be sufficient to constraint ||q||2 from above, else the constraint has to be convexified)
      if self.robot.type == "fM":
        if self.useQuatNormalization == True:
          if t != 0 and t != T-1:
            constraints.append(
              cp.norm(x[t, 6:10], 2) <= 1
            )

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
            CHandler.redim_robot(arm_length*0.99999999)

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

      if t != T-1:

        # dynamic constraints
        xbar = xprev[t]
        ubar = uprev[t]
        pbar = pprev

        A = self.constructA(xbar, ubar, pbar)
        B = self.constructB(xbar, ubar, pbar)
        F = self.constructF(xbar, ubar, pbar)

        # relaxed constraint (it can also a matrix E be chosen and E*nu be added but then the pair A,E has to be controllable)

        constraints.append(
          x[t + 1] == self.step(xbar, ubar, pbar) + A @ (x[t] - xbar) + B @ (u[t] - ubar) + F @ (p - pbar) + self.E @ nu[t]
        )

        # add constraints on helper variable for cost calculation (dyn and obst slack)
        if self.useHelperVars:
          constraints.append(
            cp.norm(self.E @ nu[t], 1) + cp.norm(nus[t], 1) <= P[t]
          )

    if self.useHelperVars:
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

      # add constraints on helper variable for cost calculation (intermediate condition slack)
      if intermediate_states is not None:
        constraints.append(
          cp.norm(nuMC, 1) <= Pf[2]
        )

    return constraints

###################################################################################################################################################################################


  def update_prob(self, x, u, p, nu, nus, nuIC, nuTC, nuMC, P, Pf, dx_lq, du_lq, dp_lq,
                  prob_value, xprev, uprev, pprev,
                  nuprev, nusprev, nuICprev, nuTCprev, nuMCprev, Pprev, Pfprev,
                  dx_lqprev, du_lqprev, dp_lqprev, cost_aug_nl, cost_aug_nl_prev):


    # calculate accuracy of the convex prediction of the nonconvex problem
    roh = (cost_aug_nl_prev - cost_aug_nl) / (cost_aug_nl_prev - prob_value)

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
      nuMCprev = np.array(nuMC)
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

    if self.useAdaptiveWeights:
      if np.max(np.abs(nu)) <= 1e-4 and np.max(np.abs(nus)) <= 1e-5 and np.max(np.abs(nuIC)) <= 1e-5 and np.max(np.abs(nuTC)) <= 1e-5 and np.max(np.abs(nuMC)) <= 1e-5:
        self.weightsFac = self.adapWeightsFac

    return xprev, uprev, pprev, nuprev, nusprev, nuICprev, nuTCprev, nuMCprev, Pprev, Pfprev, dx_lqprev, du_lqprev, dp_lqprev, roh

  ###################################################################################################################################################################################
  

  def get_initial_guess(self,initial_x, initial_u, initial_p, intermediate_states, T, nObstacles):

    class initial_guess():
      def __init__(self):
        self.x = None
        self.u = None
        self.nu = None
        self.nus = None
        self.nuIC = None
        self.nuTC = None
        self.nuMC = None
        self.p = None
        self.P = None
        self.Pf = None
        self.dx_lq = None
        self.du_lq = None
        self.dp_lq = None

    i_g = initial_guess()
    i_g.x = np.array(self.nondim_x(initial_x))
    i_g.u = np.array(self.nondim_u(initial_u))
    i_g.p = np.array(self.nondim_p(initial_p)).reshape((1))

    i_g.nu = np.random.normal(0, 1e-12, (T - 1, self.robot.max_x.shape[0]))
    if nObstacles == 0:
      i_g.nus = np.random.normal(0, 1e-12, (T, 1))
    else:
      i_g.nus = np.random.normal(0, 1e-12, (T, nObstacles))
    i_g.nuIC = np.random.normal(0, 1e-12, (self.robot.max_x.shape[0]))
    i_g.nuTC = np.random.normal(0, 1e-12, (self.robot.max_x.shape[0]))
    if intermediate_states is not None:
      i_g.nuMC = np.random.normal(0, 1e-12, (self.robot.max_x.shape[0]))
    i_g.P = np.random.normal(0,1e-12,(T))
    if intermediate_states is not None:
      i_g.Pf = np.random.normal(0,1e-12,(3))
    else:
      i_g.Pf = np.random.normal(0,1e-12,(2))
    i_g.dx_lq = np.random.normal(0,1e-12,(T))
    i_g.du_lq = np.random.normal(0,1e-12,(T))
    i_g.dp_lq = np.random.normal(0, 1e-12, (1))

    return i_g

