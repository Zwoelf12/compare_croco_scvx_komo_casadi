from physics.collision.collisionHandler import collisionHandler
from optimization.KOMO.opt_problems import Komo
from optimization.SCP.SCvx import SCvx
from optimization.CASADI import Casadi

class OptProblem():
    def __init__(self):
        self.x0 = None  # initial state
        self.xf = None  # final state
        self.intermediate_states = None # intermediate states
        self.obs = None  # included obstacles
        self.algorithm = None # used algorithm
        self.robot = None  # used robot model
        self.par = None # method specific parameter
        self.alg_par = None # algorithm specific parameter
        self.CHandler = None # Collision handler
        self.tf_min = None # minimal flight time
        self.tf_max = None # maximum flight time
        self.initial_x = None # initialization of states
        self.initial_u = None # initialization of actions
        self.initial_p = None # initialization of time_dilation
        self.prob_name = None # short problem description

    def solve_problem(self):
        
        self.CHandler = collisionHandler(self.robot)
        self.CHandler.obs_data = self.obs

        if self.obs is not None:
            for obs in self.obs:
                self.CHandler.addObject(obs.type, obs.shape, obs.pos, obs.quat)

        if self.algorithm == "SCVX":
            
            Scvx = SCvx(self.robot,self.alg_par)
            solution = Scvx.solve(self.x0,
                                  self.xf,
                                  self.intermediate_states,
                                  self.tf_min,
                                  self.tf_max,
                                  self.initial_x,
                                  self.initial_u,
                                  self.initial_p,
                                  self.par.num_time_steps,
                                  self.par.max_num_iter,
                                  self.CHandler)
            
        elif self.algorithm == "KOMO":

            solution = Komo.solve(self.par.phases,
                                  self.par.time_steps_per_phase,
                                  self.par.time_per_phase,
                                  self.x0,
                                  self.xf,
                                  self.intermediate_states,
                                  self.obs,
                                  self.robot,
                                  self.initial_x,
                                  self.initial_u,
                                  self.alg_par)
            
        elif self.algorithm == "CASADI":

            if self.par.use_c_code:
                Casadi.update_c_code(self.robot,self.par.discretization_method)

            solution = Casadi.solve(self.robot,
                                    self.x0,
                                    self.xf,
                                    self.intermediate_states,
                                    self.tf_max,
                                    self.obs,
                                    self.initial_x,
                                    self.initial_u,
                                    self.par.num_time_steps,
                                    self.par.discretization_method,
                                    self.par.use_c_code,
                                    self.alg_par)

        else:
            print("unknown algorithm")

        return solution

