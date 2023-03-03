from physics.collision.collisionHandler import collisionHandler
from optimization.CROCO.croco_AuLa import run_croco
from optimization.KOMO.opt_problems import Komo
from optimization.SCP.SCvx import SCvx
from optimization import opt_utils as ou

class OptProblem():
    def __init__(self):
        self.x0 = None  # initial state
        self.xf = None  # final state
        self.obs = None  # included obstacles
        self.algorithm = None # used algorithm
        self.robot = None  # used robot model
        self.par = None # parameter used in algorithm
        self.CHandler = None # Collision handler
        self.tf_min = None # minimal flight time
        self.tf_max = None # maximum flight time
        self.initial_x = None # initialization of states
        self.initial_u = None # initialization of actions
        self.initial_p = None # initialization of time_dilation

    def solve_problem(self):

        self.CHandler = collisionHandler(self.robot)
        self.CHandler.obs_data = self.obs

        if self.obs is not None:
            for obs in self.obs:
                self.CHandler.addObject(obs.type, obs.shape, obs.pos, obs.quat)

        if self.algorithm == "SCVX":

            scvx = SCvx(self.robot)
            solution = scvx.solve(self.x0,
                                  self.xf,
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
                                         self.obs,
                                         self.robot,
                                         self.initial_x,
                                         self.initial_u)

        elif self.algorithm == "CROCO":

            ou.gen_yaml_files(self.initial_x, self.initial_u, self.obs, self.x0, self.xf, self.robot)

            self.robot.t_dil_croco = self.tf_max

            solution = run_croco("../scripts/temp/env.yaml","../scripts/temp/guess.yaml",self.robot,True)

        else:
            print("unknown algorithm")

        return solution
