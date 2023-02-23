from physics.collision.collisionHandler import collisionHandler
from optimization.SCP.SCvx_paper_comparision_V2 import SCvx

class OptProblem():
    def __init__(self):
        self.x0 = None  # initial state
        self.xm = None  # intermediate state
        self.tm = None  # time at which intermediate state should be reached
        self.xf = None  # final state
        self.obs = None  # included obstacles
        self.algorithm = None # used algorithm
        self.robot = None  # used robot model
        self.par = None # parameter used in algorithm
        self.CHandler = None # Collision handler
        self.tf_min = None
        self.tf_max = None

    def solve_problem(self, param):
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
                                  param.num_time_steps,
                                  param.max_num_iter,
                                  self.CHandler)

        return solution
