import numpy as np

class SCVX_parameter():

    def __init__(self, prob_name):

        if prob_name == "simple_flight_wo_obs":

            # user defined parameter
            self.lam = 1e2  # weight 3 slack in cost
            self.alp = 0.  # weight for time in cost
            self.bet = 1e3  # weight for input in cost
            self.gam = 0.  # weight ratio between input and time penalty
            self.adapWeightsFac = 1e1  # determines by how much the slack cost is decreased and the problem cost is increased when slack is in a reasonable range
            self.weightsFac = 1
            self.E = np.eye(13)  # weight matrix for dynamic slack
            # when to update trust region
            self.roh0 = 0.
            self.roh1 = 0.1
            self.roh2 = 0.7
            # growth and shrink parameter for trust region
            self.bgr = 2
            self.bsh = 2
            # initial trust region and minimal and maximal trust region radius
            self.eta = 10
            self.eta0 = 1e-3
            self.eta1 = 100
            # stopping criterion
            self.eps = 1e-3
            self.eps_t = 1e-4
            # norm order for trust region
            self.q = "inf"

        elif prob_name == "complex_flight_spheres":

            # user defined parameter
            self.lam = 1e3  # weight 3 slack in cost
            self.alp = 0.  # weight for time in cost
            self.bet = 1e3  # weight for input in cost
            self.gam = 0.  # weight ratio between input and time penalty
            self.adapWeightsFac = 1e1  # determines by how much the slack cost is decreased and the problem cost is increased when slack is in a reasonable range
            self.weightsFac = 1
            self.E = np.eye(13)  # weight matrix for dynamic slack
            # when to update trust region
            self.roh0 = 0.
            self.roh1 = 0.1
            self.roh2 = 0.7
            # growth and shrink parameter for trust region
            self.bgr = 2
            self.bsh = 2
            # initial trust region and minimal and maximal trust region radius
            self.eta = 10
            self.eta0 = 1e-3
            self.eta1 = 100
            # stopping criterion
            self.eps = 1e-3
            self.eps_t = 1e-4
            # norm order for trust region
            self.q = "inf"

        elif prob_name == "recovery_flight":

            # user defined parameter
            self.lam = 1e3  # weight 3 slack in cost
            self.alp = 0.  # weight for time in cost
            self.bet = 1e3  # weight for input in cost
            self.gam = 0.  # weight ratio between input and time penalty
            self.adapWeightsFac = 1e1  # determines by how much the slack cost is decreased and the problem cost is increased when slack is in a reasonable range
            self.weightsFac = 1
            self.E = np.eye(13)  # weight matrix for dynamic slack
            # when to update trust region
            self.roh0 = 0.
            self.roh1 = 0.1
            self.roh2 = 0.7
            # growth and shrink parameter for trust region
            self.bgr = 2
            self.bsh = 2
            # initial trust region and minimal and maximal trust region radius
            self.eta = 10
            self.eta0 = 1e-3
            self.eta1 = 100
            # stopping criterion
            self.eps = 1e-3
            self.eps_t = 1e-4
            # norm order for trust region
            self.q = "inf"

        elif prob_name == "flip":

            # user defined parameter
            self.lam = 1e3  # weight 3 slack in cost
            self.alp = 0.  # weight for time in cost
            self.bet = 1e3  # weight for input in cost
            self.gam = 0.  # weight ratio between input and time penalty
            self.adapWeightsFac = 1e1  # determines by how much the slack cost is decreased and the problem cost is increased when slack is in a reasonable range
            self.weightsFac = 1
            self.E = np.eye(13)  # weight matrix for dynamic slack
            # when to update trust region
            self.roh0 = 0.
            self.roh1 = 0.1
            self.roh2 = 0.7
            # growth and shrink parameter for trust region
            self.bgr = 2
            self.bsh = 2
            # initial trust region and minimal and maximal trust region radius
            self.eta = 10
            self.eta0 = 1e-3
            self.eta1 = 100
            # stopping criterion
            self.eps = 1e-3
            self.eps_t = 1e-4
            # norm order for trust region
            self.q = "inf"


class KOMO_parameter():

    def __init__(self, prob_name):

        if prob_name == "simple_flight_wo_obs":
            self.weight_dynamics = 1e2*0.1
            self.weight_input = 1e3*7

        elif prob_name == "complex_flight_spheres":
            self.weight_dynamics = 1e3*5
            self.weight_input = 1e2*10

        elif prob_name == "recovery_flight":
            self.weight_dynamics = 1e2 * 0.1
            self.weight_input = 1e3 * 7

        elif prob_name == "flip":
            self.weight_dynamics = 1e2 * 0.1
            self.weight_input = 1e3 * 7


class CASADI_parameter():

    def __init__(self, prob_name):

        if prob_name == "simple_flight_wo_obs":
            self.par = None

        elif prob_name == "complex_flight_spheres":
            self.par = None

        elif prob_name == "recovery_flight":
            self.par = None

        elif prob_name == "flip":
            self.par = None

