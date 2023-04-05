from optimization.KOMO.opt_problems import KomoProblem_fM
from optimization.KOMO.opt_problems import KomoProblem_dI
from optimization.opt_utils import extract_sol_dI, extract_sol_fM
import time

def solve(phases, time_steps_per_phase,
               time_per_phase, x0, xf, intermediate_states, obs,
               robot, initial_x, initial_u, alg_par):

    if robot.type == "dI":
        C, komo = KomoProblem_dI.doubleIntegrator_flight(phases, time_steps_per_phase,
                                                            time_per_phase, x0, xf, obs,
                                                            robot, initial_x)

        # solve optimization problem
        ts = time.time()
        komo.optimize()
        tf = time.time()

        solution = extract_sol_dI(C, komo, phases, time_steps_per_phase, time_per_phase,
                                  x0, ["drone"])
        solution.time = tf - ts

    elif robot.type == "fM":
        C, komo = KomoProblem_fM.fullModel_flight(phases, time_steps_per_phase,
                                                     time_per_phase, x0, xf, intermediate_states, obs,
                                                     robot, alg_par, initial_x, initial_u)

        # solve optimization problem
        ts = time.time()
        komo.optimize()
        tf = time.time()

        solution = extract_sol_fM(C, komo, phases, time_steps_per_phase, time_per_phase,
                                  x0, robot.nrMotors, ["drone"])
        solution.time = tf - ts
        solution.num_iter = komo.getIterations()
        solution.hessian_evals = komo.getEvals()
        print("komo constraint violations: ", komo.getConstraintViolations())

    else:
        print("unknown robot type")
        return False

    return solution