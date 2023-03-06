from optimization import opt_utils as ou
from visualization.report_visualization import report_compare
from optimization import problem_setups
from visualization.animation_visualization import animate_fM
from visualization.initial_guess_visualization import visualize_initial_guess

list_of_solvers = ["KOMO", "SCVX", "CASADI"]
animate_solution = "SCVX"

# choose which problem should be solved
prob = 2
if prob == 1:
    prob_setup = problem_setups.simple_flight_wo_obs()
    prob_name = "simple_flight_wo_obs"
elif prob == 2:
    prob_setup = problem_setups.complex_flight_spheres()
    prob_name = "complex_flight_spheres"
elif prob == 3:
    prob_setup = problem_setups.recovery_flight()
    prob_name = "recovery_flight"
elif prob == 4:
    prob_setup = problem_setups.flip()
    prob_name = "flip"

# define robot model
nr_motors = 4

# load solutions
solutions = ou.load_opt_output(prob_name, nr_motors, list_of_solvers)

# visualize solutions
report_compare(solutions, list_of_solvers)

sol = solutions[list_of_solvers[0]]
visualize_initial_guess(sol.initial_x, sol.x0, sol.xf, sol.intermediate_states, sol.obs)

# for solver_name in list_of_solvers:
sol = solutions[animate_solution]
animate_fM(sol.data,sol.obs)