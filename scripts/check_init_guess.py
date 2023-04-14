import visualization.initial_guess_visualization as iv
import optimization.opt_utils as ou

dat = ou.load_object("data/data_quim/fail_guess_croc_huge_noise/initial_guess_recovery_flight_1")
states = dat["init_x"]

iv.visualize_initial_guess(states, states[0], states[-1], None, None)
