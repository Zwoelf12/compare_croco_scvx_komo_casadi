from visualization.geom_visualization import draw_obs
from optimization import opt_utils as ou
import rowan
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from stl import mesh
from mpl_toolkits import mplot3d
import rowan
import copy
import matplotlib.patches as mpatches
from coloraide import Color
import pylab as ply
import matplotlib

# line cyclers adapted to colourblind people
from cycler import cycler
line_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

# adapted standard line cycler
standard_cycler = (cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#9467bd", "#bcbd22"])+
				   cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-.", "-", "--","-"]))

bw_cycler = cycler(color=["0","0.05","0.1","0.15","0.2","0.25","0.3","0.35","0.4","0.45","0.5"])

fancy_cycler = (cycler(color=["dodgerblue", "orange", "deeppink", "limegreen"])+
				   cycler(linestyle=["-", "--", "-.", ":"]))

# set line cycler
plt.rc("axes", prop_cycle=standard_cycler)

# set latex font
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Computer Modern Roman",
})

SMALL_SIZE = 24#12
MEDIUM_SIZE = 26#15
BIGGER_SIZE = 28#18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def report_compare(solutions, list_of_solvers):

	robot = solutions[list_of_solvers[0]].robot
	x0 = solutions[list_of_solvers[0]].x0
	xf = solutions[list_of_solvers[0]].xf
	t_dil = solutions[list_of_solvers[0]].t_dil
	obstacles = solutions[list_of_solvers[0]].obs

	nSolvers = len(list_of_solvers)

	nTime_steps = {}
	time_points = {}
	for solver_name in list_of_solvers:
		nTime_steps[solver_name] = solutions[solver_name].data.shape[0]
		time_points[solver_name] = np.linspace(0,t_dil,nTime_steps[solver_name])

	fig_states, axs = plt.subplots(2, 2)

	ax1 = axs[0, 0]
	ax2 = axs[0, 1]
	ax3 = axs[1, 0]
	ax4 = axs[1, 1]

	legend_ax1 = []
	legend_ax2 = []
	legend_ax3 = []
	legend_ax4 = []

	for solver_name in list_of_solvers:

		sol = solutions[solver_name]
		time_vec = time_points[solver_name]

		# plot positions
		ax1.plot(time_vec, sol.data[:, :3])
		legend_ax1.extend([solver_name + ": \n $p_x$", "$p_y$", "$p_z$"])

		# plot velocities
		ax2.plot(time_vec, sol.data[:, 3:6])
		legend_ax2.extend([solver_name + ": \n $v_x$", "$v_y$", "$v_z$"])

		# plot quaternions
		ax3.plot(time_vec, sol.data[:, 6:10])
		legend_ax3.extend([solver_name + ": \n $q_1$", "$q_2$", "$q_3$", "$q_4$"])

		# plot rotational velocities
		ax4.plot(time_vec, sol.data[:, 10:13])
		legend_ax4.extend([solver_name + ": \n $\omega_x$", "$\omega_y$", "$\omega_z$"])

	# make legend
	ax1.legend(legend_ax1)
	ax2.legend(legend_ax2)
	ax3.legend(legend_ax3)
	ax4.legend(legend_ax4)

	ax1.set_xlabel("time [s]")
	ax1.set_ylabel("positions [m]")
	ax1.set_title("position comparision")

	ax2.set_xlabel("time [s]")
	ax2.set_ylabel("velocities [m/s]")
	ax2.set_title("velocity comparision")

	ax3.set_xlabel("time [s]")
	ax3.set_ylabel("quaternion [1]")
	ax3.set_title("quaternion comparision")

	ax4.set_xlabel("time [s]")
	ax4.set_ylabel("rotational velocities [rad/s]")
	ax4.set_title("rotational velocities comparision")

	# plot the trajectory obtained by the optimizer
	fig_traj = plt.figure()
	ax = mplot3d.Axes3D(fig_traj)

	colors = ["r","b","g","m"]
	for i, solver_name in zip(range(nSolvers),list_of_solvers):
		sol = solutions[solver_name]
		ax.plot3D(sol.data[:, 0], sol.data[:, 1], sol.data[:, 2], colors[i] , marker = 'o', markersize=4)

	ax.scatter3D(x0[0], x0[1], x0[2], color='k')
	ax.scatter3D(xf[0], xf[1], xf[2], color='k')

	if obstacles is not None:
		for obs in obstacles:
			draw_obs(obs.type, obs.shape, obs.pos, ax)


	plt.legend(list_of_solvers)

	ax = plt.gca()
	ax.set_xlim([robot.min_x[0], robot.max_x[0]])
	ax.set_ylim([robot.min_x[1], robot.max_x[1]])
	ax.set_zlim([robot.min_x[2], robot.max_x[2]])

	# plot trajectory obtained by the optimizer with the stl models
	fig_model_traj = plt.figure()
	ax = mplot3d.Axes3D(fig_model_traj)

	quad_model = mesh.Mesh.from_file('physics/multirotor_models/crazyflie2.stl')
	colors = ["k","b","g","m"]

	T = len(solutions[list_of_solvers[0]].data[:,0])

	for i, solver_name in zip(range(nSolvers),list_of_solvers):
		sol = solutions[solver_name]
		ax.plot3D(sol.data[:, 0], sol.data[:, 1], sol.data[:, 2], colors[i] , linestyle = ':', linewidth = 0.5, markersize=4)

	if solutions[list_of_solvers[0]].intermediate_states is not None:
		numberOfIntermediateStates = len(solutions[list_of_solvers[0]].intermediate_states)
	else:
		numberOfIntermediateStates = 0

	for t in range(T):
		for i, solver_name in zip(range(nSolvers),list_of_solvers):
			sol = solutions[solver_name] #5, 10, 20
			if t%(T/5) == 0 or t == T-1 or t == 9 or t == 72:
				model = copy.deepcopy(quad_model)

				quat_rot = sol.data[t,6:10]
				rot_mat = rowan.to_matrix(quat_rot)
				pos = sol.data[t,:3]
				
				model.rotate_using_matrix(rot_mat)
				model.translate(pos)
				
				if obstacles is not None or numberOfIntermediateStates > 1:
					arrow_l = 0.2
					linewidth = 1.2
				elif numberOfIntermediateStates > 0:
					arrow_l = 0.12
					linewidth = 1.2
				else:
					arrow_l = 0.085
					linewidth = 1.2

				z_vec = np.vstack((pos, rowan.rotate(quat_rot, np.array([0, 0, arrow_l]))))

				if t == 0:
					surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "dodgerblue"))
					ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")
				elif t == T-1:
					surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "deeppink"))
					ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")
				else:
					surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = colors[i]))
					ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")

				surf._facecolors2d=surf._facecolor3d
				surf._edgecolors2d=surf._edgecolor3d


	if solutions[list_of_solvers[0]].intermediate_states is not None:
		for i_s in solutions[list_of_solvers[0]].intermediate_states:
			model = copy.deepcopy(quad_model)

			quat_rot = solutions[list_of_solvers[0]].data[i_s.timing,6:10]
			rot_mat = rowan.to_matrix(quat_rot)
			pos = solutions[list_of_solvers[0]].data[i_s.timing,:3]
			
			model.rotate_using_matrix(rot_mat)
			model.translate(pos)

			if obstacles is not None or numberOfIntermediateStates > 1:
				arrow_l = 0.2
				linewidth = 1.2
			elif numberOfIntermediateStates > 0:
				arrow_l = 0.12
				linewidth = 1.2
			else:
				arrow_l = 0.085
				linewidth = 1.2

			z_vec = np.vstack((pos, rowan.rotate(quat_rot, np.array([0, 0, arrow_l]))))

			surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "orange"))
			ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")

			surf._facecolors2d=surf._facecolor3d
			surf._edgecolors2d=surf._edgecolor3d

	if obstacles is not None:
		for obs in obstacles:
			draw_obs(obs.type, obs.shape, obs.pos, ax)

	set_axes_equal(ax)
	ax.set_ylabel("\ny [m]")
	ax.set_xlabel("\nx [m]")
	ax.set_zlabel("\nz [m]")

	init_s_color = mpatches.Patch(color='dodgerblue', label='initial state')
	final_s_color = mpatches.Patch(color='deeppink', label='final state')
	intermediate_s_color = mpatches.Patch(color='orange', label='intermediate state')
	if solutions[list_of_solvers[0]].intermediate_states is not None:
		ax.legend(handles=[init_s_color,final_s_color,intermediate_s_color])
	else:
		ax.legend(handles=[init_s_color,final_s_color])
	
	#plt.legend()

	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	zlim = ax.get_zlim()

	# plot the problem description

	fig_prob = plt.figure()
	ax = mplot3d.Axes3D(fig_prob)

	timings = [0,T-1]

	if solutions[list_of_solvers[0]].intermediate_states is not None:
		timings.extend([i_s.timing for i_s in solutions[list_of_solvers[0]].intermediate_states])

	for t in timings:
		model = copy.deepcopy(quad_model)

		quat_rot = solutions[list_of_solvers[0]].data[t,6:10]
		rot_mat = rowan.to_matrix(quat_rot)
		pos = solutions[list_of_solvers[0]].data[t,:3]
		
		model.rotate_using_matrix(rot_mat)
		model.translate(pos)

		if obstacles is not None or numberOfIntermediateStates > 1:
			arrow_l = 0.2
			linewidth = 1.2
		elif numberOfIntermediateStates > 0:
			arrow_l = 0.12
			linewidth = 1.2
		else:
			arrow_l = 0.085
			linewidth = 1.2

		z_vec = np.vstack((pos, rowan.rotate(quat_rot, np.array([0, 0, arrow_l]))))

		if t == 0:
			surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "dodgerblue"))
			ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")
		elif t == T-1:
			surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "deeppink"))
			ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")
		else:
			surf = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors, color = "orange"))
			ax.quiver(z_vec[0, 0], z_vec[0, 1], z_vec[0, 2], z_vec[1, 0], z_vec[1, 1], z_vec[1, 2], lw=linewidth , color="r")
			

		surf._facecolors2d=surf._facecolor3d
		surf._edgecolors2d=surf._edgecolor3d

	if obstacles is not None:
		for obs in obstacles:
			draw_obs(obs.type, obs.shape, obs.pos, ax)

	set_axes_equal(ax)

	#if solutions[list_of_solvers[0]].intermediate_states is not None:
	#	plt.legend(["initial state", "final state", "intermediate state"])
	#else:
	#	plt.legend(["initial state", "final state"])

	ax = plt.gca()
	ax.set_ylabel("y [m]")
	ax.set_xlabel("x [m]")
	ax.set_zlabel("z [m]")
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_zlim(zlim)
	
	
	# plot integration error
	if nSolvers <= 2:
		fig_int_err, axs = plt.subplots(1, nSolvers)
		if nSolvers == 1:
			axs = [axs]

	else:
		fig_int_err, axs = plt.subplots(1, nSolvers)

	for i,solver_name in zip(range(nSolvers),list_of_solvers):
		int_error = solutions[solver_name].int_err
		time_vec = time_points[solver_name][1:]

		axs[i].plot(time_vec, int_error)
		axs[i].set_title("integration error " + solver_name)
		axs[i].set_ylabel("integration error []")

		leg = ["$e_{I,p_x}$", "$e_{I,p_y}$", "$e_{I,p_z}$", "$e_{I,v_x}$", "$e_{I,v_y}$", "$e_{I,v_z}$",
			   "$e_{I,q_1}$", "$e_{I,q_2}$", "$e_{I,q_3}$", "$e_{I,q_4}$", "$e_{I,\omega_x}$","$e_{I,\omega_y}$",
			   "$e_{I,\omega_z}$"]

		axs[i].legend(leg)
		axs[i].set_xlabel("time [s]")

	

	# plot actions
	if nSolvers == 1:
		fig_ac, axs = plt.subplots(1, 1)
		axs = [axs]
	elif nSolvers == 2:
		fig_ac, axs = plt.subplots(1, 2)
		axs = [axs[0],axs[1]]
	elif nSolvers == 3:
		fig_ac, axs = plt.subplots(1, 3)
		axs = [axs[0],axs[1],axs[2]]
	else:
		fig_ac, axs = plt.subplots(2, 2)
		axs = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

	for i,solver_name in zip(range(nSolvers),list_of_solvers):
		actions = solutions[solver_name].data[:-1, 13:]
		time_vec = time_points[solver_name][:-1]

		axs[i].plot(time_vec, actions)
		if robot.nrMotors == 2:
			leg = [solver_name + ": \n $f_1$", "$f_2$"]
		elif robot.nrMotors == 3:
			leg = [solver_name + ": \n $f_1$", "$f_2$", "$f_3$"]
		elif robot.nrMotors == 4:
			leg = [solver_name + ": \n $f_1$", "$f_2$", "$f_3$", "$f_4$"]
		elif robot.nrMotors == 6:
			leg = [solver_name + ": \n $f_1$", "$f_2$", "$f_3$", "$f_4$", "$f_5$", "$f_6$"]
		elif robot.nrMotors == 8:
			leg = [solver_name + ": \n $f_1$", "$f_2$", "$f_3$", "$f_4$", "$f_5$", "$f_6$", "$f_7$", "$f_8$"]

		axs[i].legend(leg)
		axs[i].set_ylabel("forces [N]")
		inputCost = np.round((actions ** 2).sum(),3)
		axs[i].set_title("force comparision \n input cost: \n {}: {}".format(solver_name, inputCost))
		axs[i].set_xlabel("time [s]")


	"""
	# plot hessians
	if "CASADI" in list_of_solvers:
		fig_HCas = plt.figure()
		ply.spy(np.array(solutions["CASADI"].hessian))
		fig_HCas.savefig("plots/hessian_casadi.pdf",bbox_inches='tight')

	if "KOMO" in list_of_solvers:
		fig_HKom = plt.figure()
		ply.spy(np.array(solutions["KOMO"].hessian))
		fig_HKom.savefig("plots/hessian_komo.pdf",bbox_inches='tight')
	"""

	# safe figures
	plt.rcParams.update({
  		"text.usetex": False,
		})
	
	fig_states.savefig("plots/states.pdf",bbox_inches='tight')
	fig_traj.savefig("plots/trajectory.pdf",bbox_inches='tight')
	fig_int_err.savefig("plots/int_error.pdf",bbox_inches='tight')
	fig_ac.savefig("plots/actions.pdf",bbox_inches='tight')
	fig_model_traj.savefig("plots/model_traj.pdf",bbox_inches='tight')
	fig_prob.savefig("plots/prob.pdf",bbox_inches='tight')


def report_noise(prob_name, path, nr_motors, t2w_vec, noise_vec, robot_type, prob_setup):

	data_scvx_allProbs = []
	data_komo_allProbs = []
	time_komo_allProbs = []
	time_scvx_allProbs = []
	time_cvx_solver_allProbs = []
	time_cvxpy_allProbs = []
	nItr_scvx_allProbs = []
	succ_rate_komo_allProbs = []
	succ_rate_scvx_allProbs = []
	keys = []

	initial_guess_komo = []
	initial_guess_scvx = []

	int_error_komo = []
	int_error_scvx = []
	int_error_komo_small_step_size = []
	int_error_scvx_small_step_size = []

	check_komo_list = []
	check_scvx_list = []
	check_spec_list = []

	constr_viol_list = []

	for noise in noise_vec:
		for nrM in nr_motors:
			for t2w_fac in t2w_vec:

				data_komo_all = ou.load_object(path + prob_name + "_" + robot_type + "_nrM" + str(nrM) + "_t2w" + str(t2w_fac) + "_noise"+str(noise)+"_adaptiveWeights_False2_komo")
				data_scvx_all = ou.load_object(path + prob_name + "_" + robot_type + "_nrM" + str(nrM) + "_t2w" + str(t2w_fac) + "_noise"+str(noise)+"_adaptiveWeights_False2_scvx")

				# extract different data
				check_komo = data_komo_all["check"]
				check_spec = data_komo_all["check_spec"]
				check_scvx = data_scvx_all["check"]

				check_scvx_list.append(check_scvx)
				check_komo_list.append(check_komo)
				check_spec_list.append(check_spec)

				data_scvx = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["data"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])
				data_komo = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_komo_all["data"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])
				time_komo = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_komo_all["time"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])
				time_scvx = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["time"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])
				time_cvx_solver = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["cvx_time"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])
				time_cvxpy = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["cvxpy_time"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])

				nItr_scvx = np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["num_itr"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec])

				succ_rate_komo = sum(np.logical_and(check_komo, np.logical_not(check_spec))) / len(check_komo)
				succ_rate_scvx = sum(check_scvx) / len(check_scvx)

				initial_guess_komo.append([d for d in data_komo_all["init_guess"]])
				initial_guess_scvx.append([d for d in data_scvx_all["init_guess"]])

				int_error_scvx_small_step_size.append(np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["int_error_small_step_size"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec]))
				int_error_komo_small_step_size.append(np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_komo_all["int_error_small_step_size"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec]))

				int_error_scvx.append(np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_scvx_all["int_error"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec]))
				int_error_komo.append(np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_komo_all["int_error"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec]))

				constr_viol_list.append(np.array([d for (d, c_scvx, c_komo, c_spec) in zip(data_komo_all["constr_viol"], check_scvx, check_komo, check_spec) if c_scvx and c_komo and not c_spec]))

				data_scvx_allProbs.append(data_scvx)
				data_komo_allProbs.append(data_komo)
				time_komo_allProbs.append(time_komo)
				time_scvx_allProbs.append(time_scvx)
				time_cvx_solver_allProbs.append(time_cvx_solver)
				time_cvxpy_allProbs.append(time_cvxpy)
				nItr_scvx_allProbs.append(nItr_scvx)
				succ_rate_komo_allProbs.append(succ_rate_komo)
				succ_rate_scvx_allProbs.append(succ_rate_scvx)
				keys.append([str(nrM), str(t2w_fac), str(noise)])

	# plot success rate
	fig = plt.figure()
	ax = plt.gca()
	ax.set_prop_cycle(standard_cycler[:2])
	ticks = []
	for n, succ_rate_scvx, succ_rate_komo in zip(range(len(keys)), succ_rate_scvx_allProbs,
												 succ_rate_komo_allProbs):

		width = 0.35
		x = n

		ax.bar(x - width / 2, succ_rate_scvx, width)
		ax.bar(x + width / 2, succ_rate_komo, width)

		if len(noise_vec) >= 2 and len(t2w_vec) == 1 and len(nr_motors) == 1:
			ticks.append("noise \n factor: \n" + keys[n][2])
		else:
			ticks.append("$n_m$: " + keys[n][0] + ",\n $r_{t2w}$: " + keys[n][1])

	#ax.set_title(prob_name.replace("_", " "))  # ("num motors: " + keys[n][0] + ", t2w: " + keys[n][1])
	ax.set_ylabel("Success Rate in [\%]")
	ax.set_xticks(np.arange(len(keys)))
	ax.set_xticklabels(ticks)
	ax.yaxis.set_major_formatter(PercentFormatter(1))
	ax.legend(["SCvx", "KOMO"])

	if len(noise_vec) <= 2:

###########################################################################################################################
		
		# plot input cost
		fig = plt.figure()
		#fig.suptitle(prob_name.replace("_", " "))
		for n, data_scvx, data_komo in zip(range(len(keys)),data_scvx_allProbs, data_komo_allProbs):

			nTrials_scvx = data_scvx.shape[0]
			nTrials_komo = data_komo.shape[0]

			ax = fig.add_subplot(len(nr_motors),len(t2w_vec),1+n)

			if robot_type == "dI":
				inputCostKOMO = np.array([np.sum(data_komo[i, :-1, 6:] ** 2) for i in range(nTrials_komo)])
				inputCostSCVX = np.array([np.sum(data_scvx[i, :-1, 6:] ** 2) for i in range(nTrials_scvx)])
			else:
				inputCostKOMO = np.array([np.sum(data_komo[i, :-1, 13:] ** 2) for i in range(nTrials_komo)])
				inputCostSCVX = np.array([np.sum(data_scvx[i, :-1, 13:] ** 2) for i in range(nTrials_scvx)])


			s_w = np.empty(inputCostSCVX.shape)
			s_w.fill(1/inputCostSCVX.shape[0])
			
			k_w = np.empty(inputCostKOMO.shape)
			k_w.fill(1/inputCostKOMO.shape[0])
				
			ax.hist([np.round(inputCostSCVX,3),np.round(inputCostKOMO,3)], bins = 25, weights = [s_w, k_w]) 
			
			ax.legend(["SCvx", "KOMO"], title="$n_M$: " + keys[n][0] + ", $r_{t2w}$: " + keys[n][1], bbox_to_anchor=(1.1,1.01))

			if robot_type == "fM":
				if n == 1:
					ax.set_ylabel("Frequency")
				
				if n == 2:
					ax.set_xlabel("Optimal Value")
			else:
				ax.set_ylabel("Frequency")
				ax.set_xlabel("Optimal Value")

		
		if robot_type == "fM":
			fig.set_size_inches(3.5,4)
		else:
			fig.set_size_inches(3,4)
		plt.tight_layout()
			

###########################################################################################################################

		# plot time to solve
		fig = plt.figure()
		ax = plt.gca()
		ax.set_prop_cycle(standard_cycler[:4])
		ticks = []
		for n, time_komo, time_scvx, time_cvx_solver, time_cvxpy in zip(range(len(keys)),
																		time_komo_allProbs,
																		time_scvx_allProbs,
																		time_cvx_solver_allProbs,
																		time_cvxpy_allProbs):

			time_komo_std = np.std(time_komo)
			time_komo_mean = np.mean(time_komo)
			time_scvx_std = np.std(time_scvx - time_cvx_solver - time_cvxpy)
			time_scvx_mean = np.mean(time_scvx - time_cvx_solver - time_cvxpy)
			time_cvx_solver_std = np.std(time_cvx_solver)
			time_cvx_solver_mean = np.mean(time_cvx_solver)
			time_cvxpy_std = np.std(time_cvxpy)
			time_cvxpy_mean = np.mean(time_cvxpy)

			width = 0.35
			x = n

			ax.bar(x - width / 2, time_cvx_solver_mean, width, yerr=time_cvx_solver_std)#, label="Gurobi")
			ax.bar(x - width / 2, time_cvxpy_mean, width, yerr=time_cvxpy_std, bottom=time_cvx_solver_mean)#, label="CVXPY")
			ax.bar(x - width / 2, time_scvx_mean, width, yerr=time_scvx_std, bottom=time_cvx_solver_mean + time_cvxpy_mean)#, label="SCVX")
			ax.bar(x + width / 2, time_komo_mean, width, yerr=time_komo_std)#, label="KOMO")

			ticks.append("$n_M$: " + keys[n][0] + ",\n $r_{t2w}$: " + keys[n][1])

		#ax.set_title(prob_name.replace("_", " "))
		ax.set_ylabel("Time [s]")
		ax.set_xticks(np.arange(len(keys)))
		ax.set_xticklabels(ticks)
		ax.legend(["Gurobi", "CVXPY", "SCvx", "KOMO"])
		
		if robot_type == "fM":
			fig.set_size_inches(2.5,4)
		else:
			fig.set_size_inches(2.5,4)
		plt.tight_layout()

###########################################################################################################################

		if inputCostKOMO.all() and inputCostSCVX.all():
			# plot optimal solutions
			fig = plt.figure()

			#fig.suptitle(prob_name.replace("_", " "))
			for n, data_scvx, data_komo in zip(range(len(keys)),data_scvx_allProbs, data_komo_allProbs):

				nTrials_scvx = data_scvx.shape[0]
				nTrials_komo = data_komo.shape[0]

				if robot_type == "dI":
					inputCostKOMO = np.array([np.sum(data_komo[i, :-1, 6:] ** 2) for i in range(nTrials_komo)])
					inputCostSCVX = np.array([np.sum(data_scvx[i, :-1, 6:] ** 2) for i in range(nTrials_scvx)])
				else:
					inputCostKOMO = np.array([np.sum(data_komo[i, :-1, 13:] ** 2) for i in range(nTrials_komo)])
					inputCostSCVX = np.array([np.sum(data_scvx[i, :-1, 13:] ** 2) for i in range(nTrials_scvx)])

				idx_scvx = np.argmin(inputCostSCVX)
				idx_komo = np.argmin(inputCostKOMO)

				opt_data_scvx = data_scvx[idx_scvx]
				opt_data_komo = data_komo[idx_komo]

				ax = fig.add_subplot(len(t2w_vec),len(nr_motors), 1 + n, projection = "3d")

				ax.plot(opt_data_scvx[:,0], opt_data_scvx[:,1], opt_data_scvx[:,2])
				ax.plot(opt_data_komo[:,0], opt_data_komo[:,1], opt_data_komo[:,2])

				if robot_type == "fM":
					for t in range(data_scvx.shape[1]):
						if t%10 == 0:
							origin_scvx = opt_data_scvx[t, :3]
							origin_komo = opt_data_komo[t, :3]

							quat_scvx = opt_data_scvx[t, 6:10]
							quat_komo = opt_data_komo[t, 6:10]

							if "sphere" in prob_name:
								arrow_l = 0.4
								linewidth = 0.8
							else:
								arrow_l = 0.1
								linewidth = 0.6

							z_vec_scvx = np.vstack((origin_scvx, rowan.rotate(quat_scvx, np.array([0, 0, arrow_l]))))
							z_vec_komo = np.vstack((origin_komo, rowan.rotate(quat_komo, np.array([0, 0, arrow_l]))))

							ax.quiver(z_vec_scvx[0, 0], z_vec_scvx[0, 1], z_vec_scvx[0, 2],
									  z_vec_scvx[1, 0], z_vec_scvx[1, 1], z_vec_scvx[1, 2], lw=linewidth , color="r")#"#1f77b4")

							ax.quiver(z_vec_komo[0, 0], z_vec_komo[0, 1], z_vec_komo[0, 2],
									  z_vec_komo[1, 0], z_vec_komo[1, 1], z_vec_komo[1, 2], lw=linewidth , color="k")#"#ff7f0e")

				if prob_setup.obs is not None:
					for obs in prob_setup.obs:
						draw_obs(obs.type, obs.shape, obs.pos, ax)

				ax.set_ylabel("y [m]")
				ax.set_xlabel("x [m]")
				ax.set_zlabel("z [m]")
				ax.legend(["SCvx", "KOMO"], title="$n_M$: " + keys[n][0] + ", $r_{t2w}$: " + keys[n][1])
				set_axes_equal(ax)
			
			if robot_type == "fM":
				fig.set_size_inches(7,3.8)
			else:
				fig.set_size_inches(3.5,3.5)
			plt.margins(x=0)
			plt.margins(y=0)

###########################################################################################################################

		# plot integration error comparision with smaller stepsize for SCVX
		fig = plt.figure()
		#fig.suptitle("SCVX Integration Error with 5x Smaller Stepsize")
		width = 0.35

		for n, int_err_scvx_small_dt, int_err_scvx in zip(range(len(keys)), int_error_scvx_small_step_size, int_error_scvx):
			ax = fig.add_subplot(len(nr_motors), len(t2w_vec), 1 + n)
			nTrials = len(int_err_scvx)

			for tr in range(nTrials):
				x = tr
				ax.bar(x - width / 2, np.max(np.abs(int_err_scvx[tr])), width, color="#ff7f0e")
				ax.bar(x + width / 2, np.max(np.abs(int_err_scvx_small_dt[tr])), width, color="r")


			orig_dt = prob_setup.tf_max/prob_setup.t_steps_scvx

			ax.legend(["$\Delta t$: {}".format(round(orig_dt, 4)), "$\Delta t$: {}".format(round(orig_dt / 5, 4))],
					  title="$n_M$: " + keys[n][0] + ", $r_{t2w}$: " + keys[n][1])
					  
			#if n == 0 or n == 3 or n==6:
			ax.set_ylabel("Max. Integration Error")
			
			if n == 2:  
				ax.set_xlabel("Trial (Feasible)")

###########################################################################################################################

		# plot integration error comparision with smaller stepsize for KOMO
		fig = plt.figure()
		#fig.suptitle("KOMO Integration Error with 5x Smaller Stepsize")
		width = 0.35

		for n, int_err_komo_small_dt, int_err_komo, constr_viol_komo in zip(range(len(keys)), int_error_komo_small_step_size, int_error_komo, constr_viol_list):
			ax = fig.add_subplot(len(nr_motors), len(t2w_vec), 1 + n)
			nTrials = len(int_err_komo)

			for tr in range(nTrials):
				x = tr
				ax.bar(x - width, np.max(np.abs(int_err_komo[tr])), width, color="#1f77b4")
				ax.bar(x, np.max(np.abs(int_err_komo_small_dt[tr])), width, color="b")
				#ax.bar(x + width, constr_viol_komo[tr], width, color="c")


			orig_dt = prob_setup.tf_max / prob_setup.t_steps_komo

			ax.legend(["$\Delta t$: {}".format(round(orig_dt, 4)), "$\Delta t$: {}".format(round(orig_dt / 5, 4))],
					  title="$n_M$: " + keys[n][0] + ", $r_{t2w}$: " + keys[n][1])
					  
			#if n == 0 or n == 3 or n==6:
			ax.set_ylabel("Max. Integration Error")
			
			if n == 2: 
				ax.set_xlabel("Trial (Feasible)")

###########################################################################################################################
		# plot initial guess comparision

		nTrials = len(check_komo_list[0])
		init_comp_matrix = np.zeros([len(keys), nTrials])
		x_ticks = [str(i+1) for i in range(nTrials)]
		y_ticks = []

		#plt.figure()
		#ax = plt.axes(projection='3d')
		#g = True
		for n, c_scvx, c_komo, c_spec, init_guess in zip(range(len(keys)), check_scvx_list, check_komo_list, check_spec_list, initial_guess_scvx):

			for tr in range(nTrials):
				if c_scvx[tr] and (c_komo[tr] and not c_spec[tr]):
					init_comp_matrix[n, tr] = 0
					#if g == True:
					#	ax.plot3D(init_guess[tr][0][:, 0], init_guess[tr][0][:, 1], init_guess[tr][0][:, 2], 'r', marker='o', markersize=4)
					#	g = False
				elif c_scvx[tr] and not (c_komo[tr] and not c_spec[tr]):
					init_comp_matrix[n, tr] = 1
				elif not c_scvx[tr] and (c_komo[tr] and not c_spec[tr]):
					init_comp_matrix[n, tr] = 2
				elif not c_scvx[tr] and not (c_komo[tr] and not c_spec[tr]):
					init_comp_matrix[n, tr] = 3
					#ax.plot3D(init_guess[tr][0][:, 0], init_guess[tr][0][:, 1], init_guess[tr][0][:, 2], 'b', marker='o', markersize=4)

			y_ticks.append("num M: " + keys[n][0] + ", t2w: " + keys[n][1])


		cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
		cticks = ["SCvx: true \n KOMO: true",
				  "SCvx: true \n KOMO: false",
				  "SCvx: false \n KOMO: true",
				  "SCvx: false \n KOMO: false"]

		fig, ax = plt.subplots()
		plt.title("Comparision of Initial Guesses")
		cmesh = plt.pcolormesh(init_comp_matrix, cmap=cmap, vmin = 0, vmax = 4, edgecolor = "k")
		cbar = fig.colorbar(cmesh)
		cbar.set_ticks(ticks = [0.5,1.5,2.5,3.5])
		cbar.set_ticklabels(cticks)
		cbar.ax.set_title("feasibility")
		ax.set_xticks(np.arange(nTrials) + 0.5)
		ax.set_xticklabels(x_ticks)
		ax.set_yticks(np.arange(len(nr_motors)*len(t2w_vec)) + 0.5)
		ax.set_yticklabels(y_ticks)
		ax.set_xlabel("trials")


def report_all(prob_names, solver_names):

	success_obstacle_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	success_recovery_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	success_flip = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	success_loop = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}

	time_obstacle_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	time_recovery_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	time_flip = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	time_loop = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}

	cost_obstacle_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	cost_recovery_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	cost_flip = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	cost_loop = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}

	nIter_obstacle_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	nIter_recovery_flight = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	nIter_flip = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}
	nIter_loop = {"KOMO":[], "SCVX":[], "CASADI":[], "CROCO":[]}

	for p_n in prob_names:
		
		dat = {}
		for s_n in solver_names:
			dat[s_n] = ou.load_object("data/all_runs/" + p_n + "_4m_" + s_n)

		if "obstacle_flight" in p_n:
			for s_n in solver_names:
				success_obstacle_flight[s_n].append(dat[s_n].success)

				if dat[s_n].success:
					time_obstacle_flight[s_n].append(dat[s_n].time)

					actions = dat[s_n].data[:-1,13:]
					cost_val = np.round((actions ** 2).sum(),7)
					#cost_val = (actions ** 2).sum()
					cost_obstacle_flight[s_n].append(cost_val)
					nIter_obstacle_flight[s_n].append(dat[s_n].num_iter)

		elif "recovery_flight" in p_n:
			for s_n in solver_names:
				success_recovery_flight[s_n].append(dat[s_n].success)

				if dat[s_n].success:
					time_recovery_flight[s_n].append(dat[s_n].time)

					actions = dat[s_n].data[:-1,13:]
					cost_val = np.round((actions ** 2).sum(),7)
					#cost_val = (actions ** 2).sum()
					cost_recovery_flight[s_n].append(cost_val)
					nIter_recovery_flight[s_n].append(dat[s_n].num_iter)

		elif "flip" in p_n:
			for s_n in solver_names:
				success_flip[s_n].append(dat[s_n].success)

				if dat[s_n].success:
					time_flip[s_n].append(dat[s_n].time)

					actions = dat[s_n].data[:-1,13:]
					cost_val = np.round((actions ** 2).sum(),7)
					#cost_val = (actions ** 2).sum()
					cost_flip[s_n].append(cost_val)
					nIter_flip[s_n].append(dat[s_n].num_iter)

		elif "loop" in p_n:
			for s_n in solver_names:
				success_loop[s_n].append(dat[s_n].success)

				if dat[s_n].success:
					time_loop[s_n].append(dat[s_n].time)

					actions = dat[s_n].data[:-1,13:]
					cost_val = np.round((actions ** 2).sum(),7)
					#cost_val = (actions ** 2).sum()
					cost_loop[s_n].append(cost_val)
					nIter_loop[s_n].append(dat[s_n].num_iter)
		
		else:
			print("unknown problem")

	n_runs = int(len(prob_names)/4)

	solver_names_correct = {"KOMO":"KOMO",
			 				"SCVX":"DC (SCvx)",
							"CROCO":"DDP",
							"CASADI":"DC (CasADi)"}

	# Plot
	costs = [cost_obstacle_flight, cost_recovery_flight, cost_flip, cost_loop]
	times = [time_obstacle_flight, time_recovery_flight, time_flip, time_loop]
	nIter = [nIter_obstacle_flight, nIter_recovery_flight, nIter_flip, nIter_loop]

	all_dataframes = []
	for c,t,i in zip(costs,times,nIter):

		data_frames = []
		for s_n in solver_names:
			names = pd.DataFrame({"Solver":[solver_names_correct[s_n] for _ in range(n_runs)]})
			data_c = pd.DataFrame({"Cost":c[s_n]}).astype(float)
			data_t = pd.DataFrame({"Time [s]":t[s_n]}).astype(float)
			data_h = pd.DataFrame({"\# Iterations":i[s_n]}).astype(float)
			data_frames.append(pd.concat([data_c,data_t,data_h,names], axis = 1))

		df = pd.concat(data_frames,axis = 0)
		df.index = range(len(df.index))
		all_dataframes.append(df)

	plt.rc("axes", prop_cycle=fancy_cycler)	

	########################################################################################################################################################################
	# cost plots

	SMALL_SIZE = 19#18#12
	MEDIUM_SIZE = 24#15#15
	BIGGER_SIZE = 24#18#18

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	
	kwargs = {"element":"step"}

	#print(all_dataframes[0].to_string())
	#print(all_dataframes[1].to_string())
	#print(all_dataframes[2].to_string())
	#print(all_dataframes[3].to_string())

	plt.figure()
	sns.displot(all_dataframes[0], x = "Cost", hue="Solver", kind="hist", stat="percent", common_norm = False, legend = False, **kwargs)
	plt.xlabel("Cost")
	plt.legend([], [], frameon=False)
	fig_width, fig_height = plt.gcf().get_size_inches()
	plt.gcf().set_size_inches(fig_width, fig_height*0.7)
	plt.tight_layout()

	plt.savefig("plots/cost_s1.pdf")

	plt.figure()
	sns.displot(all_dataframes[1], x = "Cost", hue="Solver", kind="hist", stat="percent", common_norm = False, legend = False, **kwargs)
	plt.xlabel("Cost")
	KOMO_leg = mpatches.Patch(color='dodgerblue', label=solver_names_correct['KOMO'])
	SCVX_leg = mpatches.Patch(color='orange', label=solver_names_correct['SCVX'])
	CASADI_leg = mpatches.Patch(color='deeppink', label=solver_names_correct['CASADI'])
	CROCO_leg = mpatches.Patch(color='limegreen', label=solver_names_correct['CROCO'])
	ax = plt.gca()
	ax.legend(handles=[KOMO_leg,SCVX_leg,CASADI_leg,CROCO_leg])
	fig_width, fig_height = plt.gcf().get_size_inches()
	plt.gcf().set_size_inches(fig_width, fig_height*0.7)
	plt.tight_layout()

	plt.savefig("plots/cost_s2.pdf")

	plt.figure()
	sns.displot(all_dataframes[2], x = "Cost", hue="Solver", kind="hist", stat="percent", common_norm = False, legend = False, **kwargs)
	plt.xlabel("Cost")
	plt.legend([], [], frameon=False)
	#plt.legend([solver_names_correct[solver_names[0]],solver_names_correct[solver_names[1]],solver_names_correct[solver_names[2]],solver_names_correct[solver_names[3]]])
	fig_width, fig_height = plt.gcf().get_size_inches()
	plt.gcf().set_size_inches(fig_width, fig_height*0.7)
	plt.tight_layout()

	plt.savefig("plots/cost_s4.pdf")

	plt.figure()
	sns.displot(all_dataframes[3], x = "Cost", hue="Solver", kind="hist", stat="percent", common_norm = False, legend = False, **kwargs)
	plt.xlabel("Cost")
	plt.legend([], [], frameon=False)
	fig_width, fig_height = plt.gcf().get_size_inches()
	plt.gcf().set_size_inches(fig_width, fig_height*0.7)
	plt.tight_layout()

	plt.savefig("plots/cost_s3.pdf")

	########################################################################################################################################################################
	# time plots

	print("scenario: 1")
	print("KOMO: mean: {}, std: {}".format(np.mean(all_dataframes[0]["Time [s]"][:30]), np.std(all_dataframes[0]["Time [s]"][:30])))
	print("SCVX: mean: {}, std: {}".format(np.mean(all_dataframes[0]["Time [s]"][30:60]), np.std(all_dataframes[0]["Time [s]"][30:60])))
	print("CASADI: mean: {}, std: {}".format(np.mean(all_dataframes[0]["Time [s]"][60:90]), np.std(all_dataframes[0]["Time [s]"][60:90])))
	print("CROCO: mean: {}, std: {}".format(np.mean(all_dataframes[0]["Time [s]"][90:120]), np.std(all_dataframes[0]["Time [s]"][90:120])))

	print("scenario: 2")
	print("KOMO: mean: {}, std: {}".format(np.mean(all_dataframes[1]["Time [s]"][:30]), np.std(all_dataframes[1]["Time [s]"][:30])))
	print("SCVX: mean: {}, std: {}".format(np.mean(all_dataframes[1]["Time [s]"][30:60]), np.std(all_dataframes[1]["Time [s]"][30:60])))
	print("CASADI: mean: {}, std: {}".format(np.mean(all_dataframes[1]["Time [s]"][60:90]), np.std(all_dataframes[1]["Time [s]"][60:90])))
	print("CROCO: mean: {}, std: {}".format(np.mean(all_dataframes[1]["Time [s]"][90:120]), np.std(all_dataframes[1]["Time [s]"][90:120])))

	print("scenario: 3")
	print("KOMO: mean: {}, std: {}".format(np.mean(all_dataframes[2]["Time [s]"][:30]), np.std(all_dataframes[2]["Time [s]"][:30])))
	print("SCVX: mean: {}, std: {}".format(np.mean(all_dataframes[2]["Time [s]"][30:60]), np.std(all_dataframes[2]["Time [s]"][30:60])))
	print("CASADI: mean: {}, std: {}".format(np.mean(all_dataframes[2]["Time [s]"][60:90]), np.std(all_dataframes[2]["Time [s]"][60:90])))
	print("CROCO: mean: {}, std: {}".format(np.mean(all_dataframes[2]["Time [s]"][90:120]), np.std(all_dataframes[2]["Time [s]"][90:120])))

	print("scenario: 4")
	print("KOMO: mean: {}, std: {}".format(np.mean(all_dataframes[3]["Time [s]"][:30]), np.std(all_dataframes[3]["Time [s]"][:30])))
	print("SCVX: mean: {}, std: {}".format(np.mean(all_dataframes[3]["Time [s]"][30:60]), np.std(all_dataframes[3]["Time [s]"][30:60])))
	print("CASADI: mean: {}, std: {}".format(np.mean(all_dataframes[3]["Time [s]"][60:90]), np.std(all_dataframes[3]["Time [s]"][60:90])))
	print("CROCO: mean: {}, std: {}".format(np.mean(all_dataframes[3]["Time [s]"][90:120]), np.std(all_dataframes[3]["Time [s]"][90:120])))

	SMALL_SIZE = 26#18#12
	MEDIUM_SIZE = 28#15#15
	BIGGER_SIZE = 28#18#18

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	
	fig = plt.figure()
	#sns.histplot(all_dataframes[0], x = "time", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[0], x = "Time [s]", y = "Solver")
	plt.xlabel("Time [s]")
	plt.ylabel(" ")
	plt.xscale('log') 
	ax = plt.gca()
	ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.2, .4, .6, .8)))

	#plt.legend([], [], frameon=False)
	plt.tight_layout()
	
	fig.savefig("plots/time_s1.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[1], x = "time", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[1], x = "Time [s]", y = "Solver")
	plt.xlabel("Time [s]")
	plt.ylabel(" ")
	plt.xscale('log') 
	#plt.legend([], [], frameon=False)
	plt.tight_layout()

	fig.savefig("plots/time_s2.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[2], x = "time", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[2], x = "Time [s]", y = "Solver")
	plt.xlabel("Time [s]")
	plt.ylabel(" ")
	plt.xscale('log') 
	#plt.legend(solver_names)
	plt.tight_layout()

	fig.savefig("plots/time_s4.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[3], x = "time", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[3], x = "Time [s]", y = "Solver")
	plt.xlabel("Time [s]")
	plt.ylabel(" ")
	plt.xscale('log') 
	ax = plt.gca()
	ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.2, .4, .6, .8)))
	#plt.legend([], [], frameon=False)
	plt.tight_layout()

	fig.savefig("plots/time_s3.pdf")


	########################################################################################################################################################################
	# evals plots
	
	fig = plt.figure()
	#sns.histplot(all_dataframes[0], x = "hEvals", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[0], x = "\# Iterations", y = "Solver")
	plt.xlabel("\# Iterations")
	plt.ylabel(" ")
	plt.xscale('log') 
	#plt.legend([], [], frameon=False)
	plt.tight_layout()

	fig.savefig("plots/nIter_s1.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[1], x = "hEvals", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[1], x = "\# Iterations", y = "Solver")
	plt.xlabel("\# Iterations")
	plt.ylabel(" ")
	plt.xscale('log') 
	plt.xlim([100,1500])
	#plt.legend([], [], frameon=False)
	plt.tight_layout()

	fig.savefig("plots/nIter_s2.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[2], x = "hEvals", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[2], x = "\# Iterations", y = "Solver")
	plt.xlabel("\# Iterations")
	plt.ylabel(" ")
	plt.xscale('log') 
	plt.xlim([250,1500])
	#print(plt.xticks)
	#plt.legend(solver_names)
	plt.tight_layout()
	plt.xticks([300,400,600,1000], [r'$3 \times 10^2$','','', '$10^3$'])  # Set text labels.

	fig.savefig("plots/nIter_s4.pdf")

	fig = plt.figure()
	#sns.histplot(all_dataframes[3], x = "hEvals", hue="Solver", element="step", stat="percent",  bins = 15, legend = False)
	sns.boxplot(all_dataframes[3], x = "\# Iterations", y = "Solver")
	plt.xlabel("\# Iterations")
	plt.ylabel(" ")
	plt.xscale('log') 
	plt.xlim([40,1000])
	#plt.legend([], [], frameon=False)
	plt.tight_layout()

	fig.savefig("plots/nIter_s3.pdf")



	########################################################################################################################################################################
	# joined plots

	jp_obs = sns.jointplot(all_dataframes[0], y='Time [s]', x='Cost', kind="hist", hue='Solver', ratio=2, marginal_kws={'multiple': 'layer', "element":"step", "stat":"percent"})
	jp_obs.ax_joint.set_ylabel("Time [s]")
	jp_obs.ax_joint.set_xlabel("Cost")
	jp_obs.ax_joint.legend([], [], frameon=False)

	fig = jp_obs._figure
	fig.savefig("plots/time_cost_s1.pdf")

	jp_obs = sns.jointplot(all_dataframes[1], y='Time [s]', x='Cost', kind="hist", hue='Solver', ratio=2, marginal_kws={'multiple': 'layer', "element":"step", "stat":"percent"})
	jp_obs.ax_joint.set_ylabel("Time [s]")
	jp_obs.ax_joint.set_xlabel("Cost")
	jp_obs.ax_joint.legend([], [], frameon=False)

	fig = jp_obs._figure
	fig.savefig("plots/time_cost_s2.pdf")

	jp_obs = sns.jointplot(all_dataframes[2], y='Time [s]', x='Cost', kind="hist", hue='Solver', ratio=2, marginal_kws={'multiple': 'layer', "element":"step", "stat":"percent"})
	jp_obs.ax_joint.set_ylabel("Time [s]")
	jp_obs.ax_joint.set_xlabel("Cost")
	#jp_obs.ax_joint.legend([])

	fig = jp_obs._figure
	fig.savefig("plots/time_cost_s4.pdf")

	jp_obs = sns.jointplot(all_dataframes[3], y='Time [s]', x='Cost', kind="hist", hue='Solver', ratio=2, marginal_kws={'multiple': 'layer', "element":"step", "stat":"percent"})
	jp_obs.ax_joint.set_ylabel("Time [s]")
	jp_obs.ax_joint.set_xlabel("Cost")
	jp_obs.ax_joint.legend([], [], frameon=False)
	

	fig = jp_obs._figure
	fig.savefig("plots/time_cost_s3.pdf")


	########################################################################################################################################################################
	# success rates

	print("sr KOMO obstacle_flight: ", len(cost_obstacle_flight["KOMO"])/n_runs)
	print("sr SCVX obstacle_flight: ", len(cost_obstacle_flight["SCVX"])/n_runs)
	print("sr CASADI obstacle_flight: ", len(cost_obstacle_flight["CASADI"])/n_runs)
	print("sr CROCO obstacle_flight: ", len(cost_obstacle_flight["CROCO"])/n_runs)

	print("sr KOMO recovery_flight: ", len(cost_recovery_flight["KOMO"])/n_runs)
	print("sr SCVX recovery_flight: ", len(cost_recovery_flight["SCVX"])/n_runs)
	print("sr CASADI recovery_flight: ", len(cost_recovery_flight["CASADI"])/n_runs)
	print("sr CROCO recovery_flight: ", len(cost_recovery_flight["CROCO"])/n_runs)

	print("sr KOMO flip: ", len(cost_flip["KOMO"])/n_runs)
	print("sr SCVX flip: ", len(cost_flip["SCVX"])/n_runs)
	print("sr CASADI flip: ", len(cost_flip["CASADI"])/n_runs)
	print("sr CROCO flip: ", len(cost_flip["CROCO"])/n_runs)

	print("sr KOMO loop: ", len(cost_loop["KOMO"])/n_runs)
	print("sr SCVX loop: ", len(cost_loop["SCVX"])/n_runs)
	print("sr CASADI loop: ", len(cost_loop["CASADI"])/n_runs)
	print("sr CROCO loop: ", len(cost_loop["CROCO"])/n_runs)

	"""
	fig, axs = plt.subplots(2,2)
	ax_all = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
	for d,a in zip(all_dataframes,ax_all):
		sns.histplot(d, x = "cost", hue="Solver", element="step", stat="percent",  bins = 15, legend = False, ax = a)
	axs[0,0].set_title("Scenario 1")
	axs[0,0].set_xlabel(" ")
	axs[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	axs[0,1].set_title("Scenario 2")
	axs[0,1].set_ylabel(" ")
	axs[0,1].set_xlabel(" ")
	axs[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	axs[1,0].set_title("Scenario 3")
	axs[1,0].set_xlabel("Cost")
	axs[1,0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	axs[1,1].set_title("Scenario 4")
	axs[1,1].set_xlabel("Cost")
	axs[1,1].set_ylabel(" ")
	axs[1,1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	axs[1,1].legend(solver_names)

	plt.tight_layout()
	fig.savefig("plots/costs_all.pdf")
	plt.clf()

	fig, axs = plt.subplots(2,2)
	ax_all = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
	for d,a in zip(all_dataframes,ax_all):
		sns.boxplot(d, x = "Solver", y = "time", ax = a)

	axs[0,0].set_title("Scenario 1")
	axs[0,0].set_ylabel("Time [s]")
	axs[0,0].set_xlabel(" ")
	axs[0,1].set_title("Scenario 2")
	axs[0,1].set_ylabel(" ")
	axs[0,1].set_xlabel(" ")
	axs[1,0].set_title("Scenario 3")
	axs[1,0].set_ylabel("Time [s]")
	axs[1,0].set_xlabel("Solver")
	axs[1,1].set_title("Scenario 4")
	axs[1,1].set_ylabel(" ")
	axs[1,1].set_xlabel("Solver")

	plt.tight_layout()
	fig.savefig("plots/time_all.pdf")
	plt.clf()




	kwargs = {} #dict(kde_kws={'linewidth':2})

	# make pandas dataframes for better handling with seaborn
	cost_sf_df = pd.concat([pd.DataFrame({"KOMO":cost_simple_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":cost_simple_flight["SCVX"]}),
							pd.DataFrame({"CASADI": cost_simple_flight["CASADI"]})], axis = 1)
	
	cost_of_df = pd.concat([pd.DataFrame({"KOMO":cost_obstacle_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":cost_obstacle_flight["SCVX"]}),
							pd.DataFrame({"CASADI": cost_obstacle_flight["CASADI"]})], axis = 1)
	cost_rf_df = pd.concat([pd.DataFrame({"KOMO":cost_recovery_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":cost_recovery_flight["SCVX"]}),
							pd.DataFrame({"CASADI": cost_recovery_flight["CASADI"]})], axis = 1)
	cost_f_df = pd.concat([pd.DataFrame({"KOMO":cost_flip["KOMO"]}),
			    		   pd.DataFrame({"SCVX":cost_flip["SCVX"]}),
						   pd.DataFrame({"CASADI": cost_flip["CASADI"]})], axis = 1)
	
	time_sf_df = pd.concat([pd.DataFrame({"KOMO":time_simple_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":time_simple_flight["SCVX"]}),
							pd.DataFrame({"CASADI": time_simple_flight["CASADI"]})], axis = 1)
	time_of_df = pd.concat([pd.DataFrame({"KOMO":time_obstacle_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":time_obstacle_flight["SCVX"]}),
							pd.DataFrame({"CASADI": time_obstacle_flight["CASADI"]})], axis = 1)
	time_rf_df = pd.concat([pd.DataFrame({"KOMO":time_recovery_flight["KOMO"]}),
			    			pd.DataFrame({"SCVX":time_recovery_flight["SCVX"]}),
							pd.DataFrame({"CASADI": time_recovery_flight["CASADI"]})], axis = 1)
	time_f_df = pd.concat([pd.DataFrame({"KOMO":time_flip["KOMO"]}),
			    		   pd.DataFrame({"SCVX":time_flip["SCVX"]}),
						   pd.DataFrame({"CASADI": time_flip["CASADI"]})], axis = 1)

	# make pandas dataframes for better handling with seaborn
	cost_sf_df = pd.DataFrame({"KOMO":cost_simple_flight["KOMO"],
			    			   "SCVX":cost_simple_flight["SCVX"],
							   "CASADI": cost_simple_flight["CASADI"]})
	cost_of_df = pd.DataFrame({"KOMO":cost_obstacle_flight["KOMO"],
			    			   "SCVX":cost_obstacle_flight["SCVX"],
							   "CASADI": cost_obstacle_flight["CASADI"]})
	cost_rf_df = pd.DataFrame({"KOMO":cost_recovery_flight["KOMO"],
			    			   "SCVX":cost_recovery_flight["SCVX"],
							   "CASADI": cost_recovery_flight["CASADI"]})
	cost_f_df = pd.DataFrame({"KOMO":cost_flip["KOMO"],
			    			   "SCVX":cost_flip["SCVX"],
							   "CASADI": cost_flip["CASADI"]})
	
	time_sf_df = pd.DataFrame({"KOMO":time_simple_flight["KOMO"],
			    			   "SCVX":time_simple_flight["SCVX"],
							   "CASADI": time_simple_flight["CASADI"]})
	time_of_df = pd.DataFrame({"KOMO":time_obstacle_flight["KOMO"],
			    			   "SCVX":time_obstacle_flight["SCVX"],
							   "CASADI": time_obstacle_flight["CASADI"]})
	time_rf_df = pd.DataFrame({"KOMO":time_recovery_flight["KOMO"],
			    			   "SCVX":time_recovery_flight["SCVX"],
							   "CASADI": time_recovery_flight["CASADI"]})
	time_f_df = pd.DataFrame({"KOMO":time_flip["KOMO"],
			    			   "SCVX":time_flip["SCVX"],
							   "CASADI": time_flip["CASADI"]})

	

	fig, axs = plt.subplots(2,2)
	sns.barplot(time_sf_df, errorbar = "sd", ax = axs[0,0])
	sns.barplot(time_of_df, errorbar = "sd", ax = axs[0,1])
	sns.barplot(time_rf_df, errorbar = "sd", ax = axs[1,0])
	sns.barplot(time_f_df, errorbar = "sd", ax = axs[1,1])

	plt.legend()
	fig.savefig("plots/time_all.pdf")
	plt.clf()





	fig, axs = plt.subplots(2,2)

	# plot costs
	# plot simple flight costs 
	sns.histplot(cost_simple_flight["KOMO"], element="step", color="dodgerblue", label="KOMO", ax = axs[0,0], kde = True, **kwargs)
	sns.histplot(cost_simple_flight["SCVX"], element="step", color="orange", label="SCVX", ax = axs[0,1], kde = True, **kwargs)
	sns.histplot(cost_simple_flight["CASADI"], element="step", color="deeppink", ax = axs[1,0], label="CASADI", kde = True, **kwargs)

	plt.legend()
	fig.savefig("plots/costs_simple.pdf")
	plt.clf()

	fig, axs = plt.subplots(2,2)

	# plot costs
	# plot simple flight costs 
	sns.histplot(cost_obstacle_flight["KOMO"], element="step", color="dodgerblue", label="KOMO", ax = axs[0,0], kde = True, **kwargs)
	sns.histplot(cost_obstacle_flight["SCVX"], element="step", color="orange", label="SCVX", ax = axs[0,1], kde = True, **kwargs)
	sns.histplot(cost_obstacle_flight["CASADI"], element="step", color="deeppink", ax = axs[1,0], label="CASADI", kde = True, **kwargs)

	plt.legend()
	fig.savefig("plots/costs_obstacle.pdf")
	plt.clf()

	fig, axs = plt.subplots(2,2)

	# plot costs
	# plot simple flight costs 
	sns.histplot(cost_recovery_flight["KOMO"], element="step", color="dodgerblue", label="KOMO", ax = axs[0,0], kde = True, **kwargs)
	sns.histplot(cost_recovery_flight["SCVX"], element="step", color="orange", label="SCVX", ax = axs[0,1], kde = True, **kwargs)
	sns.histplot(cost_recovery_flight["CASADI"], element="step", color="deeppink", ax = axs[1,0], label="CASADI", kde = True, **kwargs)

	plt.legend()
	fig.savefig("plots/costs_recovery.pdf")
	plt.clf()

	fig, axs = plt.subplots(2,2)

	# plot costs
	# plot simple flight costs 
	sns.histplot(cost_flip["KOMO"], element="step", color="dodgerblue", label="KOMO", ax = axs[0,0], kde = True, **kwargs)
	sns.histplot(cost_flip["SCVX"], element="step", color="orange", label="SCVX", ax = axs[0,1], kde = True, **kwargs)
	sns.histplot(cost_flip["CASADI"], element="step", color="deeppink", ax = axs[1,0], label="CASADI", kde = True, **kwargs)

	plt.legend()
	fig.savefig("plots/costs_flip.pdf")
	plt.clf()
	"""
