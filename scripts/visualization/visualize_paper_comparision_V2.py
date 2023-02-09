import numpy as np
from numpy import genfromtxt
from visualization.geom_visualization import draw_obs
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

# set line cycler
plt.rc("axes", prop_cycle=standard_cycler)

# set latex font
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Computer Modern Roman",
})

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_comparision(x_hist_p,
                     u_hist_p,
                     sig_hist_p,
                     p_hist_p,
                     vd_hist_p,
                     vs_hist_p,
                     vic_hist_p,
                     vtc_hist_p,
                     l_cost_hist_p,
                     nl_cost_hist_p,
                     eta_hist_p,
                     roh_hist_p,
                     T,
                     nitr,
                     nstates,
                     nactions,
                     x0,xf,
                     CHandler):

    data = "obs_sphere_no_timeopt_trustregion_1"

    # data extraction for julia solution
    x_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/x_opt.csv", delimiter = ',', skip_header=0)
    u_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/u_opt.csv", delimiter = ',', skip_header=0)
    p_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/p_opt.csv", delimiter = ',', skip_header=0)
    vd_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/vd_opt.csv", delimiter = ',', skip_header=0)
    vs_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/vs_opt.csv", delimiter = ',', skip_header=0)
    vic_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/vic_opt.csv", delimiter = ',', skip_header=0)
    vtc_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/vtc_opt.csv", delimiter = ',', skip_header=0)
    l_cost_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/L_opt.csv", delimiter=',', skip_header=0)
    nl_cost_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/L_aug_opt.csv", delimiter=',', skip_header=0)
    eta_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/eta_opt.csv", delimiter=',', skip_header=0)
    roh_csv = genfromtxt("data/julia_data/scvx_julia_data_" + data + "/roh_opt.csv", delimiter=',', skip_header=0)

    x_hist_j = np.zeros((nitr,T,nstates))
    u_hist_j = np.zeros((nitr,T,nactions))
    sig_hist_j = np.zeros((nitr, T))
    p_hist_j = np.zeros((nitr,1,1))
    vd_hist_j = np.zeros((nitr,T-1,nstates))
    vs_hist_j = np.zeros((nitr,T,1))
    vic_hist_j = np.zeros((nitr,1,nstates))
    vtc_hist_j = np.zeros((nitr,1,nstates))
    l_cost_hist_j = np.zeros((nitr, 1, 1))
    nl_cost_hist_j = np.zeros((nitr, 1, 1))
    eta_hist_j = np.zeros((nitr, 1, 1))
    roh_hist_j = np.zeros((nitr, 1, 1))

    for itr in range(nitr):
        x_hist_j[itr] = x_csv[itr * nstates : (itr+1) * nstates].T
        u_hist_j[itr] = u_csv[itr * (nactions+1) : (itr) * (nactions+1)+nactions].T
        sig_hist_j[itr] = u_csv[(itr)*(nactions+1) + nactions].T
        p_hist_j[itr] = p_csv[itr:(itr + 1)].T
        if itr < nitr:
            vd_hist_j[itr] = vd_csv[itr * nstates:(itr + 1) * nstates].T
        vs_hist_j[itr] = vs_csv[itr:(itr + 1)].T
        vic_hist_j[itr] = vic_csv[itr * nstates:(itr + 1) * nstates].T
        vtc_hist_j[itr] = vtc_csv[itr * nstates:(itr + 1) * nstates].T
        l_cost_hist_j[itr] = l_cost_csv[itr:(itr + 1)].T
        nl_cost_hist_j[itr] = nl_cost_csv[itr:(itr + 1)].T
        eta_hist_j[itr] = eta_csv[itr:(itr + 1)].T
        roh_hist_j[itr] = roh_csv[itr:(itr + 1)].T

    # plot trajectories
    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(nitr):
        ax.plot3D(x_hist_p[i,:, 0], x_hist_p[i,:, 1], x_hist_p[i,:, 2], color = "#1f77b4", marker='o', markersize=3)
        ax.plot3D(x_hist_j[i,:, 0], x_hist_j[i,:, 1], x_hist_j[i,:, 2], color = "#ff7f0e", marker='o', markersize=3)
    ax.plot3D(x_hist_p[nitr, :, 0], x_hist_p[nitr, :, 1], x_hist_p[nitr, :, 2], color = "#1f77b4", marker='o', markersize=3)

    ax.scatter3D(x0[0], x0[1], x0[2], color='k')
    ax.scatter3D(xf[0], xf[1], xf[2], color='k')
    if CHandler.obs_data:
        for obs in CHandler.obs_data:
            draw_obs(obs.type, obs.shape, obs.pos, ax)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.legend(["Python","Julia"])

    fig, axs = plt.subplots(2,2)

    # plot convergence
    X_p = np.concatenate((x_hist_p[1:], u_hist_p[1:], np.repeat(p_hist_p[1:,:,np.newaxis],T,axis=1)),2)
    X_opt_p = X_p[-1]
    X_j = np.concatenate((x_hist_j, u_hist_j, np.repeat(p_hist_j,T,axis=1)), 2)
    X_opt_j = X_j[-1]

    dist_to_opt_p = np.array([np.linalg.norm(Xi - X_opt_p, 2) for Xi in X_p[:-1]])
    dist_to_opt_j = np.array([np.linalg.norm(Xi - X_opt_j, 2) for Xi in X_j[:-1]])

    axs[0,0].plot(dist_to_opt_p / np.linalg.norm(X_opt_p, 2),color="#1f77b4", marker='o', markersize=3)
    axs[0,0].plot(dist_to_opt_j / np.linalg.norm(X_opt_j, 2), color="#ff7f0e", marker='o', markersize=3)
    
    axs[0,0].set_ylabel("Dist. to Optimal Solution ($\Delta_x$)")
    axs[0,0].set_xlabel("\# Iteration")
    
    axs[0,0].set_yscale("log")
    axs[0, 0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # plot roh
    axs[0, 1].plot(roh_hist_p, color="#1f77b4", marker='o', markersize=3)
    axs[0, 1].plot(np.squeeze(roh_hist_j), color="#ff7f0e", marker='o', markersize=3)
    axs[0, 1].set_ylabel("Approx. Accuracy ($\\rho$)")
    axs[0, 1].set_xlabel("\# Iteration")
    axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0, 1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # plot cost history
    # nl cost
    axs[1, 0].plot(nl_cost_hist_p[:], color="#1f77b4", marker='o', markersize=3)
    axs[1, 0].plot(nl_cost_hist_j[:,0,0], color="#ff7f0e", marker='o', markersize=3)
    axs[1, 0].set_ylabel("Non-Convex Cost ($\mathcal{J}_{\lambda}$)")
    axs[1, 0].set_xlabel("\# Iteration")
    #axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1, 0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    
    # plot eta
    axs[1, 1].plot(eta_hist_p[:-1], color="#1f77b4", marker='o', markersize=3)
    axs[1, 1].plot(np.squeeze(eta_hist_j), color="#ff7f0e", marker='o', markersize=3)
    axs[1, 1].set_ylabel("Trust Region Radius ($\eta$)")
    axs[1, 1].set_xlabel("\# Iteration")
    axs[1, 1].xaxis.set_major_formatter(FormatStrFormatter('%d'))


    plt.legend(["Python", "Julia"])

    plt.show()

def plot_comparision_own_const(x_hist_p,
                     u_hist_p,
                     p_hist_p,
                     vd_hist_p,
                     vs_hist_p,
                     vic_hist_p,
                     vtc_hist_p,
                     l_cost_hist_p,
                     nl_cost_hist_p,
                     eta_hist_p,
                     roh_hist_p,
                     T,
                     nitr,
                     nstates,
                     nactions,
                     x0,xf,
                     CHandler):

    data = "obs_sphere_no_timeopt_trustregion_1"

    # data extraction for julia solution
    x_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/x_opt.csv", delimiter = ',', skip_header=0)
    u_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/u_opt.csv", delimiter = ',', skip_header=0)
    p_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/p_opt.csv", delimiter = ',', skip_header=0)
    vd_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/vd_opt.csv", delimiter = ',', skip_header=0)
    vs_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/vs_opt.csv", delimiter = ',', skip_header=0)
    vic_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/vic_opt.csv", delimiter = ',', skip_header=0)
    vtc_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/vtc_opt.csv", delimiter = ',', skip_header=0)
    l_cost_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/L_opt.csv", delimiter=',', skip_header=0)
    nl_cost_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/L_aug_opt.csv", delimiter=',', skip_header=0)
    eta_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/eta_opt.csv", delimiter=',', skip_header=0)
    roh_csv = genfromtxt("julia_data/scvx_julia_data_" + data + "/roh_opt.csv", delimiter=',', skip_header=0)

    x_hist_j = np.zeros((nitr,T,nstates))
    u_hist_j = np.zeros((nitr,T,nactions))
    sig_hist_j = np.zeros((nitr, T))
    p_hist_j = np.zeros((nitr,1,1))
    vd_hist_j = np.zeros((nitr,T-1,nstates))
    vs_hist_j = np.zeros((nitr,T,1))
    vic_hist_j = np.zeros((nitr,1,nstates))
    vtc_hist_j = np.zeros((nitr,1,nstates))
    l_cost_hist_j = np.zeros((nitr, 1, 1))
    nl_cost_hist_j = np.zeros((nitr, 1, 1))
    eta_hist_j = np.zeros((nitr, 1, 1))
    roh_hist_j = np.zeros((nitr, 1, 1))

    for itr in range(nitr):
        x_hist_j[itr] = x_csv[itr * nstates : (itr+1) * nstates].T
        u_hist_j[itr] = u_csv[itr * (nactions+1) : (itr) * (nactions+1)+nactions].T
        sig_hist_j[itr] = u_csv[(itr)*(nactions+1) + nactions].T
        p_hist_j[itr] = p_csv[itr:(itr + 1)].T
        if itr < nitr:
            vd_hist_j[itr] = vd_csv[itr * nstates:(itr + 1) * nstates].T
        vs_hist_j[itr] = vs_csv[itr:(itr + 1)].T
        vic_hist_j[itr] = vic_csv[itr * nstates:(itr + 1) * nstates].T
        vtc_hist_j[itr] = vtc_csv[itr * nstates:(itr + 1) * nstates].T
        l_cost_hist_j[itr] = l_cost_csv[itr:(itr + 1)].T
        nl_cost_hist_j[itr] = nl_cost_csv[itr:(itr + 1)].T
        eta_hist_j[itr] = eta_csv[itr:(itr + 1)].T
        roh_hist_j[itr] = roh_csv[itr:(itr + 1)].T

    # plot trajectories
    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(nitr):
        ax.plot3D(x_hist_p[i,:, 0], x_hist_p[i,:, 1], x_hist_p[i,:, 2], color="b", marker='o', markersize=3)
        ax.plot3D(x_hist_j[i,:, 0], x_hist_j[i,:, 1], x_hist_j[i,:, 2], color="r", marker='o', markersize=3)
    ax.plot3D(x_hist_p[nitr, :, 0], x_hist_p[nitr, :, 1], x_hist_p[nitr, :, 2], color="b", marker='o', markersize=3)

    ax.scatter3D(x0[0], x0[1], x0[2], color='k')
    ax.scatter3D(xf[0], xf[1], xf[2], color='k')
    if CHandler.obs_data:
        for obs in CHandler.obs_data:
            draw_obs(obs.type, obs.shape, obs.pos, ax)

    plt.legend(["python","julia"])

    fig, axs =  plt.subplots(2,3)
    # plot sigma
    #axs[0, 0].plot(sig_hist_p[-1,:], color="b")
    axs[0, 0].plot(sig_hist_j[-1,:], color="r")
    axs[0, 0].set_title("sigma")

    # plot actions
    axs[0,1].plot(u_hist_p[-1,:], color="b")
    axs[0,1].plot(u_hist_j[-1,:], color = "r")
    axs[0,1].set_title("inputs")

    # plot tilt angle
    last_itr_u_p = u_hist_p[-1,:]
    last_itr_u_j = u_hist_j[-1,:]
    n = np.array([0, 0, 1])
    tilt_angle_p = np.array([np.arccos(n.T @ a * (1 / np.linalg.norm(a, 2))) * (180 / np.pi) for a in last_itr_u_p])
    tilt_angle_j = np.array([np.arccos(n.T @ a * (1 / np.linalg.norm(a, 2))) * (180 / np.pi) for a in last_itr_u_j])
    axs[0,2].plot(tilt_angle_p,color= "b")
    axs[0,2].plot(tilt_angle_j,color= "r")
    axs[0,2].set_title("tilt angle")

    # plot cost history
    # nl cost
    axs[1, 1].plot(nl_cost_hist_p[:], color="b")
    axs[1, 1].plot(nl_cost_hist_j[:,0,0], color="r")
    axs[1, 1].set_title("nonlinear cost")

    print("nl_cost: ", nl_cost_hist_p[:])
    print("l_cost: ", l_cost_hist_p[:])

    # l cost
    axs[1,0].plot(l_cost_hist_p[:], color="b")
    axs[1,0].plot(l_cost_hist_j[:,0,0], color="r")
    axs[1,0].set_title("linear cost")

    # plot convergence
    X_p = np.concatenate((x_hist_p[1:], u_hist_p[1:], np.repeat(p_hist_p[1:,:,np.newaxis],T,axis=1)),2)
    X_opt_p = X_p[-1]
    X_j = np.concatenate((x_hist_j, u_hist_j, np.repeat(p_hist_j,T,axis=1)), 2)
    X_opt_j = X_j[-1]

    dist_to_opt_p = np.array([np.linalg.norm(Xi - X_opt_p, 2) for Xi in X_p[:-1]])
    dist_to_opt_j = np.array([np.linalg.norm(Xi - X_opt_j, 2) for Xi in X_j[:-1]])
    # difference between julia and python solutions
    dist_to_opt_jp = np.array([np.linalg.norm(Xi - X_opt_j, 2) for Xi in X_p])

    axs[1,2].plot(dist_to_opt_p / np.linalg.norm(X_opt_p, 2), marker='o', color='b')
    axs[1,2].plot(dist_to_opt_j / np.linalg.norm(X_opt_j, 2), marker='o', color='r')
    axs[1, 2].plot(dist_to_opt_jp / np.linalg.norm(X_opt_j, 2), marker='o', color='g')
    axs[1,2].set_title("dist to optimal solution")

    plt.legend(["python", "julia"])
    plt.yscale('log')

    fig, axs = plt.subplots(2, 3)

    vd_max_p = np.max(np.max(np.abs(vd_hist_p[1:]), axis=2), axis=1)
    vd_max_j = np.max(np.max(np.abs(vd_hist_j), axis=2), axis=1)

    vs_max_p = np.max(np.max(np.abs(vs_hist_p[1:]), axis=2), axis=1)
    vs_max_j = np.max(np.max(np.abs(vs_hist_j), axis=2), axis=1)

    vic_max_p = np.max(np.abs(vic_hist_p[1:]), axis=1)
    vic_max_j = np.max(np.max(np.abs(vic_hist_j), axis=2), axis=1)

    vtc_max_p = np.max(np.abs(vtc_hist_p[1:]), axis=1)
    vtc_max_j = np.max(np.max(np.abs(vtc_hist_j), axis=2), axis=1)

    # plot dynamic slack
    axs[0,0].plot(vd_max_p[:], color="b")
    axs[0,0].plot(vd_max_j[:], color="r")
    axs[0, 0].set_title("max dyn slack")

    # plot slack on path
    axs[0, 1].plot(vs_max_p[:], color="b")
    axs[0, 1].plot(vs_max_j[:], color="r")
    axs[0, 1].set_title("max obs slack")

    # plot ic slack
    axs[1, 0].plot(vic_max_p[:], color="b")
    axs[1, 0].plot(vic_max_j[:], color="r")
    axs[1, 0].set_title("init cond slack")

    # plot tc slack
    axs[1, 1].plot(vtc_max_p[:], color="b")
    axs[1, 1].plot(vtc_max_j[:], color="r")
    axs[1, 1].set_title("term cond slack")

    # plot roh
    axs[0, 2].plot(roh_hist_p, color="b")
    axs[0, 2].plot(np.squeeze(roh_hist_j), color="r")
    axs[0, 2].set_title("roh (t-region crit)")

    # plot eta
    axs[1, 2].plot(eta_hist_p, color="b")
    axs[1, 2].plot(np.squeeze(eta_hist_j), color="r")
    axs[1, 2].set_title("trust region radius")

    plt.legend(["python", "julia"])

    # plot converged trajectory
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_hist_p[-1, :, 0], x_hist_p[-1, :, 1], x_hist_p[-1, :, 2], color="b", marker='o', markersize=3)
    ax.plot3D(x_hist_j[-1, :, 0], x_hist_j[-1, :, 1], x_hist_j[-1, :, 2], color="r", marker='o', markersize=3)

    ax.scatter3D(x0[0], x0[1], x0[2], color='k')
    ax.scatter3D(xf[0], xf[1], xf[2], color='k')
    if CHandler.obs_data:
        for obs in CHandler.obs_data:
            draw_obs(obs.type, obs.shape, obs.pos, ax)

    plt.legend(["python", "julia"])

    feasibility_p = [vd_max_p[-1] <= 1e-7, vs_max_p[-1] <= 1e-7, vic_max_p[-1] <= 1e-7, vtc_max_p[-1] <= 1e-7]
    feasibility_j = [vd_max_j[-1] <= 1e-7, vs_max_j[-1] <= 1e-7, vic_max_j[-1] <= 1e-7, vtc_max_j[-1] <= 1e-7]

    print("cost of found solution (python): ", nl_cost_hist_p[-1])
    print(vd_max_p[-1])
    print("cost of found solution (julia): ", nl_cost_hist_j[-1])
    print(vs_max_p[-1])
    print("python solution feasible?: ", feasibility_p )
    print("julia solution feasible?: ", feasibility_j )
    print("trajectory time python: ", p_hist_p[-1])
    print("trajectory time julia: ", p_hist_j[-1])

    plt.show()
