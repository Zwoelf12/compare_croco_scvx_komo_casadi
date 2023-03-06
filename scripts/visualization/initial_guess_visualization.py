import matplotlib.pyplot as plt
from visualization.geom_visualization import draw_obs
import numpy as np

def visualize_initial_guess(traj,x0,xf,xm,obs):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection = "3d")

    if xm is not None:
        for x in xm:
            if "pos" in x.type:
                ax.scatter3D(x.value[0], x.value[1], x.value[2], marker="o", facecolor="r")

    ax.scatter3D(x0[0], x0[1],x0[2], marker = "o", facecolor = "r")
    ax.scatter3D(xf[0], xf[1],xf[2], marker = "o", facecolor = "r")
    ax.plot(traj[:,0], traj[:,1], traj[:,2])

    if obs is not None:
        for o in obs:
            draw_obs(o.type, o.shape, o.pos, ax)

    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")
    ax.set_zlabel("z [m]")

    set_axes_equal(ax)

    fig, axs = plt.subplots(2, 2)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # plot positions
    ax1.plot(traj[:, :3])
    legend_ax1 = ["p_x", "p_y", "p_z"]

    # plot velocities
    ax2.plot(traj[:, 3:6])
    legend_ax2 = ["v_x", "v_y", "v_z"]

    # plot quaternions
    ax3.plot(traj[:, 6:10])
    legend_ax3 = ["q_1", "q_2", "q_3", "q_4"]

    print(np.linalg.norm(traj[:, 6:10], axis=1))

    # plot rotational velocities
    ax4.plot(traj[:, 10:13])
    legend_ax4 = ["w_x", "w_y", "w_z"]

    # make legend
    ax1.legend(legend_ax1)
    ax2.legend(legend_ax2)
    ax3.legend(legend_ax3)
    ax4.legend(legend_ax4)

    ax1.set_xlabel("step %#")
    ax1.set_ylabel("positions [m]")
    ax1.set_title("position comparision")

    ax2.set_xlabel("step %#")
    ax2.set_ylabel("velocities [m/s]")
    ax2.set_title("velocity comparision")

    ax3.set_xlabel("step %#")
    ax3.set_ylabel("quaternion [1]")
    ax3.set_title("quaternion comparision")

    ax4.set_xlabel("step %#")
    ax4.set_ylabel("rotational velocities [rad/s]")
    ax4.set_title("rotational velocities comparision")

    plt.show()

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
