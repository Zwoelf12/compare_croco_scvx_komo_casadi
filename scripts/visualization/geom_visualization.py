import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import math

# function from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def draw_obs(type, Shape, pos, ax, quat=None):
    if type == "sphere":
        (n, m) = (15, 15)
        r = Shape[0]
        # Meshing a unit sphere according to n, m
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        phi = np.linspace(np.pi * (-0.5 + 1. / (m + 1)), np.pi * 0.5, num=m, endpoint=False)
        theta, phi = np.meshgrid(theta, phi)
        theta, phi = theta.ravel(), phi.ravel()
        theta = np.append(theta, [0.])  # Adding the north pole...
        phi = np.append(phi, [np.pi * 0.5])
        mesh_x, mesh_y = ((np.pi * 0.5 - phi) * np.cos(theta), (np.pi * 0.5 - phi) * np.sin(theta))
        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
        x, y, z = pos[0] + r * np.cos(phi) * np.cos(theta), pos[1] + r * np.cos(phi) * np.sin(theta), pos[2] + r * np.sin(phi)

        # Defining a custom color scalar field
        vals = np.sin(6 * phi) * np.sin(3 * theta)
        colors = np.mean(vals[triangles], axis=1)

        # Plotting
        cmap = plt.get_cmap('Reds')
        triang = mtri.Triangulation(x, y, triangles)
        collec = ax.plot_trisurf(triang, z, color=(0.8, 0.8, 0.8, 0.4), shade=True, linewidth=0.00, edgecolor='Gray')

    elif type == "box":
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi)

        x = np.cos(Phi)*np.sin(Theta)
        y = np.sin(Phi)*np.sin(Theta)
        z = np.cos(Theta)/np.sqrt(2)

        box = ax.plot_surface(pos[0] + x*(Shape[0]), pos[1] + y*(Shape[1]), pos[2] + z*(Shape[2]), color=(0.8, 0.8, 0.8, 0.3), shade=True, linewidth=0.1, edgecolor='Gray')

    elif type == "cylinder":
        def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
            z = np.linspace(0, height_z, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + center_x
            y_grid = radius * np.sin(theta_grid) + center_y
            return x_grid, y_grid, z_grid

        Xc,Yc,Zc = data_for_cylinder_along_z(pos[0], pos[1], Shape[0], Shape[1])

        ax.plot_surface(Xc, Yc, Zc + pos[2] - Shape[1]/2, color=(0.8, 0.8, 0.8, 0.3), shade=True, linewidth=0.1, edgecolor='Gray')
