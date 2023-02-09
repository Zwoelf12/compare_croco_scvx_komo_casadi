import numpy as np
import time

import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d import Axes3D

# visualization related
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import matplotlib.animation as animation
from visualization.geom_visualization import draw_obs

def animate_fM(data,objs):
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0, 0, 0]).dot(
		tf.euler_matrix(0, np.radians(-30), -np.pi/2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1, 0, 0]))

	vis["Quadrotor"].set_object(
		g.StlMeshGeometry.from_file('./physics/multirotor_models/crazyflie2.stl'))

	if objs:
		i = 1
		for obj in objs:
			if obj.type == "box":
				vis["obs" + str(i)].set_object(g.Box(obj.shape))
			elif obj.type == "sphere":
				vis["obs" + str(i)].set_object(g.Sphere(obj.shape))

			vis["obs" + str(i)].set_transform(
				tf.translation_matrix(obj.pos).dot(
					tf.quaternion_matrix(obj.quat))
			)

			i += 1

	while True:
		for row in data:
			vis["Quadrotor"].set_transform(
				tf.translation_matrix([row[0], row[1], row[2]]).dot(
					tf.quaternion_matrix(row[6:10])))
			time.sleep(0.1)
		time.sleep(1)

def animate_dI(robot,data,objs,x0,xf):

	def update_ani(t):
		copter._offsets3d = ([data[t, 0]], [data[t, 1]], [data[t, 2]])

	num_steps = len(data[:, 0])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# draw initial and final configuration
	ax.scatter([x0[0], xf[0]], [x0[1], xf[1]], [x0[2], xf[2]], 'ro')

	if objs:
		for obs in objs:
			draw_obs(obs.type, obs.shape, obs.pos, ax)

	copter = ax.scatter([], [], [], 'k*')
	ax.set_xlim(robot.min_x[0], robot.max_x[0])
	ax.set_ylim(robot.min_x[1], robot.max_x[1])
	ax.set_zlim(robot.min_x[2], robot.max_x[2])
	ax.grid()
	ani = animation.FuncAnimation(fig, update_ani, num_steps, interval=40, blit=False)
	plt.show()

def animate_dI_V2(data,objs):
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0, 0, 0]).dot(
			tf.euler_matrix(0, np.radians(-30), -np.pi / 2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1, 0, 0]))

	vis["Quadrotor"].set_object(
		g.StlMeshGeometry.from_file('../../multirotor_models/crazyflie2.stl'))

	if objs:
		i = 1
		for obj in objs:
			if obj.type == "box":
				vis["obs" + str(i)].set_object(g.Box(obj.shape))
			elif obj.type == "sphere":
				vis["obs" + str(i)].set_object(g.Sphere(obj.shape))

			vis["obs" + str(i)].set_transform(
				tf.translation_matrix(obj.pos).dot(
					tf.quaternion_matrix(obj.quat))
			)

			i += 1

	while True:
		for row in data:
			vis["Quadrotor"].set_transform(
				tf.translation_matrix([row[0], row[1], row[2]]).dot(
					tf.quaternion_matrix([1,0,0,0])))
			time.sleep(0.1)
		time.sleep(1)

