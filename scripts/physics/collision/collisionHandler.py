import fcl
import numpy as np
import sys

class collisionHandler():

    def __init__(self,robot):
        self.obs = []
        self.obs_data = []

        self.robot_type = robot.type
        self.robot_skel = fcl.Sphere(robot.arm_length)

        self.obs_manager = fcl.DynamicAABBTreeCollisionManager()

        self.dRequest = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)

    def redim_robot(self,arm_length):
        self.robot_skel = fcl.Sphere(arm_length)

    def addObject(self,type,Shape,pos,quat):
        if type == 'box':
            skel = fcl.Box(Shape[0], Shape[1], Shape[2])
        if type == 'sphere':
            skel = fcl.Sphere(Shape[0])
        if type == 'cylinder':
            skel = fcl.Cylinder(Shape[0], Shape[1])

        obj = fcl.CollisionObject(skel, fcl.Transform(quat, pos))

        self.obs.append(obj)

        self.obs_manager.registerObjects(self.obs)
        self.obs_manager.setup()

    def calcDistance(self, xprev):

        points_robot = []
        points_obs = []
        distances = []

        for obs in self.obs:
            res = fcl.DistanceResult()

            pos = xprev[0:3]
            if self.robot_type == "fM":
                quat = xprev[6:10]
            else:
                quat = [1,0,0,0]
            robot_obj = fcl.CollisionObject(self.robot_skel, fcl.Transform(quat,pos))
            dist = fcl.distance(robot_obj, obs, self.dRequest, res)

            distances.append(res.min_distance)
            points_robot.append(res.nearest_points[0])
            points_obs.append(res.nearest_points[1])

        return np.array(distances), np.array(points_robot), np.array(points_obs)

    def calcDistance_broadphase(self, xprev):
        # =====================================================================
        # Managed one to many collision checking
        # =====================================================================
        ddata = fcl.DistanceData(request=self.dRequest)

        pos = xprev[0:3]
        robot_obj = fcl.CollisionObject(self.robot_skel, fcl.Transform(pos))

        self.obs_manager.distance(robot_obj, ddata, fcl.defaultDistanceCallback)

        return ddata.result.min_distance, ddata.result.nearest_points[1], ddata.result.nearest_points[0]
