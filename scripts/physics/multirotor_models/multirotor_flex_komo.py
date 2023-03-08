import sys
sys.path += ['../../Master_Thesis_Welf/build']
import libry as ry
import numpy as np
import tempfile

def build_copter(nrMotors,r):

    C = ry.Config()
    tmp = tempfile.NamedTemporaryFile(mode = 'w+t', prefix = "flex_quad", suffix= "file.g")
    tmp.writelines("world {}\n")
    tmp.writelines("drone (world) { joint:free X:<t(0 0 1) d(1 0 0 0)> }\n")

    if nrMotors == 2:

        quat_arm = [np.cos(-.5 * np.pi / 4), 0, 0, np.sin(-.5 * np.pi / 4)]

        # position of the motors
        pos_arr_x = [np.cos(np.pi / 4) * r, -np.cos(np.pi / 4) * r]
        pos_arr_y = [-np.sin(np.pi / 4) * r, np.sin(np.pi / 4) * r]

        tmp.writelines("arm" + str(1) + "(drone){Q:<q(" + str(quat_arm[0]) + " " + str(quat_arm[1]) + " " + str(quat_arm[2]) + " " + str(quat_arm[3])
                       + ")> shape:ssBox size:[" + str(2*r) + " 0.005 0.005 0.001] color:[.9 .9 .9] mass:0.015}\n")

        for m in range(nrMotors):
            tmp.writelines("m" + str(m+1) + "(drone){ Q:[ " + str(pos_arr_x[m]) + " " + str(pos_arr_y[m])
                           + " 0] shape:cylinder size:[.02 .005] color:[.9 .9 .9]}\n" )


    elif nrMotors == 3:
        # quaternions of the arms
        quat_arm1 = [np.cos(.5 * np.pi / 2), 0, 0, np.sin(.5 * np.pi / 2)]
        quat_arm2 = [np.cos(.5 * np.pi / 6), 0, 0, np.sin(-.5 * np.pi / 6)]
        quat_arm3 = [np.cos(.5 * np.pi / 6), 0, 0, np.sin(.5 * np.pi / 6)]
        quat_arm = np.vstack((quat_arm1, quat_arm2, quat_arm3))

        pos_arm_x = [0, -np.cos(np.pi/6)*r/2, np.cos(np.pi/6)*r/2]
        pos_arm_y = [-r/2, np.sin(np.pi/6)*r/2, np.sin(np.pi/6)*r/2]

        # position of the motors
        pos_arr_x = [0, -np.cos(np.pi/6)*r, np.cos(np.pi/6)*r]
        pos_arr_y = [-r, np.sin(np.pi/6)*r, np.sin(np.pi/6)*r]

        for a in range(nrMotors):
            tmp.writelines("arm" + str(a+1) + "(drone){Q:< t(" + str(pos_arm_x[a]) + " " + str(pos_arm_y[a]) + " 0) q(" + str(quat_arm[a,0]) + " " + str(quat_arm[a,1]) + " " + str(quat_arm[a,2]) + " " + str(quat_arm[a,3])
                           + ")> shape:ssBox size:[" + str(r) + " 0.005 0.005 0.001] color:[.9 .9 .9] mass:0.015}\n")

        for m in range(nrMotors):
            tmp.writelines("m" + str(m+1) + "(drone){ Q:[ " + str(pos_arr_x[m]) + " " + str(pos_arr_y[m])
                           + " 0] shape:cylinder size:[.02 .005] color:[.9 .9 .9]}\n" )


    elif nrMotors == 4:
        # quaternions of the arms
        quat_arm1 = [np.cos(.5*np.pi/4), 0, 0, np.sin(.5*np.pi/4)]
        quat_arm2 = [np.cos(-.5 * np.pi / 4), 0, 0, np.sin(-.5 * np.pi / 4)]
        quat_arm = np.vstack((quat_arm1, quat_arm2))

        # position of the motors
        pos_arr_x = [np.cos(np.pi / 4) * r, -np.cos(np.pi / 4) * r, -np.cos(np.pi / 4) * r, np.cos(np.pi / 4) * r]
        pos_arr_y = [-np.sin(np.pi / 4) * r, -np.sin(np.pi / 4) * r, np.sin(np.pi / 4) * r, np.sin(np.pi / 4) * r]

        for a in range(int(nrMotors/2)):
            tmp.writelines("arm" + str(a+1) + "(drone){Q:<q(" + str(quat_arm[a,0]) + " " + str(quat_arm[a,1]) + " " + str(quat_arm[a,2]) + " " + str(quat_arm[a,3])
                           + ")> shape:ssBox size:[" + str(2*r) + " 0.005 0.005 0.001] color:[.9 .9 .9] mass:0.015}\n")

        for m in range(nrMotors):
            tmp.writelines("m" + str(m+1) + "(drone){ Q:[ " + str(pos_arr_x[m]) + " " + str(pos_arr_y[m])
                           + " 0] shape:cylinder size:[.02 .005] color:[.9 .9 .9]}\n" )

    elif nrMotors == 6:
        # quaternions of the arms
        quat_arm1 = [np.cos(.5*np.pi/3), 0, 0, np.sin(.5*np.pi/3)]
        quat_arm2 = [np.cos(-.5 * np.pi / 3), 0, 0, np.sin(-.5 * np.pi / 3)]
        quat_arm3 = [0, 0, 0, 1]
        quat_arm = np.vstack((quat_arm1,quat_arm2,quat_arm3))

        # position of the motors
        pos_arr_x = [np.cos(np.pi / 3)*r, -np.cos(np.pi / 3)*r,
                     -r, -np.cos(np.pi / 3)*r, np.cos(np.pi / 3)*r, r]
        pos_arr_y = [-np.sin(np.pi / 3) * r, -np.sin(np.pi / 3) * r,
                     0, np.sin(np.pi / 3) * r, np.sin(np.pi / 3) * r, 0]

        for a in range(int(nrMotors/2)):
            tmp.writelines("arm" + str(a+1) + "(drone){Q:<q(" + str(quat_arm[a,0]) + " " + str(quat_arm[a,1]) + " " + str(quat_arm[a,2]) + " " + str(quat_arm[a,3])
                           + ")> shape:ssBox size:[" + str(2*r) + " 0.005 0.005 0.001] color:[.9 .9 .9] mass:0.015 contact}\n")

        for m in range(nrMotors):
            tmp.writelines("m" + str(m + 1) + "(drone){ Q:[ " + str(pos_arr_x[m]) + " " + str(pos_arr_y[m])
                           + " 0] shape:cylinder size:[.02 .005] color:[.9 .9 .9] contact}\n")


    elif nrMotors == 8:
        # quaternions of the arms
        quat_arm1 = [np.cos(.5*np.pi/4), 0, 0, np.sin(.5*np.pi/4)]
        quat_arm2 = [np.cos(-.5*np.pi/4), 0, 0, np.sin(-.5*np.pi/4)]
        quat_arm3 = [1, 0, 0, 0]
        quat_arm4 = [np.cos(.5*np.pi/2), 0, 0, np.sin(.5*np.pi/2)]
        quat_arm = np.vstack((quat_arm1, quat_arm2, quat_arm3, quat_arm4))

        # position of the motors
        pos_arr_x = [np.cos(np.pi / 4) * r, 0, -np.cos(np.pi / 4) * r, -r,
                     -np.cos(np.pi / 4) * r, 0, np.cos(np.pi / 4) * r, r]
        pos_arr_y = [-np.sin(np.pi / 4) * r, -r, -np.sin(np.pi / 4) * r, 0,
                     np.sin(np.pi / 4) * r, r, np.sin(np.pi / 4) * r, 0]

        for a in range(int(nrMotors / 2)):
            tmp.writelines("arm" + str(a + 1) + "(drone){Q:<q(" + str(quat_arm[a, 0]) + " " + str(quat_arm[a, 1]) + " " + str(quat_arm[a, 2]) + " " + str(quat_arm[a, 3])
                + ")> shape:ssBox size:[" + str(2 * r) + " 0.005 0.005 0.001] color:[.9 .9 .9] mass:0.015}\n")

        for m in range(nrMotors):
            tmp.writelines("m" + str(m + 1) + "(drone){ Q:[ " + str(pos_arr_x[m]) + " " + str(pos_arr_y[m])
                           + " 0] shape:cylinder size:[.02 .005] color:[.9 .9 .9]}\n")
    else:
        print("wrong number of motors")

    tmp.writelines(" collisionSphere (drone){ shape:sphere size:[" + str(r) + "] color:[.9 .9 .9 0.] contact}\n")
    tmp.writelines(" helperFrame (drone){ shape:marker size:[.05] color:[.9 .9 .9 ] }\n")


    tmp.seek(0)

    C.addFile(tmp.name)
    tmp.close()
    return C
