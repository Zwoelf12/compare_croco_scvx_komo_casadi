# define world frame
world {}
coordSys (world) {type = marker, size = [0.1]}

# define quadcopter
drone (world) {joint:free X=<t(0. 1. .5)> shape:sphere size:[1e-4] color:[1 .0 .0] contact}#, mass=0.03}
coordSys (drone) {type = marker, size = [0.1]}

