from casadi import *
from casadi.tools import *
import subprocess

x = SX.sym('x',2)
u = SX.sym('u',1)

def func(x, u):
    return x

F = Function('f',[x,u],[func(x,u)])

F.generate("gen.c")

cmd = "gcc -fPIC -shared gen.c -o gen.so"
print("compiling c code")
subprocess.run(cmd.split())
print("compiling done")

f = external('f',"gen.so")

opt = casadi.Opti()
X = opt.variable(2,1)
U = opt.variable(1,10)

print(f(X[:,0],U[:,0]))
print(func(X[:,0],U[:,0]))


