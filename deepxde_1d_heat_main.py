import deepxde as dde
from deepxde.backend import tf
import numpy as np
import matplotlib.pyplot as plt

k = 0.4
L = 1
n = 1

geom = dde.geometry.Interval(0,L)
timedomain = dde.geometry.TimeDomain(0,n)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ic = dde.icbc.IC(geomtime, lambda x: np.sin(n* np.pi * x[:,0:1]/L), lambda _, on_initial: on_initial)

input_array = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])

lambda_function = lambda x : 2*x[:,0:1]

result_array = lambda_function(input_array)

print(result_array)

cond_lambda = lambda x : "Even" if x%2 == 0 else "Odd"
print(cond_lambda(4))
print(cond_lambda(3))

def double_first_column(input_array):
    print("input_array", input_array)
    print("input_array.shape", input_array.shape)
    print("input_array[:, 0:1]", input_array[:, 0:1])
    print("input_array[:, 0:1].shape", input_array[:, 0:1].shape)
    print("input_array[:, 0]", input_array[:, 0])
    print("input_array[:, 0].shape", input_array[:, 0].shape)
    return 2 * input_array[:, 0:1]

bc = dde.icbc.DirichletBC(
    geomtime, 
    lambda input_array: double_first_column(input_array),
    lambda _,
    on_boundary: on_boundary)
	
def pde(comp,u):
		du_t = dde.grad.jacobian(u,comp, i=0,j=1)
		du_xx = dde.grad.hessian(u,comp, i=0,j=0)
		return du_t - k * du_xx
	
data = dde.data.TimePDE(geomtime,
                       pde,
                       [bc,ic],
                       num_domain = 2540,
                       num_boundary = 80,
                       num_initial = 160,
                       num_test = 2540,
                       )
net = dde.nn.FNN([2] + [20]*3+ [1], "tanh", "Glorot normal")
	
plt.figure(figsize = (10,8))
plt.scatter(data.train_x_all[:,0],data.train_x_all[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations = 15000)

model.compile("L-BFGS-B")
losshistory, train_state = model.train()

dde.saveplot(losshistory,train_state, issave= True, isplot=True)

model.compile("adam", lr = 1e-3)
model.compile("L-BFGS-B")
dde.saveplot(losshistory,train_state, issave= True, isplot=True)
	