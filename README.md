# deepxde_2d_heat_main

---

1. Problem Statement: Navier Stokes Equation

: The goal is to simulate fluid flow within a rectangular domain by minimizing the residuals of the governing physical laws directly within the neural network's loss function.

* Governing Equations
The steady-state incompressible Navier-Stokes Equations and the Continuity Equation are defined as:

Momentum ($x$-direction): 

$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial x} - \frac{\mu}{\rho} \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$


Momentum ($y$-direction):

$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial y} - \frac{\mu}{\rho} \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$


Continuity:

$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} =0$

* Simulation Setup

Domain: Rectangle ($L=2,D=1$) centered at the origin.

Physical Parameters: Density ($\rho$) = 1, Dynamic Viscosity ($\mu$) = 1

* Boundary Conditions (BC):

Inlet($x = -L/2$): Constant velocity $u = 1, v = 0$

Oulet($x = L/2$): Reference pressure $p = 0$ and $v = 0$

Walls($y = \pm D/2$): No-slip condition $u = 0, v = 0$

---

2. Technical Highlights


* Neural Network Architecture


Structure: Fully Connected Neural Network(FNN).

I/O: Inputs($x,y$) $\rightarrow$ 3 Outputs ($u,v,p$)

Depth: 5 hidden layers with 64 neurons each.

Activation: Tanh(Hyperbolic Tangent) is utilized to ensure the existence of smooth higher-order derivatives required for the Laplacian
($\nabla^2$) terms in the PDE loss.


* Physics-Informed Training(Autograd)


Using the Deep

