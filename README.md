# deepxde_1d_heat_main

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


Using the DeepXDE framework, the physical residuals are optimized:

PDE Loss: Evaluated at 2,000 domain points to enforce fluid conservation laws.

BC Loss: Evaluated at 200 boundary points to satisfy Dirichlet conditions.

Automatic Differentiation: Employs Jacobians & Hessian to calculate exact spatial derivatives without the truncation errors of traditional mesh-based solvers.

---

3. Two-stage Optimization Strategy


A hybrid approach was implemented to achieve both robust exploration and high-precision convergence:

* Adam Optimizer:

Max Iterations: 3,000

A second-order optimizer that utilizes Hessian approximations to achieve high-precision refinement in the final training stage.

---

4. Result & Visualization


The trained PINN(Physically Informed Neural Network) acts as a continuous surrogate model, allowing for inference at any point within the domain.

* Flow Field Prediction:

Predications for velocity components ($u,v$) and pressure ($p$) were generated across 500,000 sample points.

* Visualization:

High-fidelity contour plots using the ($jet$) colormap visualize the velocity gradients near the walls and the pressure drop across the channel.

---

5. Implementation DetailsFramework:

* DeepXDE with a TensorFlow backend.

* Key Libraries: NumPy for data processing, Matplotlib for scientific visualization.

* Hardware: Optimized for CUDA-enabled GPU acceleration.

* Progress Tracking: TQDM integrated for real-time training monitoring.

