# deepxde_2d_heat_main

---

1. Problem Statement: Navier Stokes Equation
: The goal is to simulate fluid flow within a rectangular domain by minimizing the residuals of the governing physical laws directly within the neural network's loss function.

* Governing Equations
The steady-state incompressible Navier-Stokes Equations and the Continuity Equation are defined as:

Momentum ($x$-direction): 
$u \frac(\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho}\left(\frac{\partial^2 u}{\partial x^2} + \frac{\parital^2 u}{\partial y^2}\right) = 0$
