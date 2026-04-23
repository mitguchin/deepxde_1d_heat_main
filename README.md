# deepxde_1d_heat_main

🚀 Physics-Informed Machine Learning (PIML) PortfolioThis portfolio showcases the implementation of Physics-Informed Neural Networks (PINNs) to solve complex partial differential equations (PDEs). By embedding physical laws into the loss function, these models achieve high-fidelity predictions in spatio-temporal domains even with limited labeled data.

1. Burgers' Equation Solver (1D & 2D)A fundamental project focusing on the non-linear viscous Burgers' equation, which models advection and diffusion processes.
🧠 Model Architecture & Methodology
Framework: Implemented using PyTorch to leverage its automatic differentiation (autograd) engine.
Architecture: A Fully Connected Neural Network (FNN) featuring 5 hidden layers with 20–30 neurons each.
Activation Function: Tanh was utilized for smooth higher-order derivatives, ensuring that curvature information remains intact during Laplacian calculations.
Two-Stage Optimization:

Adam: Used initially for robust global convergence and stable early training.
L-BFGS: Employed in the final stage to utilize second-order curvature information for precise local convergence.

📊 Results

Shock Wave Capture: Successfully predicted shock formation where the wave profile steepens over time.
3D Data Storage: Implemented 3D arrays to store temporal evolution data across the spatial grid.

2. 1D Heat Equation (Thermal Conduction)
Developed using the DeepXDE library to predict thermal distribution in a space-time domain.

🛠️ Technical Implementation
Unified Spatio-Temporal Domain: Utilized GeometryXTime to merge spatial intervals and time domains into a single computational grid.
Slicing Strategy: Implemented advanced NumPy slicing ([:, 0:1]) to maintain 2D matrix shapes, preventing dimensionality errors common in standard indexing.
Differential Operators: Defined the PDE through Jacobian (1st-order) and Hessian (2nd-order) operators to satisfy the heat conduction law $u_t = k u_{xx}$.

3. 2D Navier-Stokes Equation (Fluid Dynamics)
An advanced simulation of fluid flow within a rectangular domain, predicting velocity fields ($u, v$) and pressure ($p$).
⚖️ Physical Constraints
Continuity Equation: Enforced mass conservation ($\nabla \cdot \mathbf{u} = 0$) to ensure the fluid is incompressible.
No-Slip Boundary Condition: Fixed velocities at the walls to zero to simulate realistic viscous fluid behavior.
Collocation Sampling: Strategically placed points within the domain and along boundaries to monitor PDE compliance throughout the training.

🔬 Scientific VisualizationHeatmaps: 
Generated using Seaborn and the jet colormap to visualize physical quantity distributions across space and time.
Surface Plots: Used 3D visualizations to verify the transition of initial sinusoidal states toward steady-state linear profiles.
Loss History: Documented the convergence patterns of Adam and L-BFGS phases to demonstrate model reliability.

🎓 Academic Reflection
This portfolio demonstrates the transition of deep learning from simple pattern recognition to a robust tool for Numerical Analysis. 
By integrating PDE residuals directly into the neural network's learning process, I have developed a strong foundation in Physics-Informed Machine Learning, a critical skill for high-level research in AI-driven science and engineering.
