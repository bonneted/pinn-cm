# pinn-cm
Implementation of the PINN (Physic-Informed Neural Network) framework for continuum mechanics problems.
This repository contains the code used to generate the results of the conference paper "Quantifying uncertainty of Physics-Informed Neural Networks for continuum mechanics applications".
It relies on the DeepXDE library (github.com/lululxvi/deepxde).

We address the boundary value problem introduced in [1] that has an analytical solution (considering linear elasticity).
We consider the unit square domain : $\Omega = [0,1]^2$, on which the continuum mechanics equations applies :

$\epsilon_{ij} = \frac{1}{2}(u_{i,j}+u_{j,i}) $\
$\sigma_{ij} =  \lambda \epsilon_{ij} \delta{ij} + 2 \mu \epsilon_{ij}$\
$\sigma_{ij,j} + f_{i} = 0 $

The boundary conditions are summarized in the image below, and the body forces are chosen on purpose so that there can be an analytical solution :

$fx = \lambda \left(- \pi Q y^{3} \cos{\left(\pi x \right)} + 4 \pi^{2} \sin{\left(2 \pi y \right)} \cos{\left(2 \pi x \right)}\right) + \mu \left(- \pi Q y^{3} \cos{\left(\pi x \right)} + 9 \pi^{2} \sin{\left(\pi y \right)}\right)$\
$fy = \lambda \left(- 3 Q y^{2} \sin{\left(\pi x \right)} + 2 \pi^{2} \sin{\left(2 \pi x \right)} \cos{\left(\pi y \right)}\right) + \mu \left(\frac{\pi^{2} Q y^{4} \sin{\left(\pi x \right)}}{16} - 6 Q y^{2} \sin{\left(\pi x \right)} + 2 \pi^{2} \sin{\left(2 \pi x \right)} \cos{\left(\pi y \right)}\right)$

![](figures/BVP_problem.png "Boundary conditions of the problem")

A simplified version of the problem is first considered, with only Dirichlet boundary conditions. 
The PINN converges to the analytical solution, with a final error of about 1e-5 (see results/simplified_BVP/).
This proves that the neural network has enough capacity to learn the solution.

The mixed boundary value problem is then considered, with both Dirichlet and Neumann boundary conditions.
Two implementations are considered (as illustrated in the image below): 
- direct implementation (left), where displacement is predicted by the PINN and stress is then computed using the constitutive law
- parallel implementation (right), where the PINN predicts both displacement and stress, the constitutive law is enforced "softly" using an extra loss term

![](figures/PINN_implementation.png "Two possible implementations of PINNs for continuum mechanics: direct (left) and parallel (right)")

The convergence is not as good as in the simplified problem, with a final error of about 5e-3 (see results/mixed_BVP/).

To improve the accuracy, we perform hyperparameter optimization using Weights & Biases (wandb.com).
The minimum error reached is about 3e-3, which is a small improvement, but still far from the accuracy of the simplified problem.
The model seems stuck in a local minimum due to the mixed boundary conditions (the model is capable enough to learn the solution, as shown in the simplified problem).
This encourages further investigation of better ways to implement the mixed boundary conditions in the PINN framework.

[1] Haghighat, Ehsan, Maziar Raissi, Adrian Moure, Hector Gomez, and Ruben Juanes. “A Deep Learning Framework for Solution and Discovery in Solid Mechanics.” 

## Structure
    pinn-cm/
    ├── figures/ # figures used in the README and the notebooks
    ├── HPO/ # hyperparameter optimization
    │   ├── HPO.ipynb # hyperparameter optimization for the mixed BVP
    │   ├── mixed_BVP.py # mixed BVP functions (identical to the ones in the notebook)
    │   └── wandb/ # runs stored using Weights & Biases
    ├── inverse identification/ # inverse identification of the material parameters
    │   └── linear_elasticity.ipynb
    ├── results/ # results of the notebooks
    │   ├── mixed_BVP 
    │   └── simplified_BVP 
    ├── analytical_solution.ipynb # analytical solution of the problem using Sympy (symbolic math)
    ├── mixed_BVP.ipynb # mixed boundary value problem
    └── simplified_BVP.ipynb # simplified boundary value problem (only Dirichlet BCs)

## Author
Damien Bonnet-Eymard
