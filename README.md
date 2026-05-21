# WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains

Welcome to the official repository for **WINO** (Weak-form Physics-Informed Neural Operator). This repository provides a data-free framework that synergizes the fast inference capabilities of neural operators with the geometric flexibility of the finite element method ($\varphi$-FEM).

---

## 📌 Overview

* **Unfitted Geometric Flexibility:** WINO utilizes $\varphi$-FEM, an unfitted method that seamlessly accommodates domain geometry variations without requiring body-fitted meshes. The physical domain is instead implicitly represented using a level-set function $\varphi$.
* **Data-Free Training:** Model parameters are optimized by minimizing squared weak-form residuals aligned with the $\varphi$-FEM formulation, along with squared penalties on the cut-cell auxiliary equations. This eliminates the computational burden of generating large, paired datasets of converged reference solutions.
* **Iterative Solver Acceleration:** WINO's outputs can serve as Neural Operator Warm Starts (NOWS) to seed classical nonlinear $\varphi$-FEM solvers. This hybrid approach significantly reduces iteration counts compared to standard cold-started solvers.
* **High Performance:** Numerical benchmarks demonstrate that WINO maintains a high degree of accuracy (relative error below 0.04) while reducing total computational time by 50-80% compared to purely data-driven methods.

---

## 🧠 Methodology & Architecture

The core of WINO is built upon a Fourier Neural Operator (FNO) architecture. It maps inputs to discretized approximations of the solution across a fixed background Cartesian grid.

### Inputs and Outputs

* **Dirichlet Problems:** For pure Dirichlet conditions, WINO adopts a lifting strategy where the network predicts a homogeneous displacement component $w_h$ from the level-set field $\varphi_h$, boundary data $g_h$, and body force $f_h$. The discrete displacement is recovered via $u_h=\varphi_hw_h+g_h$.
* **Neumann Problems:** For traction-driven Neumann settings, the identical operator backbone maps the level-set description $\varphi_h$, nominal loads $t_h$, and body forces $f_h$ to the displacement $u_h$. Additionally, it predicts the auxiliary variables ($y_h, p_h$) required to close the weak formulation on cut cells.

### Loss Function Formulation

WINO avoids the instability of high-order strong-form derivatives by employing an integration-based approach:

* **Momentum Residual:** Evaluated using weak-form integrals over the unchanged rectangular mesh via Gaussian quadrature.
* **Auxiliary Constraints:** Imposed via squared strong-form residuals on intersected elements.
* **Total Loss:** Training is governed by minimizing the sum of these components: $\mathcal{L}_{total}=\mathcal{L}_{weak}+\mathcal{L}_{sq}$.

---

## 🚀 Neural Operator Warm Starts (NOWS)

Solving hyperelastic PDEs often demands computationally expensive Newton-Krylov iterations. WINO alleviates this by providing a highly accurate starting guess in a matter of milliseconds.

By injecting WINO's discrete output fields as the initial Newton iterate $U^{(0)}$, the NOWS framework dramatically decreases the required outer Newton iterations and inner Krylov (e.g., GMRES or CG) steps. Crucially, this requires zero modifications to the underlying finite element discretization or the iterative algorithms themselves.

---

## 📊 Benchmarks

The WINO framework was rigorously validated on a variety of hyperelasticity configurations using the compressible Neo-Hookean constitutive model.

* **Pure Dirichlet:**
* Elliptical shape domains.
* Random shape domains (generated via sums of Gaussian functions).


* **Mixed Dirichlet & Neumann:**
* Square plate with an embedded elliptic hole.
* Shear-dominated Cook's membrane.
* Quarter pressure vessel under internal follower forces.



---

## 💻 Code Availability

The datasets and the codebase to reproduce the numerical experiments are freely available and can be found on GitHub:

**Repository Link:** [https://github.com/bokai-zhu/WINO](https://github.com/bokai-zhu/WINO)

---

## 📖 Citation

If you find this framework useful for your academic research, please consider citing the original paper:

```bibtex
@article{zhu2026wino,
  title={WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains},
  author={Zhu, Bokai and Zhang, Qinghui and Rabczuk, Timon},
  year={2026}
}

```
