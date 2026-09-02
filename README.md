# WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains

This repository accompanies the paper

**WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains**

---

## 📌 Overview

* **Unfitted Geometric Flexibility:** WINO utilizes $\varphi$-FEM, an unfitted method that seamlessly accommodates domain geometry variations without requiring body-fitted meshes. The physical domain is instead implicitly represented using a level-set function $\varphi$.
* **Data-Free Training:** Model parameters are optimized by minimizing squared weak-form residuals aligned with the $\varphi$-FEM formulation, along with squared penalties on the cut-cell auxiliary equations. This eliminates the computational burden of generating large, paired datasets of converged reference solutions.
* **Iterative Solver Acceleration:** WINO's outputs can serve as Neural Operator Warm Starts (NOWS) to seed classical nonlinear $\varphi$-FEM solvers. This hybrid approach significantly reduces iteration counts compared to standard cold-started solvers.
* **High Efficiency:** Numerical benchmarks demonstrate that WINO maintains a high degree of accuracy (relative error below 0.04) while reducing total computational time by 50-80% compared to purely data-driven methods.

---

## 🧠 Methodology & Architecture

The core of WINO is built upon a Fourier Neural Operator (FNO) architecture. It maps inputs to discretized approximations of the solution across a fixed background Cartesian grid.

### Inputs and Outputs
<img width="1441" height="873" alt="WINO_Flowchart" src="https://github.com/user-attachments/assets/efa6d934-da84-4369-8fca-3d56b6717239" />

* **Dirichlet Problems:** For pure Dirichlet conditions, WINO adopts a lifting strategy where the network predicts a homogeneous displacement component $w_h$ from the level-set field $\varphi_h$, boundary data $g_h$, and body force $f_h$. The discrete displacement is recovered via $u_h=\varphi_hw_h+g_h$.
* **Neumann Problems:** For traction-driven Neumann settings, the identical operator backbone maps the level-set description $\varphi_h$, nominal loads $t_h$, and body forces $f_h$ to the displacement $u_h$. Additionally, it predicts the auxiliary variables ($y_h, p_h$) required to close the weak formulation on cut cells.

## 📊 Benchmarks

The WINO framework was rigorously validated on a variety of hyperelasticity configurations using the compressible Neo-Hookean constitutive model.

* Elliptical shape domains.
* Random shape domains (generated via sums of Gaussian functions).
* Square plate with an embedded elliptic hole.
* Shear-dominated Cook's membrane.
* Quarter pressure vessel under internal follower forces.

---

## 💻 Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

For data generation, also install DOLFINx 0.8 (`conda-forge`) and `multiphenicsx`.

### Generate data first if `WINO/data/` is empty

Training still reads geometry/load samples from `WINO/data/*.npz`. If that folder is missing or empty, **run the matching `Phi_FEM/generate_data_*.py` before any example script**. These solvers write train/test npz files into `WINO/data/`.

| Case | Data generator | Default grid / split |
| --- | --- | --- |
| Arbitrary shapes | `Phi_FEM/generate_data_Arbit.py` | 64, 500 train + 100 test |
| Elliptical domains | `Phi_FEM/generate_data_ellip.py` | 64, 500 train + 100 test |
| Plate with hole | `Phi_FEM/generate_data_Hole.py` | 64, 300 train + 100 test |
| Cook's membrane | `Phi_FEM/generate_data_Trapezoid.py` | 51, 1000 train + 100 test |
| Pressure vessel | `Phi_FEM/generate_data_Vessel.py` | 51 (Q9 101), 1000 train + 100 test |

Example:

```bash
python Phi_FEM/generate_data_Arbit.py
```

### Main program

The main WINO training script for each benchmark is **`Examples/<case>/WINO.py`** (physics-informed / data-free residual training on the $\varphi$-FEM weak form).

```bash
python Examples/Arbit_shape/WINO.py
```

Checkpoints, figures, and logs are written under the same case folder: `model/`, `pictures/`, and `result/`. Data stay in `WINO/data/`.

| Script | Purpose |
| --- | --- |
| `WINO.py` | **Main program**: physics-informed WINO training |
| `WINO_Data.py` | data-driven WINO variant |
| `Phi_FEM_FNO.py` | supervised FNO baseline (Phi-FEM labels) |
| `WINO_Gh.py` | plate-with-hole variant with extra ghost-penalty terms |
| `NOWS.py` | Neural Operator Warm Starts for $\varphi$-FEM |
| `Test.py` | load a trained model and report errors / plots |

Other cases follow the same layout (`Elliptical_shape`, `Plate_with_hole`, `Cook_membrane`, `Pressure_vessel`).

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhu6816120wino,
  title={WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains},
  author={Zhu, Bokai and Zhang, Qinghui and Rabczuk, Timon},
  journal={Available at SSRN 6816120}
}
```


