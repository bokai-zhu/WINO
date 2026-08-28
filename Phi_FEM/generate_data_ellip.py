"""
    Elliptical domain (single-layer boundary, rotatable)
"""
import numpy as np
import os
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from ufl import inner, jump, grad, div, dot, avg
import random
import typing

import dolfinx.fem.function
import dolfinx.fem.function
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import ufl
from utils import *
import dolfinx as dfx
from dolfinx.mesh import  CellType
import multiphenicsx.fem
import multiphenicsx.fem.petsc as petsc
import petsc4py.PETSc
import multiphenicsx.fem
import multiphenicsx.fem.petsc
from petsc4py import PETSc

import time

np.pow = np.power

seed = 1603
random.seed(seed)
np.random.seed(seed)
import torch

torch.manual_seed(seed)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")

# Match SNES/KSP settings in generate_data_Hole (relative tolerance of CG linear substeps)
ksp_rtol = 1e-5
cg_rtol = ksp_rtol  # keep old name
# GMRES: Jacobi is often very slow on hyperelastic tangent stiffness; use GAMG + EW; too strict rtol may not be reached within the default max_it
ksp_rtol_gmres = 1e-5
ksp_max_it_gmres = 3000
ksp_gmres_restart = 120

# SNES: reason=-6 is DIVERGED_LINE_SEARCH (backtracking bt finds no descent step); l2 is often more stable for the residual 2-norm
DEFAULT_SNES_MAX_IT = 75
snes_linesearch_default = "l2"
_SNES_LINESEARCH_CHOICES = frozenset(("l2", "bt", "basic", "cp", "nleqerr"))

# ---------------------------------------------------------------------------
# Merged from prepare_data_plus.py: body force, level set, and parameter sampling (create_FG_numpy)
# ---------------------------------------------------------------------------

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _gauss_component(xy, mu0, mu1, sigma_x, sigma_y, amplitude):
    return amplitude * np.exp(
        -(
            ((xy[0] - mu0) ** 2 / (2.0 * sigma_x**2))
            + ((xy[1] - mu1) ** 2 / (2.0 * sigma_y**2))
        )
    )

def call_F(
    xy,
    mu0_u,
    mu1_u,
    sigma_x_u,
    sigma_y_u,
    amplitude_u,
    mu0_v,
    mu1_v,
    sigma_x_v,
    sigma_y_v,
    amplitude_v,
):
    fu = _gauss_component(xy, mu0_u, mu1_u, sigma_x_u, sigma_y_u, amplitude_u)
    fv = _gauss_component(xy, mu0_v, mu1_v, sigma_x_v, sigma_y_v, amplitude_v)
    return np.array([fu, fv])

def call_G(xy, alpha, beta):
    """
    Quasi-linear Dirichlet displacement field (linear-dominant + small nonlinear perturbation):
      g1 = s * [ alpha*x0 + 0.25*beta*y0 + eps*(x0*y0) ]
      g2 = s * [ beta*y0  - 0.25*alpha*x0 + eps*0.5*(x0^2-y0^2) ]
    where x0=x-0.5, y0=y-0.5, and eps is small so the field stays nearly linear.
    """
    x0 = xy[0] - 0.5
    y0 = xy[1] - 0.5
    # Target boundary displacement magnitude about 0.1 ~ 1
    scale = 0.1
    eps = 0.0
    g1 = scale * (alpha * x0 + 0.25 * beta * y0 + 0.5 * eps * (x0**2 - y0**2))
    g2 = scale * (beta * y0 - 0.25 * alpha * x0 + 0.5 * eps * (x0**2 - y0**2))
    return np.array([g1, g2])

def call_phi(xy, a, b, theta_rot, center=(0.5, 0.5)):
    """
    Rotated-ellipse level set (no thick wall):
      phi = x'^2/a^2 + y'^2/b^2 - 1
    where (x', y') are local coordinates after rotating about center by theta_rot.
    where a,b are the ellipse semi-axes (sampled directly).
    """
    cx, cy = center
    x0 = xy[0] - cx
    x1 = xy[1] - cy
    c = np.cos(theta_rot)
    s = np.sin(theta_rot)
    xr = c * x0 + s * x1
    yr = -s * x0 + c * x1
    eps_den = 1e-12
    return xr**2 / (a**2 + eps_den) + yr**2 / (b**2 + eps_den) - 1.0

def _ellipse_axis_aligned_half_extents(a, b, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    wx = np.sqrt((a * c) ** 2 + (b * s) ** 2)
    wy = np.sqrt((a * s) ** 2 + (b * c) ** 2)
    return float(wx), float(wy)

def _sample_ellipse_center(a, b, theta_rot, max_trials=64, margin=0.02):
    """
    Sample the ellipse center at random and check it stays inside [0,1]^2.
    """
    wx, wy = _ellipse_axis_aligned_half_extents(float(a), float(b), float(theta_rot))
    if wx + margin >= 0.5 or wy + margin >= 0.5:
        raise ValueError(
            "Ellipse geometry is too large to fit in the unit square with the given margin: "
            f"wx={wx:.6f}, wy={wy:.6f}, margin={margin:.6f}"
        )

    # Requested center range [0.4, 0.6]; intersected with the 'at least margin from the boundary' constraint
    cmin, cmax = 0.3, 0.7
    lox, hix = max(wx + margin, cmin), min(1.0 - wx - margin, cmax)
    loy, hiy = max(wy + margin, cmin), min(1.0 - wy - margin, cmax)
    if (lox > hix) or (loy > hiy):
        raise ValueError(
            "Center range [0.4,0.6]^2 has empty intersection with the in-bounds+margin constraint."
        )
    for _ in range(int(max_trials)):
        cx = float(np.random.uniform(lox, hix))
        cy = float(np.random.uniform(loy, hiy))
        if (cx - wx >= margin) and (cx + wx <= 1.0 - margin) and (
            cy - wy >= margin
        ) and (
            cy + wy <= 1.0 - margin
        ):
            return cx, cy

    raise RuntimeError("Failed to sample ellipse center: still out of bounds after multiple trials.")

def call_phi_i(xy, x_0, y_0, lx):
    return (-(lx**2) + (xy[0] - x_0) ** 2 + (xy[1] - y_0) ** 2)

def eval_phi(x, y, x_0, y_0, lx):
    return (-(lx**2) + (x - x_0) ** 2 + (y - y_0) ** 2)

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def create_FG_numpy(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])

    # Sample the two components of F independently (original Gaussian formula)
    mu0_u = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1_u = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma_x_u = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    sigma_y_u = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    amplitude_u = np.random.uniform(2.0, 4.0, size=[nb_data, 1]) * np.random.choice(
        [-1, 1], size=[nb_data, 1]
    )
    mu0_v = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    mu1_v = np.random.uniform(0.2, 0.8, size=[nb_data, 1])
    sigma_x_v = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    sigma_y_v = np.random.uniform(0.15, 0.45, size=[nb_data, 1])
    amplitude_v = np.random.uniform(2.0, 4.0, size=[nb_data, 1]) * np.random.choice(
        [-1, 1], size=[nb_data, 1]
    )
    # Scale displacement magnitude to about 0.1~1 (keep sign)
    alpha = np.random.uniform(0.2, 0.8, size=[nb_data, 1]) * np.random.choice(
        [-1, 1], size=[nb_data, 1]
    )
    beta = np.random.uniform(0.2, 0.8, size=[nb_data, 1]) * np.random.choice(
        [-1, 1], size=[nb_data, 1]
    )

    a_axis = np.random.uniform(0.2, 0.4, size=[nb_data, 1])
    b_axis = np.random.uniform(0.2, 0.4, size=[nb_data, 1])
    theta_rot = np.random.uniform(0.0, 2.0 * np.pi, size=[nb_data, 1])
    centers = np.zeros((nb_data, 2), dtype=np.float64)

    # meshgrid flatten order: y first, then x → reshape(n,n) gives phi[i,j]≡(y_i,x_j), matching targets (…,y,x,…)
    F = call_F(
        XXYY,
        mu0_u,
        mu1_u,
        sigma_x_u,
        sigma_y_u,
        amplitude_u,
        mu0_v,
        mu1_v,
        sigma_x_v,
        sigma_y_v,
        amplitude_v,
    )
    F = np.reshape(F, [2, nb_data, nb_vert, nb_vert]).transpose(1, 0, 2, 3)

    Phi = []
    for n in range(nb_data):
        cx, cy = _sample_ellipse_center(
            float(a_axis[n, 0]), float(b_axis[n, 0]), float(theta_rot[n, 0])
        )
        centers[n, 0] = cx
        centers[n, 1] = cy
        phi_n = call_phi(
            XXYY,
            float(a_axis[n, 0]),
            float(b_axis[n, 0]),
            float(theta_rot[n, 0]),
            center=(cx, cy),
        ).reshape((nb_vert, nb_vert))
        Phi.append(phi_n)
    phi = np.array(Phi)

    G = call_G(XXYY, alpha, beta)
    G = np.reshape(G, [2, nb_data, nb_vert, nb_vert]).transpose(1, 0, 2, 3)

    params = np.concatenate(
        [
            alpha,
            beta,
            a_axis,
            b_axis,
            theta_rot,
            centers,
            mu0_u,
            mu1_u,
            sigma_x_u,
            sigma_y_u,
            amplitude_u,
            mu0_v,
            mu1_v,
            sigma_x_v,
            sigma_y_v,
            amplitude_v,
        ],
        axis=1,
    )
    return F, phi, G, params

degV = 2
degPhi = degV + 1

def tensors(u):
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))
    E = 10
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    P = ufl.diff(psi, F)
    return P

def _eval_phi_from_matrix_standalone(phi_sample: np.ndarray, x: np.ndarray) -> np.ndarray:
    """phi_sample (ny, nx), x shape (3, N)."""
    ny, nx = phi_sample.shape
    xq = np.clip(x[0], 0.0, 1.0)
    yq = np.clip(x[1], 0.0, 1.0)
    fx = xq * (nx - 1)
    fy = yq * (ny - 1)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    tx = fx - x0
    ty = fy - y0
    q00 = phi_sample[y0, x0]
    q10 = phi_sample[y0, x1]
    q01 = phi_sample[y1, x0]
    q11 = phi_sample[y1, x1]
    return (
        (1.0 - tx) * (1.0 - ty) * q00
        + tx * (1.0 - ty) * q10
        + (1.0 - tx) * ty * q01
        + tx * ty * q11
    )

def _assign_vector_field_from_grid_linear_to_cg(
    func: dolfinx.fem.Function, field_sample: np.ndarray
) -> None:
    """field_sample (Ny, Nx, 2), interpolate onto the CG target space (same grid convention as PhiFemSolver)."""
    mesh = func.function_space.mesh
    bs = func.function_space.dofmap.index_map_bs
    cdim = int(field_sample.shape[-1])
    if cdim != bs:
        raise ValueError(f"channel count mismatch: field C={cdim}, function bs={bs}")
    vshape = func.function_space.value_shape
    if len(vshape) == 1:
        V_lin = dfx.fem.functionspace(mesh, ("CG", 1, (vshape[0],)))
    else:
        V_lin = dfx.fem.functionspace(mesh, ("CG", 1, tuple(vshape)))
    ny, nx, _ = field_sample.shape
    f_lin = dfx.fem.Function(V_lin)
    coords = f_lin.function_space.tabulate_dof_coordinates().reshape((-1, 3))[:, :2]
    ix = np.rint(np.clip(coords[:, 0], 0.0, 1.0) * (nx - 1)).astype(np.int32)
    iy = np.rint(np.clip(coords[:, 1], 0.0, 1.0) * (ny - 1)).astype(np.int32)
    vals = field_sample[iy, ix, :]
    arr = f_lin.x.array.reshape((-1, bs))
    arr[:] = vals
    f_lin.x.scatter_forward()
    func.interpolate(f_lin)
    func.x.scatter_forward()

def _assign_function_from_grid_linear_to_target(
    func: dolfinx.fem.Function, field_sample: np.ndarray
) -> None:
    """Same convention as generate_data_Hole.PhiFemSolver._assign_function_from_grid_linear_to_target."""
    _assign_vector_field_from_grid_linear_to_cg(
        func, np.asarray(field_sample, dtype=np.float64)
    )

def _vector_from_grid_eval(field_hw2: np.ndarray):
    """Grid field (Ny, Nx, 2) → a vector field callable by ``fem.Function.interpolate``."""
    f = np.asarray(field_hw2, dtype=np.float64)

    def ev(x):
        xx = np.asarray(x)
        return np.vstack(
            [
                _eval_phi_from_matrix_standalone(f[:, :, 0], xx.reshape(3, -1)),
                _eval_phi_from_matrix_standalone(f[:, :, 1], xx.reshape(3, -1)),
            ]
        )

    return ev

def _scalar_from_grid_eval(phi_hw: np.ndarray):
    """Grid scalar field (Ny, Nx) → interpolate callable."""
    p = np.asarray(phi_hw, dtype=np.float64)

    def ev(x):
        xx = np.asarray(x)
        return _eval_phi_from_matrix_standalone(p, xx.reshape(3, -1))

    return ev

def _build_submesh_V_dx(phi_sample: np.ndarray, nb_cell: int, deg_v: int):
    """Same macro partition as solve_one + cut away the exterior submesh where phi>0; return dx and bottom-edge Dirichlet dofs."""
    mesh_macro = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [1, 1]]),
        np.array([nb_cell, nb_cell]),
        cell_type=CellType.quadrilateral,
    )
    cell_dim = mesh_macro.geometry.dim
    phi_eval = lambda x: _eval_phi_from_matrix_standalone(
        phi_sample, x.reshape((3, -1))
    )
    all_entities = np.arange(
        mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32
    )
    cells_outside = dfx.mesh.locate_entities(
        mesh_macro, cell_dim, lambda x: phi_eval(x) > 0.0
    )
    interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
    mesh = dfx.mesh.create_submesh(
        mesh_macro, mesh_macro.topology.dim, interior_entities_macro
    )[0]
    facet_dim = mesh.geometry.dim - 1
    quad_meta = {"quadrature_degree": 5}
    dx_m = ufl.Measure("dx", domain=mesh, metadata=quad_meta)
    V = dfx.fem.functionspace(mesh, ("CG", deg_v, (cell_dim,)))
    lower_facets = dfx.mesh.locate_entities_boundary(
        mesh, facet_dim, lambda x: np.isclose(x[1], 0.0)
    )
    boundary_dofs_low = fem.locate_dofs_topological(V, facet_dim, lower_facets)
    return mesh, V, dx_m, boundary_dofs_low

def compute_errors(
    phi_sample: np.ndarray,
    u_reference: np.ndarray,
    u_predicted: np.ndarray,
    nb_cell: int,
    deg_v: int = 1,
) -> tuple[float, float]:
    """
    Compute on the dolfinx submesh (relative to ``u_reference``):

    1. **H¹ seminorm error**: \\sqrt{\\int_\\Omega \\|\\nabla(u_{pred}-u_{ref})\\|^2 \\,dx}

    2. **Discrete energy norm** (consistent with \\(\\sqrt{e^\\top K e}\\)):
       At ``u_reference``, take the variational derivative of the **volume** weak form ``\\int P(u):\\nabla v\\,dx`` to get the tangent stiffness
       ``K`` (homogeneous Dirichlet on the bottom edge); ``e`` is the dof vector of ``u_predicted - u_reference``;
       ``energy_norm = \\sqrt{e^\\top K e}``.

    Note: ``K`` contains only the Neo-Hookean volume term from ``tensors`` in this file, not the full Phi-FEM block Jacobian (y/p, band stabilization, etc.).

    Parameters
    ----
    phi_sample : (Ny, Nx), level set on grid nodes; \\Omega = \\{\\phi \\le 0\\} (same as ``Phi.not_omega``).
    u_reference, u_predicted : (Ny, Nx, 2)
    nb_cell : number of macro cells (same as ``PhiFemSolver(nb_cell=nb_vert-1)``)
    deg_v : degree of the displacement space; default 1 matches ``degV`` in this file.
    """
    if phi_sample.ndim != 2:
        raise ValueError(f"phi_sample must be 2D, got {phi_sample.shape}")
    if u_reference.shape != u_predicted.shape or u_reference.shape[-1] != 2:
        raise ValueError(
            f"u_reference/u_predicted must be (Ny,Nx,2), got {u_reference.shape}"
        )
    mesh, V, dx_m, boundary_dofs_low = _build_submesh_V_dx(
        phi_sample, nb_cell, deg_v
    )
    comm = mesh.comm

    u_ref = fem.Function(V)
    u_pred = fem.Function(V)
    _assign_vector_field_from_grid_linear_to_cg(u_ref, u_reference.astype(np.float64))
    _assign_vector_field_from_grid_linear_to_cg(u_pred, u_predicted.astype(np.float64))

    e_fn = fem.Function(V)
    e_fn.x.array[:] = u_pred.x.array - u_ref.x.array
    e_fn.x.scatter_forward()

    # --- H1 seminorm error ---
    form_h1 = fem.form(inner(grad(e_fn), grad(e_fn)) * dx_m)
    h1_sq = fem.assemble_scalar(form_h1)
    h1_sq = comm.allreduce(h1_sq, op=MPI.SUM)
    h1_err = float(np.sqrt(max(h1_sq, 0.0)))

    form_h1_ref = fem.form(inner(grad(u_ref), grad(u_ref)) * dx_m)
    h1_ref_sq = comm.allreduce(fem.assemble_scalar(form_h1_ref), op=MPI.SUM)
    h1_ref = float(np.sqrt(max(h1_ref_sq, 0.0)))

    # --- sqrt(e^T K e), K = d/du ∫ P(u):∇v dx at u_ref ---
    u_zero = np.array([0.0, 0.0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_zero, boundary_dofs_low, V)
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    L_vol = inner(tensors(u_ref), grad(v)) * dx_m
    J_form = fem.form(ufl.derivative(L_vol, u_ref, du))
    J_mat = assemble_matrix(J_form, bcs=[bc])
    J_mat.assemble()

    e_vec = u_pred.x.petsc_vec.copy()
    e_vec.axpy(-1.0, u_ref.x.petsc_vec)
    Ke = J_mat.createVecLeft()
    J_mat.mult(e_vec, Ke)
    energy_sq = float(e_vec.dot(Ke))
    energy_norm = float(np.sqrt(max(energy_sq, 0.0)))

    Ku = J_mat.createVecLeft()
    J_mat.mult(u_ref.x.petsc_vec, Ku)
    ref_energy_sq = float(u_ref.x.petsc_vec.dot(Ku))
    ref_energy_norm = float(np.sqrt(max(ref_energy_sq, 0.0)))

    return h1_err / h1_ref, energy_norm / ref_energy_norm

class Phi:
    def __init__(self, a_axis, b_axis, theta_rot, center=(0.5, 0.5)) -> None:
        self.a_axis = float(a_axis)
        self.b_axis = float(b_axis)
        self.theta_rot = float(theta_rot)
        self.center = center

    def eval(self, x):
        return call_phi(
            x,
            self.a_axis,
            self.b_axis,
            self.theta_rot,
            center=self.center,
        )

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) > 0.0

class GExpr:
    def __init__(self, alpha_G, beta_G):
        self.alpha_G = alpha_G
        self.beta_G = beta_G

    def eval(self, x):
        return call_G(x, self.alpha_G, self.beta_G)

class FExpr:
    def __init__(
        self,
        mu0_u,
        mu1_u,
        sigma_x_u,
        sigma_y_u,
        amplitude_u,
        mu0_v,
        mu1_v,
        sigma_x_v,
        sigma_y_v,
        amplitude_v,
    ):
        self.mu0_u = mu0_u
        self.mu1_u = mu1_u
        self.sigma_x_u = sigma_x_u
        self.sigma_y_u = sigma_y_u
        self.amplitude_u = amplitude_u
        self.mu0_v = mu0_v
        self.mu1_v = mu1_v
        self.sigma_x_v = sigma_x_v
        self.sigma_y_v = sigma_y_v
        self.amplitude_v = amplitude_v

    def eval(self, x):
        return call_F(
            x,
            self.mu0_u,
            self.mu1_u,
            self.sigma_x_u,
            self.sigma_y_u,
            self.amplitude_u,
            self.mu0_v,
            self.mu1_v,
            self.sigma_x_v,
            self.sigma_y_v,
            self.amplitude_v,
        )

def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol

class NonLinearPhiFEM:
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        F: list[ufl.Form],
        J: list[list[ufl.Form]],
        solutions,
        bcs: list[dolfinx.fem.DirichletBC],
        restriction,
        spaces,
        P: typing.Optional[list[list[ufl.Form]]] = None,
    ) -> None:
        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        self._restriction = restriction
        self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction
        )
        self._solutions = solutions
        self._spaces = spaces
        self._bcs = bcs
        self._P = P

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guesses provided in `self._solutions`,
        properly stacked together and restricted in a single block vector.
        """
        x = multiphenicsx.fem.petsc.create_vector_block(
            self._F, restriction=self._restriction
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [VV.dofmap for VV in self._spaces], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    x_wrapper_local[:] = sub_solution_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT,
            mode=petsc4py.PETSc.ScatterMode.FORWARD,
        )
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            x, [VV.dofmap for VV in self._spaces], self._restriction
        ) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    sub_solution_local[:] = x_wrapper_local

    def obj(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()  # type: ignore[no-any-return]

    def F(  # type: ignore[no-any-unimported]
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        F_vec: petsc4py.PETSc.Vec,
    ) -> None:
        """Assemble the residual."""
        self.update_solutions(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector_block(  # type: ignore[misc]
            F_vec,
            self._F,
            self._J,
            self._bcs,
            x0=x,
            scale=-1.0,
            restriction=self._restriction,
            restriction_x0=self._restriction,
        )

    def J(  # type: ignore[no-any-unimported]
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        J_mat: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat,
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix_block(
            J_mat,
            self._J,
            self._bcs,
            diagonal=1.0,  # type: ignore[arg-type]
            restriction=(self._restriction, self._restriction),
        )
        J_mat.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            multiphenicsx.fem.petsc.assemble_matrix_block(
                P_mat,
                self._P,
                self._bcs,
                diagonal=1.0,  # type: ignore[arg-type]
                restriction=(self._restriction, self._restriction),
            )
            P_mat.assemble()

class PhiFemSolver:
    def __init__(
        self,
        nb_cell,
        Phi_data,
        G_data,
        F_data,
        init_guess_data=None,
        linear_solver="gmres",
        deg_v=None,
        n_load_steps_f=1,
        snes_max_it=None,
        snes_linesearch=None,
        snes_retry_basic_on_ls_fail=True,
    ):
        self.N = N = nb_cell
        self.Phi_data = Phi_data
        self.G_data = G_data
        self.F_data = F_data
        self.init_guess_data = init_guess_data
        self._init_guess_channels = None
        self.linear_solver = linear_solver
        self.deg_v = int(degV if deg_v is None else deg_v)
        self.deg_phi = self.deg_v + 1
        # Number of body-force continuation steps (continuation on f)
        self.n_load_steps_f = n_load_steps_f
        self.snes_max_it = int(
            snes_max_it if snes_max_it is not None else DEFAULT_SNES_MAX_IT
        )
        _ls = snes_linesearch if snes_linesearch is not None else snes_linesearch_default
        if _ls not in _SNES_LINESEARCH_CHOICES:
            raise ValueError(
                f"snes_linesearch must be one of {sorted(_SNES_LINESEARCH_CHOICES)}, got: {_ls!r}"
            )
        self.snes_linesearch = _ls
        self.snes_retry_basic_on_ls_fail = bool(snes_retry_basic_on_ls_fail)
        if self.linear_solver not in ("cg", "gmres", "lu"):
            raise ValueError(
                f"linear_solver must be 'cg', 'gmres', or 'lu', got: {self.linear_solver}"
            )
        if init_guess_data is not None:
            if init_guess_data.ndim != 4:
                raise ValueError(
                    f"init_guess must be 4D (batch, Ny, Nx, C), got shape={init_guess_data.shape}"
                )
            c = int(init_guess_data.shape[-1])
            if c == 2:
                self._init_guess_channels = 2
            elif c == 8:
                self._init_guess_channels = 8
            else:
                raise ValueError(
                    f"last dim of init_guess must be 2 (u only) or 8 (u+y+p), got C={c}"
                )
            if init_guess_data.shape[0] != Phi_data.shape[0]:
                raise ValueError("init_guess batch size does not match Phi")
            if init_guess_data.shape[1:3] != Phi_data.shape[1:3]:
                raise ValueError(
                    f"init_guess Ny,Nx do not match Phi: {init_guess_data.shape[1:3]} vs {Phi_data.shape[1:3]}"
                )
        if F_data.shape[0] != Phi_data.shape[0]:
            raise ValueError("F_data batch size does not match Phi_data")
        if G_data.shape[0] != Phi_data.shape[0]:
            raise ValueError("G_data batch size does not match Phi_data")
        if (
            G_data.shape[1:3] != Phi_data.shape[1:3]
            or F_data.shape[1:3] != Phi_data.shape[1:3]
        ):
            raise ValueError("G_data/F_data Ny,Nx do not match Phi")
        if int(F_data.shape[-1]) != 2 or int(G_data.shape[-1]) != 2:
            raise ValueError("last dim of G_data and F_data must be 2")

        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
            cell_type=CellType.quadrilateral
        )
        # Macro output: CG1 on quads (Q4 / bilinear), same resolution as the nb_vert sample grid
        self.V_macro = dolfinx.fem.functionspace(
            self.mesh_macro, ("CG", 1, (self.mesh_macro.geometry.dim,))
        )
        self.V_macro_tensor = dolfinx.fem.functionspace(
            self.mesh_macro,
            ("CG", 1, (self.mesh_macro.geometry.dim, self.mesh_macro.geometry.dim)),
        )
        self.V_macro_scalar = dolfinx.fem.functionspace(
            self.mesh_macro, ("CG", 1)
        )

        coords = self.V_macro.tabulate_dof_coordinates()
        coords = coords.reshape((-1, 3))
        coords = coords[:, :2]

        coords = np.round(coords, decimals=5)
        self.sorted_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.padding = 1e-14

    def make_matrix(self, expr, V, V_target):
        """Convert an expression of degree k to a matrix with nodal values.

        Args:
            expr (FEniCS expression): the expression to convert

        Returns:
            np array : a matrix of size N+1 * N+1
        """

        expr.x.scatter_forward()
        u2 = dolfinx.fem.Function(V_target)
        u1_2_u2_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            u2.function_space.mesh,
            u2.function_space.element,
            V.mesh,
            padding=self.padding,
        )

        u2.interpolate(expr, nmm_interpolation_data=u1_2_u2_nmm_data)
        u2.x.scatter_forward()
        # Get block size (vector=2, tensor=4)
        bs = V_target.dofmap.index_map_bs

        res = u2.x.array  # or u2.vector.array

        # Reshape to (Num_Nodes, Block_Size)
        # e.g. u: (N, 2), y: (N, 4)
        res_reshaped = res.reshape((-1, bs))

        # Sort by coordinates
        res_sorted = res_reshaped[self.sorted_indices]
        # Reshape to image layout (Block_Size, H, W)
        # H = W = self.N + 1
        # grid_dim = self.N + 1
        total_nodes = res_sorted.shape[0]
        grid_dim = int(np.round(np.sqrt(total_nodes)))

        # First convert to (H, W, Channels)
        expr_sorted = res_sorted.reshape(grid_dim, grid_dim, bs)
        expr_sorted = np.transpose(expr_sorted, (2, 1, 0))
        return expr_sorted

    def _sorted_vector_to_matrix(self, vector_array, bs):
        """Internal helper: reshape a flat vector and sort it into a matrix"""
        # Reshape to (Num_Nodes, Block_Size)
        res_reshaped = vector_array.reshape((-1, bs))

        # Sort
        res_sorted = res_reshaped[self.sorted_indices]

        # Compute grid dimensions dynamically
        total_nodes = res_sorted.shape[0]
        grid_dim = int(np.round(np.sqrt(total_nodes)))

        # Reshape to (H, W, Channels)
        expr_sorted = res_sorted.reshape(grid_dim, grid_dim, bs)

        # Transpose to (Channels, H, W)
        return np.transpose(expr_sorted, (2, 1, 0))

    def get_analytical_matrix(self, func_eval, V_target):
        """Interpolate an analytic function directly onto the macro mesh"""
        u_target = dolfinx.fem.Function(V_target)
        u_target.interpolate(func_eval)
        u_target.x.scatter_forward()

        return self._sorted_vector_to_matrix(u_target.x.array, V_target.dofmap.index_map_bs)

    def solve_one(self, i):
        self.index = i
        """Computation of phiFEM

        Args:
            i (int): index of the problem to solve

        Returns:
            tuple: (u_mat, y_mat, p_mat, load_loop_time, total_cg_iters, total_newton_iters)
            Same as generate_data_Hole; y_mat is the stress tensor field, p_mat is body force f on the macro grid.
        """
        newton_residual_history: list[float] = []
        total_newton_iters = 0
        total_cg_iters = 0
        phi_sample = np.asarray(self.Phi_data[i, :, :, 0], dtype=np.float64)
        g_sample = np.asarray(self.G_data[i], dtype=np.float64)
        f_sample = np.asarray(self.F_data[i], dtype=np.float64)

        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1
        vertices_dim = 0

        def _marked_outside(x):
            v = _eval_phi_from_matrix_standalone(
                phi_sample, np.asarray(x).reshape(3, -1)
            )
            return v > 0.0

        all_entities = np.arange(
            self.mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32
        )
        cells_outside = dfx.mesh.locate_entities(
            self.mesh_macro, cell_dim, _marked_outside
        )
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)

        mesh = dfx.mesh.create_submesh(
            self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro
        )[0]

        V = dfx.fem.functionspace(mesh, ("CG", self.deg_v, (cell_dim,)))
        V_phi = dfx.fem.functionspace(mesh, ("CG", self.deg_phi))
        Z_N = dfx.fem.functionspace(
            mesh, ("CG", self.deg_v, (cell_dim, cell_dim))
        )
        if self.deg_v == 1:
            Q_N = dfx.fem.functionspace(
                mesh, ("DG", self.deg_v - 1, (cell_dim,))
            )
        else:
            Q_N = dfx.fem.functionspace(
                mesh, ("CG", self.deg_v - 1, (cell_dim,))
            )

        dofs_V = np.arange(
            0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        )
        spaces = [V]
        restricts = [dofs_V]
        restriction = [multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)]
        neumann_cells, neumann_facets = [], []
        # Level-set band cells: macro cells crossed by the domain boundary φ=0 (not a 'hole' geometry; the boundary band of Ω={φ≤0})
        cut_band_cells = []

        mesh.topology.create_connectivity(cell_dim, facet_dim)
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, vertices_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, vertices_dim)
        interior_entities = np.arange(
            mesh.topology.index_map(cell_dim).size_global, dtype=np.int32
        )

        c_to_v_map = np.reshape(c_to_v.array, (-1, 4))
        assert c_to_v_map.shape[0] == len(interior_entities)
        points = mesh.geometry.x.T
        phi_values = _eval_phi_from_matrix_standalone(phi_sample, points)

        phi_0 = phi_values
        phi_cells = phi_0[c_to_v_map]
        cells_boundary_all = (
                ((phi_cells[:, 0] * phi_cells[:, 1]) <= 0.0)
                | ((phi_cells[:, 1] * phi_cells[:, 2]) <= 0.0)
                | ((phi_cells[:, 2] * phi_cells[:, 3]) <= 0.0)
                | ((phi_cells[:, 3] * phi_cells[:, 0]) <= 0.0)
                | (near(phi_cells[:, 0] * phi_cells[:, 1], 0.0))
                | (near(phi_cells[:, 1] * phi_cells[:, 2], 0.0))
                | (near(phi_cells[:, 2] * phi_cells[:, 3], 0.0))
                | (near(phi_cells[:, 3] * phi_cells[:, 0], 0.0))
        )
        cells_boundary = np.where(cells_boundary_all == True)[0]
        cut_band_cells.append(cells_boundary)

        neumann_facets_measure, neumann_facets_stab_measure, neumann_cells_measure = (
            [],
            [],
            [],
        )
        neumann_values, neumann_stab_values = [], []
        c2f_map = np.reshape(c_to_f.array, (-1, 4))

        boundary_cells_0 = cut_band_cells[0]
        if len(boundary_cells_0) > 0:
            neumann_cells_0 = np.unique(boundary_cells_0)
            # Same as Poisson phi-FEM: all facets of boundary cells
            boundary_facets = np.unique(c2f_map[neumann_cells_0].flatten())
            mesh.topology.create_connectivity(facet_dim, cell_dim)
            f_to_c = mesh.topology.connectivity(facet_dim, cell_dim)
            imap_f = mesh.topology.index_map(facet_dim)
            exterior_facets = np.array(
                [
                    f
                    for f in range(imap_f.size_local)
                    if f_to_c.links(f).size == 1
                ],
                dtype=np.int32,
            )
            # dS(1): same as Poisson phi-FEM, all facets of boundary cells
            neumann_stab_facets_0 = boundary_facets
            # ds(4): facets of boundary cells that lie on submesh ∂Ω (ellipse boundary and outer frame)
            neumann_facets_0 = np.intersect1d(boundary_facets, exterior_facets)

            neumann_facets_measure.append(neumann_facets_0)
            neumann_facets_stab_measure.append(neumann_stab_facets_0)
            neumann_cells_measure.append(neumann_cells_0)
            neumann_values.append(4)
            neumann_stab_values.append(1)

        # create meshtags for cells
        full_neumann_values = []
        for j in range(len(neumann_cells_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_cells_measure[j]), dtype=np.intc
            )
            full_neumann_values.append(values_Neumann)

        if len(neumann_cells_measure) > 0:
            values_cells = np.hstack(full_neumann_values)
            entities_cells = np.hstack(neumann_cells_measure)
            sorted_cells = np.argsort(entities_cells)
            subdomains_cell = dfx.mesh.meshtags(
                mesh,
                cell_dim,
                entities_cells[sorted_cells],
                values_cells[sorted_cells],
            )
        else:
            subdomains_cell = dfx.mesh.meshtags(
                mesh,
                cell_dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.intc),
            )

        full_neumann_values_facets = []
        full_neumann_entities_facets = []
        for j in range(len(neumann_facets_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_facets_measure[j]), dtype=np.intc
            )
            full_neumann_values_facets.append(values_Neumann)
            full_neumann_entities_facets.append(neumann_facets_measure[j])

        entities_ds = (
            np.hstack(full_neumann_entities_facets)
            if full_neumann_entities_facets
            else np.array([], dtype=np.int32)
        )
        values_ds = (
            np.hstack(full_neumann_values_facets)
            if full_neumann_values_facets
            else np.array([], dtype=np.intc)
        )
        if entities_ds.size > 0:
            sorted_ds_facets = np.argsort(entities_ds)
            subdomains_facet_ds = dfx.mesh.meshtags(
                mesh,
                facet_dim,
                entities_ds[sorted_ds_facets],
                values_ds[sorted_ds_facets],
            )
        else:
            subdomains_facet_ds = dfx.mesh.meshtags(
                mesh,
                facet_dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.intc),
            )

        full_stab_values_facets = []
        full_stab_entities_facets = []
        for j in range(len(neumann_facets_stab_measure)):
            values_Neumann_stab = neumann_stab_values[j] * np.ones(
                len(neumann_facets_stab_measure[j]), dtype=np.intc
            )
            full_stab_values_facets.append(values_Neumann_stab)
            full_stab_entities_facets.append(neumann_facets_stab_measure[j])

        entities_dS = (
            np.hstack(full_stab_entities_facets)
            if full_stab_entities_facets
            else np.array([], dtype=np.int32)
        )
        values_dS = (
            np.hstack(full_stab_values_facets)
            if full_stab_values_facets
            else np.array([], dtype=np.intc)
        )
        if entities_dS.size > 0:
            sorted_dS_facets = np.argsort(entities_dS)
            subdomains_facet = dfx.mesh.meshtags(
                mesh,
                facet_dim,
                entities_dS[sorted_dS_facets],
                values_dS[sorted_dS_facets],
            )
        else:
            subdomains_facet = dfx.mesh.meshtags(
                mesh,
                facet_dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.intc),
            )

        V2 = dolfinx.fem.functionspace(mesh, ("CG", 2, (2,)))
        f_expr = dolfinx.fem.Function(V2)
        f_expr.interpolate(_vector_from_grid_eval(f_sample))
        load_factor_f = dolfinx.fem.Constant(mesh, np.array(0.0, dtype=np.float64))

        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=subdomains_cell,
            metadata={"quadrature_degree": 4},
        )
        ds = ufl.Measure(
            "ds",
            domain=mesh,
            subdomain_data=subdomains_facet_ds,
            metadata={"quadrature_degree": 4},
        )
        dS = ufl.Measure(
            "dS",
            domain=mesh,
            subdomain_data=subdomains_facet,
            metadata={"quadrature_degree": 4},
        )
        if len(neumann_stab_values) > 0:
            dS_stab_measure = dS(neumann_stab_values[0])
        else:
            dS_stab_measure = dS
        if len(neumann_values) > 0:
            dx_band_measure = dx(neumann_values[0])
            ds_band_measure = ds(neumann_values[0])
        else:
            dx_band_measure = dx
            ds_band_measure = ds

        gamma_div = 0.01
        sigma_N = 0.01

        u_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
        if self.init_guess_data is not None:
            init_sample = self.init_guess_data[i]
            u_init_sample = init_sample[:, :, 0:2]
            _assign_function_from_grid_linear_to_target(u_split[0], u_init_sample)
        v_split = [ufl.TestFunction(VVV) for VVV in spaces]
        du_split = [ufl.TrialFunction(VVV) for VVV in spaces]

        phi = dolfinx.fem.Function(V_phi)
        phi.interpolate(_scalar_from_grid_eval(phi_sample))

        g = dolfinx.fem.Function(V)
        _assign_function_from_grid_linear_to_target(g, g_sample)

        u1 = u_split[0] * phi + g
        v1 = v_split[0] * phi
        Pu1 = tensors(u1)
        Pv1 = ufl.derivative(Pu1, u_split[0], v_split[0])
        Gh1 = (
            sigma_N
            * ufl.avg(h)
            * ufl.inner(ufl.jump(Pu1, n), ufl.jump(Pv1, n))
            * dS_stab_measure
        )

        F = [0.0 for k in range(len(spaces))]

        dx_full_omega_1 = dx

        au1v1 = ufl.inner(Pu1, ufl.grad(v1)) * dx_full_omega_1
        F[0] += au1v1 - load_factor_f * ufl.inner(f_expr, v1) * dx_full_omega_1 + Gh1

        dx_Neumann_omega_1 = dx_band_measure
        ds_Neumann_omega_1 = ds_band_measure

        dsi = ds_Neumann_omega_1
        dxi = dx_Neumann_omega_1
        F[0] -= ufl.inner(ufl.dot(Pu1, n), v1) * dsi
        F[0] += gamma_div * (
            ufl.inner(ufl.div(Pu1), ufl.div(Pv1)) * dxi
            + load_factor_f * ufl.inner(f_expr, ufl.div(Pv1)) * dxi
        )

        J = [
            [
                ufl.derivative(F[i], u_split[j], du_split[j])
                for j in range(len(u_split))
            ]
            for i in range(len(F))
        ]

        problem = NonLinearPhiFEM(F, J, tuple(u_split), [], restriction, spaces)
        F_vec = multiphenicsx.fem.petsc.create_vector_block(
            problem._F, restriction=restriction
        )
        J_mat = multiphenicsx.fem.petsc.create_matrix_block(
            problem._J, restriction=(restriction, restriction)
        )
        snes = petsc4py.PETSc.SNES().create(mesh.comm)
        snes.setTolerances(max_it=int(self.snes_max_it))
        snes.setType("newtonls")
        opts = PETSc.Options()
        opts["snes_linesearch_type"] = str(self.snes_linesearch)
        if self.snes_linesearch == "basic":
            opts["snes_linesearch_damping"] = "0.8"
        ksp = snes.getKSP()
        if self.linear_solver == "lu":
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
        elif self.linear_solver == "cg":
            ksp.setType("cg")
            ksp.getPC().setType("jacobi")
            ksp.setTolerances(rtol=ksp_rtol)
        else:
            ksp.setType("gmres")
            try:
                ksp.setGMRESRestart(int(ksp_gmres_restart))
            except Exception:
                pass
            pc = ksp.getPC()
            try:
                pc.setType("gamg")
            except Exception:
                pc.setType("jacobi")
            ksp.setTolerances(
                rtol=float(ksp_rtol_gmres),
                atol=1e-12,
                max_it=int(ksp_max_it_gmres),
            )
            # Relax inner rtol from the Newton residual to reduce DIVERGED_LINEAR_SOLVE (-3) from chasing 1e-5 until max_it
            snes.setUseEW(True)
        snes.setObjective(problem.obj)
        snes.setFunction(problem.F, F_vec)
        snes.setJacobian(problem.J, J=J_mat, P=None)

        solution = problem.create_snes_solution()

        def _snes_monitor(_snes, its, fnorm, *args):
            # During body-force continuation SNES fnorm is for the scaled load, so step changes look like false oscillations. Plot ||F(u)|| at full load f
            if int(self.n_load_steps_f) <= 1:
                newton_residual_history.append(float(fnorm))
                return
            lf_save = np.asarray(load_factor_f.value, dtype=np.float64).copy()
            load_factor_f.value = np.array(1.0, dtype=np.float64)
            try:
                r_full = float(problem.obj(_snes, solution))
            finally:
                load_factor_f.value = lf_save
            newton_residual_history.append(r_full)

        snes.setMonitor(_snes_monitor)
        converged_reason = -1
        solved_ok = True
        t_load_loop_start = time.time()

        # Last converged solution (load lf_current); on failure restore from here, reduce the increment, and retry
        solution_last_good = solution.duplicate()
        solution.copy(solution_last_good)

        lf_current = 0.0
        lf_target = 1.0
        n_seg = max(int(self.n_load_steps_f), 1)
        delta_init = (lf_target - lf_current) / float(n_seg)
        # On non-convergence, halve the increment; continue only if it stays above this threshold, else abort (avoid infinite refinement)
        delta_min = 0.01
        delta = delta_init

        while lf_current < lf_target - 1e-14:
            lf_next = min(lf_current + delta, lf_target)
            solution_last_good.copy(solution)
            problem.update_solutions(solution)
            load_factor_f.value = np.array(lf_next, dtype=np.float64)
            snes.solve(None, solution)
            converged_reason = snes.getConvergedReason()
            if (
                int(converged_reason) == -6
                and self.snes_retry_basic_on_ls_fail
                and str(self.snes_linesearch) != "basic"
            ):
                try:
                    ls = snes.getLineSearch()
                    ls.setType("basic")
                    try:
                        ls.setDamping(0.75)
                    except Exception:
                        opts["snes_linesearch_damping"] = "0.75"
                except Exception:
                    pass
                snes.solve(None, solution)
                converged_reason = snes.getConvergedReason()
            try:
                total_newton_iters += int(snes.getIterationNumber())
            except Exception:
                pass
            try:
                total_cg_iters += int(snes.getLinearSolveIterations())
            except Exception:
                pass
            if converged_reason < 0:
                new_delta = 0.5 * delta
                print(
                    f"SNES did not converge: reason={converged_reason}, "
                    f"load [{lf_current:.6g} -> {lf_next:.6g}], delta={delta:.3e}"
                )
                if int(converged_reason) == -3:
                    print(
                        "  (reason=-3: KSP/GMRES did not converge inside the Newton step;"
                        "try PhiFemSolver(..., linear_solver='lu') or reduce the load / check material parameters)"
                    )
                if int(converged_reason) == -6:
                    print(
                        "  (reason=-6: line search failed DIVERGED_LINE_SEARCH;"
                        "try increasing n_load_steps_f, snes_linesearch='cp', or a weaker load)"
                    )
                if new_delta <= delta_min:
                    print(
                        f"load increment halved to {new_delta:.3e} <= {delta_min:g}; still not converging, abort."
                    )
                    solved_ok = False
                    break
                print(f"  -> reduce load increment: delta {delta:.3e} -> {new_delta:.3e}, retry.")
                delta = new_delta
                continue

            problem.update_solutions(solution)
            solution.copy(solution_last_good)
            lf_current = lf_next
            delta = delta_init

        load_loop_time = time.time() - t_load_loop_start

        self._last_newton_residual_history = newton_residual_history

        if not solved_ok:
            return (
                None,
                None,
                None,
                load_loop_time,
                total_cg_iters,
                total_newton_iters,
            )

        u_sol = u_split[0]
        u_mat = self.make_matrix(u_sol, V, self.V_macro)

        P_ufl = tensors(u_sol * phi + g)
        P_u_func = dolfinx.fem.Function(Z_N)
        P_expr = dolfinx.fem.Expression(P_ufl, Z_N.element.interpolation_points())
        P_u_func.interpolate(P_expr)
        y_mat = self.make_matrix(P_u_func, Z_N, self.V_macro_tensor)

        p_mat = self.get_analytical_matrix(
            _vector_from_grid_eval(f_sample), self.V_macro
        )

        return (
            u_mat,
            y_mat,
            p_mat,
            load_loop_time,
            total_cg_iters,
            total_newton_iters,
        )

    def solve_several(self):
        U_list = []
        Y_list = []
        P_list = []
        load_loop_times = []
        total_cg_iters_all = 0
        total_newton_iters_all = 0
        self.newton_residual_histories = []
        nb = len(self.Phi_data)
        for i in range(nb):
            print(f"Data : {i}/{nb}")
            u_mat, y_mat, p_mat, t_load, cg_i, newt_i = self.solve_one(i)
            self.newton_residual_histories.append(
                list(getattr(self, "_last_newton_residual_history", []))
            )
            load_loop_times.append(t_load)
            total_cg_iters_all += int(cg_i)
            total_newton_iters_all += int(newt_i)
            if u_mat is None:
                continue
            U_list.append(u_mat)
            Y_list.append(y_mat)
            P_list.append(p_mat)

        if len(U_list) == 0:
            return (
                None,
                None,
                None,
                np.array(load_loop_times),
                total_cg_iters_all,
                total_newton_iters_all,
            )
        return (
            np.stack(U_list),
            np.stack(Y_list),
            np.stack(P_list),
            np.array(load_loop_times),
            total_cg_iters_all,
            total_newton_iters_all,
        )

def generate_parameters(nb_data, nb_vert):
    """
    Build phi, G, F and FEM params on the sample grid in memory; do not write to disk.
    F,G: (n, nb_vert, nb_vert, 2); phi: (n, nb_vert, nb_vert, 1)
    """
    F, phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    phi_out = phi[:, :, :, np.newaxis]
    F_hwc = np.transpose(F, (0, 2, 3, 1))
    G_hwc = np.transpose(G, (0, 2, 3, 1))
    return phi_out, G_hwc, F_hwc, params

def create_parameters(nb_vert=64, nb_training_shapes=300):
    """Counterpart of generate_data_Hole.create_parameters; Arbit also returns body force F."""
    phi_out, G, F, _params = generate_parameters(nb_training_shapes, nb_vert)
    return phi_out, G, F

def go(Phi, G, F, init_guess=None, nb_vert=64, n_train=None, n_test=None):
    """
    Main function to generate data.
    Same flow as ``generate_data_Hole.go``; this example uses inputs ``[Phi, G, F]``.

    init_guess: optional, (batch, Ny, Nx, 2) for u only, or (batch, Ny, Nx, 8) for u+y+p (only the first two components of u are used).
    n_train / n_test: if both are positive ints, save two npz files in batch order (first n_train train, then n_test test);
        if both are None, save a single npz.
    """
    save = True
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(base_dir), "data")

    if (n_train is None) ^ (n_test is None):
        raise ValueError("n_train and n_test must both be set or both be None")

    ti0 = time.time()
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        Phi_data=Phi,
        G_data=G,
        F_data=F,
        init_guess_data=init_guess,
        linear_solver="lu",
        n_load_steps_f=4,
        deg_v=1,
    )
    U, Y, P, _, total_cg_iters, total_newton_iters = solver.solve_several()
    if solver.linear_solver in ("cg", "gmres"):
        print(f"[KSP] total linear iterations={total_cg_iters}")
    print(f"[Newton] total iterations={total_newton_iters}")
    duration = time.time() - ti0
    print("duration to solve u:", duration)

    if U is None:
        print("solve_several: no successful samples, skip saving")
        return

    U = U.transpose((0, 3, 2, 1))
    Y = Y.transpose((0, 3, 2, 1))
    P = P.transpose((0, 3, 2, 1))

    if save:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        model_inputs = np.concatenate([Phi, G, F], axis=-1)
        model_targets = U
        batch_num = model_inputs.shape[0]

        if n_train is not None and n_test is not None:
            need = int(n_train) + int(n_test)
            if batch_num < need:
                print(
                    f"Warning: batch size {batch_num} < train+test need {need}, "
                    "train/test will be truncated to the available count."
                )
            n_tr = min(int(n_train), batch_num)
            rest = batch_num - n_tr
            n_te = min(int(n_test), rest)
            path_train = os.path.join(
                data_dir,
                f"Hyperelasticity_ellip_GF_u_s{nb_vert}_n{n_tr}_train.npz",
            )
            path_test = os.path.join(
                data_dir,
                f"Hyperelasticity_ellip_GF_u_s{nb_vert}_n{n_te}_test.npz",
            )
            np.savez_compressed(
                path_train,
                inputs=model_inputs[:n_tr],
                targets=model_targets[:n_tr],
            )
            np.savez_compressed(
                path_test,
                inputs=model_inputs[n_tr : n_tr + n_te],
                targets=model_targets[n_tr : n_tr + n_te],
            )
            print(f"saved: {path_train}")
            print(f"saved: {path_test}")

def solve_only(
    Phi,
    G,
    F,
    init_guess=None,
    n_load_steps_f=4,
    deg_v=1,
    nb_vert=64,
    linear_solver="gmres"
):
    """Run the Phi-FEM solve only; do not save data.
    Same as ``generate_data_Hole.solve_only``; grid body force ``F`` is required.
    Returns (sum of body-force loading-loop times in seconds over samples inside ``solve_one``, solution tuple (U, Y, P)).

    When ``n_load_steps_f > 1``, each curve in ``last_newton_residual_histories`` is recomputed at SNES
    monitor points under **full load** (``load_factor_f=1``) as ``||F(u)||``, to avoid residual-scale jumps
    from scaled loads at step changes; for a single load step this matches the residual norm reported by SNES.
    """
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        Phi_data=Phi,
        G_data=G,
        F_data=F,
        init_guess_data=init_guess,
        linear_solver=linear_solver,
        deg_v=deg_v,
        n_load_steps_f=n_load_steps_f
    )
    U, Y, P, load_loop_times, total_cg_iters, total_newton_iters = (
        solver.solve_several()
    )
    duration = float(np.sum(load_loop_times))
    solve_only.last_total_cg_iters = int(total_cg_iters)
    solve_only.last_total_newton_iters = int(total_newton_iters)
    solve_only.last_newton_residual_histories = list(
        getattr(solver, "newton_residual_histories", [])
    )
    if U is None:
        return float("nan"), (None, None, None)
    U = U.transpose((0, 3, 2, 1))
    Y = Y.transpose((0, 3, 2, 1))
    P = P.transpose((0, 3, 2, 1))
    return duration, (U, Y, P)

solve_only.last_total_cg_iters = 0
solve_only.last_total_newton_iters = 0
solve_only.last_newton_residual_histories = []

if __name__ == "__main__":
    nb_vert = 64
    # 500 = 400 train + 100 test
    Phi, G, F = create_parameters(nb_training_shapes=600, nb_vert=nb_vert)
    go(Phi, G, F, nb_vert=nb_vert, n_train=500, n_test=100)
