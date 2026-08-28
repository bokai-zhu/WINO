"""
    Plate with hole, Q4 (bilinear) displacement elements, 3x3 Gauss-point convention matching fno_utils;
    GRF Neumann BC, incremental iteration, quadrilateral elements, for NOWS2.0.
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
from scipy.stats import qmc
import typing

import dolfinx.fem.function
import dolfinx.fem.function
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import ufl
from utils import *
import dolfinx as dfx
from dolfinx.mesh import CellType
import multiphenicsx as mphx
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

degV = 2
degPhi = degV + 1

def call_F(xy, amplitude):
    return np.array([0 * xy[0] + amplitude, amplitude + 0 * xy[1]])

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise about origin by angle (radians). Matches prepare_data.py.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def call_phi_i(xy, x_0, y_0, lx):
    # return -(-(lx**2) + (xy[0] - x_0) ** 2 + (xy[1] - y_0) ** 2)
    return -(-1 + (xy[0] - x_0) ** 2 / lx ** 2 + (xy[1] - y_0) ** 2 / lx ** 2)

def call_phi(xy, param_holes):
    nb_holes = param_holes.shape[0]
    phi = 1.0
    considered_holes = list(range(nb_holes))
    for hole in considered_holes:
        x_i, y_i, li = param_holes[hole]
        phi *= call_phi_i(xy, x_i, y_i, li)
    return (-1.0) ** (len(considered_holes) + 1) * phi

def call_phi_ellipse_i(xy, cx, cy, a, b, theta):
    """
    Rotated-ellipse level set: interior > 0, exterior < 0 (matches locate_entities treating phi>0 as the hole).
    theta is the counterclockwise angle (radians) from the x-axis to the first ellipse axis (semi-axis a).
    """
    dx = xy[0] - cx
    dy = xy[1] - cy
    c = np.cos(theta)
    s = np.sin(theta)
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    return 1.0 - (xp / a) ** 2 - (yp / b) ** 2

def call_phi_ellipse(xy, param_holes):
    """param_holes: (n_holes, 5), each row (cx, cy, a, b, theta)."""
    nb_holes = param_holes.shape[0]
    phi = np.ones_like(xy[0], dtype=float)
    for k in range(nb_holes):
        cx, cy, a, b, th = param_holes[k]
        phi *= call_phi_ellipse_i(xy, cx, cy, a, b, th)
    return (-1.0) ** (nb_holes + 1) * phi

def ellipse_axis_aligned_half_extents(a, b, theta):
    """Axis-aligned half-width/half-height of a rotated ellipse in x and y (analytic bounds)."""
    c = np.cos(theta)
    s = np.sin(theta)
    wx = np.sqrt((a * c) ** 2 + (b * s) ** 2)
    wy = np.sqrt((a * s) ** 2 + (b * c) ** 2)
    return wx, wy

def ellipse_inside_unit_square(cx, cy, a, b, theta, margin=0.02):
    """
    Check that the rotated ellipse (including boundary) lies fully in [0,1]^2 with margin (out-of-bounds idea from prepare_data).
    """
    wx, wy = ellipse_axis_aligned_half_extents(a, b, theta)
    lo_x, hi_x = cx - wx, cx + wx
    lo_y, hi_y = cy - wy, cy + wy
    return (
        lo_x >= margin
        and hi_x <= 1.0 - margin
        and lo_y >= margin
        and hi_y <= 1.0 - margin
    )

def _adjust_ellipse_to_fit_square(cx, cy, a, b, theta, margin=0.02, shrink=0.92):
    """If out of bounds, shrink a, b proportionally; if still not, translate the center (out-of-bounds idea from prepare_data)."""
    a, b = float(a), float(b)
    cx, cy = float(cx), float(cy)
    amin, bmin = 1e-3, 1e-3
    for _ in range(80):
        if ellipse_inside_unit_square(cx, cy, a, b, theta, margin=margin):
            return cx, cy, a, b, theta
        a = max(a * shrink, amin)
        b = max(b * shrink, bmin)
    wx, wy = ellipse_axis_aligned_half_extents(a, b, theta)
    cx = float(np.clip(cx, margin + wx, 1.0 - margin - wx))
    cy = float(np.clip(cy, margin + wy, 1.0 - margin - wy))
    return cx, cy, a, b, theta

def generate_random_boundary(x_grid, mean_val, variance, length_scale=0.2):
    N_control = 32
    jitter = 1e-8
    x_control = np.linspace(0, 1, N_control)
    diffs = x_control[:, None] - x_control[None, :]
    K = np.exp(-0.5 * (diffs**2) / (length_scale**2))
    L = np.linalg.cholesky(K + jitter * np.eye(N_control))
    z = np.random.normal(size=N_control)
    y_control = np.dot(L, z)
    y_final = np.interp(x_grid, x_control, y_control)
    return mean_val + np.sqrt(variance) * y_final

def create_FG_numpy(nb_data, nb_vert):
    xy = np.linspace(0.0, 1.0, nb_vert)
    XX, YY = np.meshgrid(xy, xy)
    XX = XX.flatten()
    YY = YY.flatten()
    XXYY = np.stack([XX, YY])

    # Ellipse version: gamma, cx, cy, semi-minor b, aspect ratio r=a/b, rotation theta(rad), body-force amplitude
    sampler = qmc.LatinHypercube(d=7)
    sample = sampler.random(nb_data)
    eps = 1e-5
    low_bounds = [10 - eps, 0.4 - eps, 0.4 - eps, 0.1, 1.0, -np.pi, -1]
    up_bounds = [50, 0.6, 0.6, 0.15, 1.5, np.pi, 1]
    params = qmc.scale(sample, low_bounds, up_bounds)

    gamma_samples = params[:, 0:1]
    amplitude_f = params[:, 6:7]
    margin_hole = 0.02

    Phi = []
    for n in range(nb_data):
        cx = float(params[n, 1])
        cy = float(params[n, 2])
        b = float(params[n, 3])
        r = float(np.clip(params[n, 4], 1.0, 1.5))
        a = b * r
        theta = float(params[n, 5])
        cx, cy, a, b, theta = _adjust_ellipse_to_fit_square(
            cx, cy, a, b, theta, margin=margin_hole
        )
        params[n, 1] = cx
        params[n, 2] = cy
        params[n, 3] = b
        params[n, 4] = r
        params[n, 5] = theta
        params_holes = np.array([[cx, cy, a, b, theta]])
        # meshgrid(xy,xy) default indexing='xy': row i -> y=xy[i], column j -> x=xy[j]
        # After reshape, phi[i,j] = phi(y_i, x_j), matching [row=y, col=x] in _eval_phi_from_matrix
        phi = call_phi_ellipse(XXYY, params_holes).reshape((nb_vert, nb_vert))
        Phi.append(phi)
    Phi = np.array(Phi)

    F = call_F(XXYY, amplitude_f)
    F = np.reshape(F, [2, nb_data, nb_vert, nb_vert]).transpose(1, 0, 3, 2)

    G_list = []
    x_unique = np.linspace(0.0, 1.0, nb_vert)
    for n in range(nb_data):
        target_mean = gamma_samples[n, 0]
        boundary_profile = generate_random_boundary(
            x_unique, mean_val=target_mean, variance=0.1
        )
        g_field_y = np.tile(boundary_profile, (nb_vert, 1))
        g_sample = np.stack([np.zeros_like(g_field_y), g_field_y])
        G_list.append(g_sample)
    G = np.array(G_list)

    return F, Phi, G, params

def _neo_hookean_first_pk_from_F(F):
    """
    Same Neo-Hookean energy as tensors(u); F is a UFL variable (2x2); returns the first Piola-Kirchhoff stress P.
    """
    C = F.T * F
    Ic = ufl.tr(C)
    J = ufl.det(F)
    j_eps = 1e-8
    j_beta = 30.0
    J_safe = j_eps + ufl.ln(1.0 + ufl.exp(j_beta * J)) / j_beta
    E = 100
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J_safe) + (lmbda / 2) * (ufl.ln(J_safe)) ** 2
    return ufl.diff(psi, F)

def tensors(u):
    d = len(u)
    I = ufl.Identity(d)
    F = ufl.variable(I + ufl.grad(u))
    return _neo_hookean_first_pk_from_F(F)

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
    """field_sample (Ny, Nx, 2), interpolate onto the CG target space (same as PhiFemSolver._assign_function_from_grid_linear_to_target)."""
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

def _build_submesh_V_dx(phi_sample: np.ndarray, nb_cell: int, deg_v: int):
    """Same macro partition as solve_one + hole-cut submesh where phi>0; return dx and bottom-edge Dirichlet dofs."""
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
    deg_v: int = 2,
) -> dict:
    """
    Compute on the dolfinx submesh (relative to ``u_reference``):

    1. **H¹ seminorm error**: \\sqrt{\\int_\\Omega \\|\\nabla(u_{pred}-u_{ref})\\|^2 \\,dx}

    2. **Discrete energy norm** (consistent with \\(\\sqrt{e^\\top K e}\\)):
       At ``u_reference``, take the variational derivative of the **volume** weak form ``\\int P(u):\\nabla v\\,dx`` to get the tangent stiffness
       ``K`` (homogeneous Dirichlet on the bottom edge); ``e`` is the dof vector of ``u_predicted - u_reference``;
       ``energy_norm = \\sqrt{e^\\top K e}``.

    Note: ``K`` contains only the Neo-Hookean volume term, not the full Phi-FEM block Jacobian (y/p, hole-edge stabilization, etc.).

    Parameters
    ----
    phi_sample : (Ny, Nx)
    u_reference, u_predicted : (Ny, Nx, 2)
    nb_cell : number of macro cells (same as ``PhiFemSolver(nb_cell=nb_vert-1)``)
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
    # return {
    #     "h1_seminorm_error": h1_err,
    #     "h1_seminorm_reference": h1_ref,
    #     "h1_seminorm_error_relative": (h1_err / h1_ref) if h1_ref > 1e-30 else float("nan"),
    #     "energy_norm_sqrt_eTKe": energy_norm,
    #     "energy_norm_reference_sqrt_uTKu": ref_energy_norm,
    #     "energy_norm_relative": (energy_norm / ref_energy_norm)
    #     if ref_energy_norm > 1e-30
    #     else float("nan"),
    # }

class Phi_i:
    def __init__(self, x_0, y_0, lx) -> None:
        self.x_0 = x_0
        self.y_0 = y_0
        self.lx = lx

    def eval(self, x):
        return -(-(self.lx**2) + (x[0] - self.x_0) ** 2 + (x[1] - self.y_0) ** 2)

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) > 0.0

class GExpr:
    def __init__(self, boundary_values_1d, load_factor):
        self.values = boundary_values_1d
        self.factor = load_factor
        self.x_grid = np.linspace(0, 1, len(boundary_values_1d))

    def eval(self, x):
        # x shape: (3, N_points)
        # Interpolate in x to get the corresponding boundary values
        # values is already (mean + fluctuation) at this point
        # Only multiply by load_factor (0 -> 1) to apply loading
        val_interp = np.interp(x[0], self.x_grid, self.values)
        return np.array([0.0 * x[0], val_interp * self.factor])

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
        init_guess_data=None,
        linear_solver="gmres",
        deg_v=None,
        cg_fallback_to_lu=False,
        n_load_steps_f=3,
    ):
        self.N = N = nb_cell
        # Unify input format: (batch, Ny, Nx, t)
        self.Phi_data = Phi_data
        self.G_data = G_data
        # Optional initial guess: (batch, Ny, Nx, 2) displacement u only; (batch, Ny, Nx, 8) is [u(2), y(4), p(2)]
        self.init_guess_data = init_guess_data
        self._init_guess_channels = None
        self.linear_solver = linear_solver
        self.cg_fallback_to_lu = False
        self.deg_v = int(degV if deg_v is None else deg_v)
        self.deg_phi = self.deg_v
        self.n_load_steps_f = int(n_load_steps_f)
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

        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [1, 1]]),
            np.array([N, N]),
            cell_type=CellType.quadrilateral,
        )
        self.V_macro = dolfinx.fem.functionspace(
            self.mesh_macro, ("CG", 1, (self.mesh_macro.geometry.dim,))
        )
        # Tensor macro space (for y): 4 components (2x2)
        self.V_macro_tensor = dolfinx.fem.functionspace(
            self.mesh_macro, ("CG", 1, (self.mesh_macro.geometry.dim, self.mesh_macro.geometry.dim))
        )
        # Scalar space for the Phi field
        self.V_macro_scalar = dolfinx.fem.functionspace(
            self.mesh_macro, ("CG", 1)
        )

        coords = self.V_macro.tabulate_dof_coordinates()
        coords = coords.reshape((-1, 3))
        coords = coords[:, :2]

        coords = np.round(coords, decimals=5)
        self.sorted_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.padding = 1e-14

    def _eval_phi_from_matrix(self, phi_sample, x):
        """Evaluate phi(x,y) by bilinear interpolation; x has shape (3, N)."""
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

    def _eval_channels_from_matrix(self, field_sample, x):
        """Evaluate a multi-channel field by bilinear interpolation; return shape=(C, N)."""
        ny, nx, cdim = field_sample.shape
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

        out = []
        for c in range(cdim):
            q00 = field_sample[y0, x0, c]
            q10 = field_sample[y0, x1, c]
            q01 = field_sample[y1, x0, c]
            q11 = field_sample[y1, x1, c]
            out_c = (
                (1.0 - tx) * (1.0 - ty) * q00
                + tx * (1.0 - ty) * q10
                + (1.0 - tx) * ty * q01
                + tx * ty * q11
            )
            out.append(out_c)
        return np.vstack(out)

    def _assign_function_from_grid_nearest(self, func, field_sample):
        """
        Assign a regular-grid field directly to function dofs (nearest node, no interpolation).
        field_sample: (Ny, Nx, C); C must equal the function block size.
        """
        ny, nx, cdim = field_sample.shape
        bs = func.function_space.dofmap.index_map_bs
        if cdim != bs:
            raise ValueError(f"channel count mismatch: field C={cdim}, function bs={bs}")

        coords = func.function_space.tabulate_dof_coordinates().reshape((-1, 3))[:, :2]
        # Regular-grid node indices (x,y ∈ [0,1])
        ix = np.rint(np.clip(coords[:, 0], 0.0, 1.0) * (nx - 1)).astype(np.int32)
        iy = np.rint(np.clip(coords[:, 1], 0.0, 1.0) * (ny - 1)).astype(np.int32)

        vals = field_sample[iy, ix, :]  # (ndof, bs)
        arr = func.x.array.reshape((-1, bs))
        arr[:] = vals
        func.x.scatter_forward()

    def _assign_function_from_grid_linear_to_target(self, func, field_sample):
        """
        Assign the regular-grid initial guess onto a matching CG1 space, then linearly interpolate to the target space.
        When degV=2, lift initial_guess into the quadratic space via first-order linear interpolation.
        """
        mesh = func.function_space.mesh
        bs = func.function_space.dofmap.index_map_bs
        cdim = int(field_sample.shape[-1])
        if cdim != bs:
            raise ValueError(f"channel count mismatch: field C={cdim}, function bs={bs}")

        vshape = func.function_space.value_shape
        if len(vshape) == 0:
            V_lin = dfx.fem.functionspace(mesh, ("CG", 1))
        elif len(vshape) == 1:
            V_lin = dfx.fem.functionspace(mesh, ("CG", 1, (vshape[0],)))
        else:
            V_lin = dfx.fem.functionspace(mesh, ("CG", 1, tuple(vshape)))

        f_lin = dfx.fem.Function(V_lin)
        self._assign_function_from_grid_nearest(f_lin, field_sample)
        func.interpolate(f_lin)
        func.x.scatter_forward()

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
            np array : matrix of the phiFEM solution
        """
        newton_residual_history = []
        nb_holes = int(1)
        phi_sample = self.Phi_data[i, :, :, 0]
        # G_data shape: (nb_data, Ny, Nx, 2)
        g_sample = self.G_data[i]
        boundary_1d = g_sample[0, :, 1]

        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1
        vertices_dim = 0

        phi_eval = lambda x: self._eval_phi_from_matrix(phi_sample, x.reshape((3, -1)))
        all_entities = np.arange(
            self.mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32
        )
        cells_outside = dfx.mesh.locate_entities(
            self.mesh_macro, cell_dim, lambda x: phi_eval(x) > 0.0
        )
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)

        mesh = dfx.mesh.create_submesh(
            self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro
        )[0]

        V = dfx.fem.functionspace(mesh, ("CG", self.deg_v, (cell_dim,)))
        V_phi = dfx.fem.functionspace(mesh, ("CG", self.deg_phi))
        Z_N = dfx.fem.functionspace(mesh, ("CG", self.deg_v, (cell_dim, cell_dim)))
        if self.deg_v == 1:
            Q_N = dfx.fem.functionspace(mesh, ("DG", self.deg_v - 1, (cell_dim,)))
        else:
            Q_N = dfx.fem.functionspace(mesh, ("CG", self.deg_v - 1, (cell_dim,)))

        dofs_V = np.arange(
            0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        )
        spaces = [V]
        restricts = [dofs_V]
        restriction = [multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)]
        neumann_cells, neumann_facets = [], []
        hole_restriction_cells = []

        mesh.topology.create_connectivity(cell_dim, facet_dim)
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, vertices_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, vertices_dim)
        interior_entities = np.arange(
            mesh.topology.index_map(cell_dim).size_global, dtype=np.int32
        )

        nv_cell = int(np.asarray(c_to_v.links(0)).size)
        nf_cell = int(np.asarray(c_to_f.links(0)).size)
        c_to_v_map = np.reshape(c_to_v.array, (-1, nv_cell))
        assert c_to_v_map.shape[0] == len(interior_entities)
        points = mesh.geometry.x.T
        phi_values = phi_eval(points)
        phi_cells = phi_values[c_to_v_map]
        # A cell intersects the zero level set if φ changes sign on any edge (v_i, v_{i+1}) (3 edges for triangles, 4 for quads)
        cells_boundary_all = np.zeros(phi_cells.shape[0], dtype=bool)
        for ei in range(nv_cell):
            ej = (ei + 1) % nv_cell
            prod = phi_cells[:, ei] * phi_cells[:, ej]
            cells_boundary_all |= (prod <= 0.0) | near(prod, 0.0)
        cells_boundary = np.where(cells_boundary_all == True)[0]
        hole_restriction_cells.append(cells_boundary)

        neumann_facets_measure, neumann_facets_stab_measure, neumann_cells_measure = (
            [],
            [],
            [],
        )
        neumann_values, neumann_stab_values = [], []
        c2f_map = np.reshape(c_to_f.array, (-1, nf_cell))

        for j in range(nb_holes):
            hole_restriction_cells_j = hole_restriction_cells[j]
            if len(hole_restriction_cells_j) > 0:
                neumann_cells_j = np.unique(hole_restriction_cells_j)
                omega_1_small_cells_j = np.setdiff1d(interior_entities, neumann_cells_j)
                omega_1_small_facets_j = c2f_map[omega_1_small_cells_j].flatten()
                neumann_facets_j = np.unique(c2f_map[neumann_cells_j].flatten())
                neumann_stab_facets_j = np.unique(
                    np.intersect1d(omega_1_small_facets_j, neumann_facets_j)
                )

                neumann_facets_measure.append(neumann_facets_j)
                neumann_facets_stab_measure.append(neumann_stab_facets_j)
                neumann_cells_measure.append(neumann_cells_j)
                neumann_values.append(4 + j)
                neumann_stab_values.append(30 + j)

        neumann_cells = np.unique(np.concatenate(hole_restriction_cells))
        mesh_neumann = dfx.mesh.create_submesh(mesh, cell_dim, neumann_cells)[0]
        restr_Neumann_Z_N = dfx.fem.locate_dofs_topological(
            Z_N, cell_dim, list(neumann_cells)
        )
        restr_Neumann_Q_N = dfx.fem.locate_dofs_topological(
            Q_N, cell_dim, list(neumann_cells)
        )
        restricts.append(restr_Neumann_Z_N)
        restricts.append(restr_Neumann_Q_N)

        restriction.append(
            multiphenicsx.fem.DofMapRestriction(Z_N.dofmap, restr_Neumann_Z_N)
        )
        restriction.append(
            multiphenicsx.fem.DofMapRestriction(Q_N.dofmap, restr_Neumann_Q_N)
        )
        spaces.append(Z_N)
        spaces.append(Q_N)

        start = time.time()
        # create meshtags for cells
        full_neumann_values = []
        for j in range(len(neumann_cells_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_cells_measure[j]), dtype=np.intc
            )
            full_neumann_values.append(values_Neumann)

        values_cells = np.hstack(full_neumann_values)
        entities_cells = np.hstack(neumann_cells_measure)
        sorted_cells = np.argsort(entities_cells)

        subdomains_cell = dfx.mesh.meshtags(
            mesh,
            cell_dim,
            entities_cells[sorted_cells],
            values_cells[sorted_cells],
        )

        full_neumann_values_facets = []
        for j in range(len(neumann_facets_measure)):
            values_Neumann = neumann_values[j] * np.ones(
                len(neumann_facets_measure[j]), dtype=np.intc
            )
            full_neumann_values_facets.append(values_Neumann)

            values_Neumann_stab = neumann_stab_values[j] * np.ones(
                len(neumann_facets_stab_measure[j]), dtype=np.intc
            )
            full_neumann_values.append(values_Neumann_stab)
        # Find the top boundary (y=1.0) and tag it as 1 for ds(1)
        upper_facets_loc = dfx.mesh.locate_entities_boundary(
            mesh, facet_dim, lambda x: np.isclose(x[1], 1.0)
        )
        upper_values_loc = np.ones(len(upper_facets_loc), dtype=np.intc) * 1  # Tag 1 for top Neumann

        # Merge hole facets with top-boundary facets
        # Append top-boundary facets to the list
        values_facets = np.hstack(full_neumann_values_facets + [upper_values_loc])
        entities_facets = np.hstack(neumann_facets_measure + [upper_facets_loc])
        sorted_facets = np.argsort(entities_facets)

        subdomains_facet = dfx.mesh.meshtags(
            mesh,
            facet_dim,
            entities_facets[sorted_facets],
            values_facets[sorted_facets],
        )

        end = time.time()

        V_phi = dfx.fem.functionspace(mesh, ("CG", self.deg_phi))

        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        # Align with the Q4 3x3 Gauss-point convention in fno_utils:
        # Use the default rule compatible with all versions + high enough degree (3 points/direction → degree-5 accuracy per direction).
        quad_meta_cell = {"quadrature_degree": 5}
        quad_meta_facet = {"quadrature_degree": 5}
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=subdomains_cell,
            metadata=quad_meta_cell,
        )
        ds = ufl.Measure(
            "ds",
            domain=mesh,
            subdomain_data=subdomains_facet,
            metadata=quad_meta_facet,
        )
        dS = ufl.Measure(
            "dS",
            domain=mesh,
            subdomain_data=subdomains_facet,
            metadata=quad_meta_facet,
        )
        nb_incr = self.n_load_steps_f
        gamma_div, gamma_u, gamma_p = 0.01, 0.001, 0.01
        sigma_N = 0.01
        # gamma_div, gamma_u, gamma_p = 1.0, 0.01, 0.01
        # sigma_N = 0.01

        uyp_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
        if self.init_guess_data is not None:
            init_sample = self.init_guess_data[i]
            # When degV=2, prefer a displacement-only initial guess:
            # Assign u by CG1->target linear interpolation and keep y/p at 0, so mixed-variable inconsistency does not stall Newton.
            u_init_sample = init_sample[:, :, 0:2]
            self._assign_function_from_grid_linear_to_target(uyp_split[0], u_init_sample)
        # Force bottom-edge Dirichlet(0) only when there is no init_guess.
        # With a network initial guess, hard-zeroing introduces large gradients near the bottom edge and makes J/residual diagnostics inconsistent with the raw guess.
        lower_facets0 = dolfinx.mesh.locate_entities_boundary(
            mesh, 1, lambda x: np.isclose(x[1], 0.0)
        )
        boundary_dofs_low0 = fem.locate_dofs_topological(V, 1, lower_facets0)
        if self.init_guess_data is None:
            uyp_split[0].x.array[boundary_dofs_low0] = 0.0
        u0_init = uyp_split[0]
        dim_u = u0_init.function_space.value_shape[0]

        vzq_split = [ufl.TestFunction(VVV) for VVV in spaces]
        duyp_split = [ufl.TrialFunction(VVV) for VVV in spaces]
        Phis_list = []

        phi_func = dolfinx.fem.Function(V_phi)
        phi_func.interpolate(lambda x: phi_eval(x.reshape((3, -1))))
        Phis_list.append(phi_func)
        
        # Initialize loading parameters
        current_load = 0.0
        target_load = 1.0  # target is 100% of gamma_G
        d_load = 1.0 / nb_incr  # initial step size
        min_d_load = 1e-2  # minimum step size; abort if smaller
        
        # Record whether the solve fully succeeded
        solved_successfully = False
        total_cg_iters = 0
        total_newton_iters = 0

        # Loop until the target load (timing covers only this load-stepping loop)
        t_load_loop_start = time.time()
        while current_load < target_load:
            # Try one step
            next_load_factor = current_load + d_load
            
            # Prevent overshoot on the last step
            if next_load_factor > target_load:
                next_load_factor = target_load
            
            print(f"Trying load factor: {next_load_factor:.4f} (Step size: {d_load:.4f})")
                
            # --- GExpr interpolated from an array ---
            g_expr = GExpr(boundary_1d, next_load_factor)
            g = dolfinx.fem.Function(V)
            g.interpolate(g_expr.eval)
            u1 = uyp_split[0]
            v1 = vzq_split[0]
            y, p_N = uyp_split[1], uyp_split[1 + 1]
            z, q_N = vzq_split[1], vzq_split[1 + 1]
            Pu1 = tensors(u1)
            Pv1 = ufl.derivative(Pu1, u1, v1)

            F = [0.0 for k in range(len(spaces))]

            dx_full_omega_1 = dx

            au1v1 = ufl.inner(Pu1, ufl.grad(v1)) * dx_full_omega_1
            F[0] += au1v1 - ufl.inner(g, v1) * ds(1)

            index_ = 1
            for ind in range(nb_holes):
                phi = Phis_list[ind]

                dx_Neumann_omega_1 = dx(neumann_values[ind])
                ds_Neumann_omega_1 = ds(neumann_values[ind])
                dS_Neumann = dS(neumann_values[ind])
                dS_Neumann_stab = dS(neumann_stab_values[ind])

                Gh1 = (
                    sigma_N
                    * ufl.avg(h)
                    * ufl.inner(ufl.jump(Pu1, n), ufl.jump(Pv1, n))
                    * dS_Neumann_stab
                )

                F[0] += gamma_u * ufl.inner(Pu1, Pv1) * dx_Neumann_omega_1 + Gh1

                dsi = ds_Neumann_omega_1
                dxi = dx_Neumann_omega_1
                F[0] += (
                    ufl.inner(ufl.dot(y, n), v1) * dsi
                    + gamma_u * ufl.inner(y, Pv1) * dxi
                )
                P_u = Pu1

                F[index_] += gamma_u * ufl.inner(P_u, z) * dxi

                F[index_] += (
                    gamma_u * ufl.inner(y, z) * dxi
                    + gamma_div * ufl.inner(ufl.div(y), ufl.div(z)) * dxi
                    + gamma_p
                    * h ** (-2)
                    * ufl.inner(ufl.dot(y, ufl.grad(phi)), ufl.dot(z, ufl.grad(phi)))
                    * dxi
                )
                F[index_] += (
                    gamma_p
                    * h ** (-3)
                    * ufl.inner(p_N * phi, ufl.dot(z, ufl.grad(phi)))
                    * dxi
                )

                F[index_ + 1] += (
                    gamma_p
                    * h ** (-3)
                    * ufl.inner(ufl.dot(y, ufl.grad(phi)), q_N * phi)
                    * dxi
                )

                F[index_ + 1] += (
                    gamma_p * h ** (-4) * ufl.inner(p_N * phi, q_N * phi) * dxi
                )

            J = [
                [
                    ufl.derivative(F[i], uyp_split[j], duyp_split[j])
                    for j in range(len(uyp_split))
                ]
                for i in range(len(F))
            ]
            zeros_1d = np.zeros_like(boundary_1d) 
            
            # 2. Use the new GExpr class with a zero array; load_factor can be anything (e.g. 1.0) since the result is zero
            g_expr_null = GExpr(zeros_1d, load_factor=1.0)
            
            g_null = dolfinx.fem.Function(V)
            g_null.interpolate(g_expr_null.eval)
            lower_facets = dolfinx.mesh.locate_entities_boundary(
                mesh, 1, lambda x: np.isclose(x[1], 0.0)
            )
            boundary_dofs_low = fem.locate_dofs_topological(V, 1, lower_facets)
            bc_low = fem.dirichletbc(g_null, boundary_dofs_low)
            problem = NonLinearPhiFEM(
                F, J, tuple(uyp_split), [bc_low], restriction, spaces
            )
            F_vec = mphx.fem.petsc.create_vector_block(
                problem._F, restriction=restriction
            )
            J_mat = mphx.fem.petsc.create_matrix_block(
                problem._J, restriction=(restriction, restriction)
            )
            snes = petsc4py.PETSc.SNES().create(mesh.comm)
            snes.setTolerances(max_it=50)
            snes.setType("newtonls") # use Newton with line search
            ksp = snes.getKSP()
            if self.linear_solver == "lu":
                # LU direct solver (MUMPS)
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.getPC().setFactorSolverType("mumps")
            elif self.linear_solver == "cg":
                # CG iterative solver (aligned with generate_data_Hole_ellip.py)
                ksp.setType("cg")
                ksp.getPC().setType("jacobi")
                ksp.setTolerances(rtol=cg_rtol)
            else:
                # GMRES iterative solver
                ksp.setType("gmres")
                try:
                    ksp.setGMRESRestart(int(gmres_restart))
                except Exception:
                    pass
                pc = ksp.getPC()
                try:
                    pc.setType("gamg")
                except Exception:
                    pc.setType("jacobi")
                ksp.setTolerances(
                    rtol=float(gmres_rtol),
                    atol=1e-12,
                    max_it=int(gmres_max_it),
                )
            opts = PETSc.Options()
            opts["snes_linesearch_type"] = "bt"
            snes.setObjective(problem.obj)
            snes.setFunction(problem.F, F_vec)
            snes.setJacobian(problem.J, J=J_mat, P=None)
            step_residuals = []

            def _snes_monitor(_snes, its, fnorm, *args):
                step_residuals.append(float(fnorm))

            snes.setMonitor(_snes_monitor)
            solution = problem.create_snes_solution()
            # Keep the last converged solution; restore it on failure, otherwise SNES corrupts the solution vector
            last_converged_solution = solution.copy()
            try:
                snes.solve(None, solution)
                converged_reason = snes.getConvergedReason()
            except Exception:
                converged_reason = -1
            newton_residual_history.extend(step_residuals)
            try:
                total_newton_iters += int(snes.getIterationNumber())
            except Exception:
                pass
            if self.linear_solver in ("cg", "gmres"):
                try:
                    total_cg_iters += int(snes.getLinearSolveIterations())
                except Exception:
                    pass
            if converged_reason > 0:
                print(f"Converged at {next_load_factor:.4f}")
                
                current_load = next_load_factor
                problem.update_solutions(solution)
                last_converged_solution = solution.copy()
                if d_load < (1.0 / nb_incr):
                    d_load = min(d_load * 1.5, 1.0 / nb_incr)
                if current_load >= target_load:
                    solved_successfully = True
                    break
                    
            else:
                d_load /= 2.0
                solution.copy(last_converged_solution)
                problem.update_solutions(solution)
                if d_load < min_d_load:
                    break
        
        load_loop_time = time.time() - t_load_loop_start

        self._last_newton_residual_history = newton_residual_history
        if not solved_successfully:
            return None, None, None, load_loop_time, total_cg_iters, total_newton_iters

        u_sol = uyp_split[0]
        u_mat = self.make_matrix(u_sol, V, self.V_macro)

        y_sol = uyp_split[1]
        y_mat = self.make_matrix(y_sol, Z_N, self.V_macro_tensor)

        p_sol = uyp_split[2]
        p_mat = self.make_matrix(p_sol, Q_N, self.V_macro)

        return u_mat, y_mat, p_mat, load_loop_time, total_cg_iters, total_newton_iters

    def solve_several(self):
        U_list = []
        Y_list = []
        P_list = []
        S_list = []
        load_loop_times = []
        total_cg_iters_all = 0
        total_newton_iters_all = 0
        self.newton_residual_histories = []
        nb = len(self.Phi_data)
        for i in range(nb):
            print(f"Data : {i}/{nb}")
            u_mat, y_mat, p_mat, t_load, cg_iters_i, newton_iters_i = self.solve_one(i)
            self.newton_residual_histories.append(
                list(getattr(self, "_last_newton_residual_history", []))
            )
            load_loop_times.append(t_load)
            total_cg_iters_all += int(cg_iters_i)
            total_newton_iters_all += int(newton_iters_i)
            if u_mat is None:
                continue
            U_list.append(u_mat)
            Y_list.append(y_mat)
            P_list.append(p_mat)

        if len(U_list) == 0:
            # Do not raise; let the caller decide how to handle failure (e.g. time/iteration stats or skip the sample)
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

def create_parameters(nb_vert=64, nb_training_shapes=300):
    nb_data = nb_training_shapes
    F, phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    # Unify output to (batch, Ny, Nx, t)
    phi_out = phi[:, :, :, None]
    g_out = G.transpose((0, 2, 3, 1))
    return phi_out, g_out

def go(Phi, G, init_guess=None, nb_vert=64, n_train=400, n_test=100):
    """
    Main function to generate data.
    init_guess: optional, (batch, Ny, Nx, 2) for u only, or (batch, Ny, Nx, 8) for u+y+p.
    Save only two npz files (train then test in batch order: first n_train, then n_test).
    """
    save = True
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(base_dir), "data")

    ti0 = time.time()
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        Phi_data=Phi,
        G_data=G,
        init_guess_data=init_guess,
        linear_solver="gmres",
        deg_v=2,
        n_load_steps_f=1
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

    # Unify to (batch, N_y, N_x, t)
    U = U.transpose((0, 3, 2, 1))
    Y = Y.transpose((0, 3, 2, 1))
    P = P.transpose((0, 3, 2, 1))
    # Phi = Phi.transpose((0, 3, 2, 1))
    # G = G.transpose((0, 3, 2, 1))

    if save:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        model_inputs = np.concatenate([Phi, G], axis=-1)
        model_targets = U
        batch_num = model_inputs.shape[0]
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
            data_dir, f"Hyperelasticity_Hole_G_u_s{nb_vert}_n{n_tr}_train.npz"
        )
        path_test = os.path.join(
            data_dir, f"Hyperelasticity_Hole_G_u_s{nb_vert}_n{n_te}_test.npz"
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

def solve_only(Phi, G, init_guess=None, deg_v=1, nb_vert=64, n_load_steps_f=1):
    """Run the Phi-FEM solve only; do not save data.
    init_guess: optional, (batch, Ny, Nx, 2) for u only, or (batch, Ny, Nx, 8) for u+y+p.
    Returns (accumulated seconds of the load-stepping while loop, solution tuple).
    Time is the sum over samples of the ``while current_load < target_load`` section inside solve_one.
    """
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        Phi_data=Phi,
        G_data=G,
        init_guess_data=init_guess,
        linear_solver="gmres",
        deg_v=deg_v,
        n_load_steps_f=n_load_steps_f
    )
    U, Y, P, load_loop_times, total_cg_iters, total_newton_iters = solver.solve_several()
    duration = float(np.sum(load_loop_times))
    # Keep the old field name (last_total_cg_iters) for the total iterative linear-solver steps (CG/GMRES)
    solve_only.last_total_cg_iters = int(total_cg_iters)
    solve_only.last_total_ksp_iters = int(total_cg_iters)
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
solve_only.last_total_ksp_iters = 0
solve_only.last_total_newton_iters = 0
solve_only.last_newton_residual_histories = []
cg_rtol = 1e-8
gmres_rtol = 1e-5
gmres_max_it = 20000
gmres_restart = 80

if __name__ == "__main__":
    nb_vert = 64
    Phi, G = create_parameters(nb_training_shapes=400)
    go(Phi, G, nb_vert=nb_vert, n_train=300, n_test=100)

