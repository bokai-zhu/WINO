"""
    Second order, GRF Neumann BC, incremental iteration, triangular elements, N*N, symmetric trapezoid, corners removed
    Note: the right edge of the trapezoid must lie on grid nodes, otherwise the error is large (a Phi-FEM limitation).
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
degPhi = degV

def call_F(xy, amplitude):
    return np.array([0 * xy[0] + amplitude, amplitude + 0 * xy[1]])

def generate_random_boundary(y_grid, mean_val, variance, length_scale=0.4):
    """Generate a 1D Gaussian random field along the y-axis"""
    npts = len(y_grid)
    n_control = 32
    jitter = 1e-8
    y_min, y_max = y_grid.min(), y_grid.max()
    y_control = np.linspace(y_min, y_max, n_control)
    diffs = y_control[:, None] - y_control[None, :]
    K = np.exp(-0.5 * (diffs ** 2) / (length_scale ** 2))
    L = np.linalg.cholesky(K + jitter * np.eye(n_control))
    z = np.random.normal(size=n_control)
    val_control = np.dot(L, z)
    val_final = np.interp(y_grid, y_control, val_control)
    return mean_val + np.sqrt(variance) * val_final

def create_FG_numpy(nb_data, nb_vert):
    Nx = Ny = nb_vert
    xx = np.linspace(0.0, 4.0, Nx)
    yy = np.linspace(0.0, 2.0, Ny)
    XX, YY = np.meshgrid(xx, yy)
    XX_flat = XX.flatten()
    YY_flat = YY.flatten()
    XXYY = np.stack([XX_flat, YY_flat])

    # 1) Sample continuous variables (gamma_G, amplitude_f)
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(nb_data)
    low_bounds = [5, -1.0]
    up_bounds = [15.000001, 1.0]
    params_cont = qmc.scale(sample, low_bounds, up_bounds)
    gamma_samples = params_cont[:, 0:1]
    amplitude_f = params_cont[:, 1:2]

    # 2) Sample y0/y1/y2/y3 continuously at random (no longer snapped to grid nodes)
    # Geometry domain y ∈ [0, 2]; range can be adjusted as needed
    y0_arr = np.random.uniform(0., 0.2, size=nb_data)
    # l_arr = np.random.uniform(1.7, 1.7, size=nb_data)
    y1_arr = np.random.uniform(1.8, 2, size=nb_data)
    # y0_arr = np.minimum(y0_arr, 2 - l_arr)
    # y1_arr = y0_arr + l_arr
    y2_arr = np.random.uniform(2.0, 2.0, size=nb_data)
    y3_arr = np.random.uniform(1.4, 1.4, size=nb_data)
    params = np.column_stack([gamma_samples, y0_arr, y1_arr, y2_arr, y3_arr, amplitude_f])

    # 3) Build Phi (symmetric trapezoid)
    Phi = []
    for n in range(nb_data):
        y0, y1, y2, y3 = y0_arr[n], y1_arr[n], y2_arr[n], y3_arr[n]
        B_x = y0 + (y3 - y0) / 4.0 * XX_flat
        T_x = y1 + (y2 - y1) / 4.0 * XX_flat
        phi_flat = (YY_flat - B_x) * (YY_flat - T_x)
        Phi.append(phi_flat.reshape((Ny, Nx)))
    Phi = np.array(Phi).transpose(0, 2, 1)

    # 4) Body force F
    F = call_F(XXYY, amplitude_f)
    F = np.reshape(F, [2, nb_data, Ny, Nx]).transpose(1, 0, 3, 2)

    # 5) Boundary G
    G_list = []
    for n in range(nb_data):
        target_mean = gamma_samples[n, 0]
        boundary_profile = generate_random_boundary(yy, mean_val=target_mean, variance=0.1)
        g_field_y = np.tile(boundary_profile[:, None], (1, Nx))
        g_sample = np.stack([np.zeros_like(g_field_y), g_field_y])
        G_list.append(g_sample)
    G = np.array(G_list).transpose(0, 1, 3, 2)

    return F, Phi, G, params

def tensors(u):
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))
    E = 500
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    P = ufl.diff(psi, F)
    return P

def _eval_phi_from_matrix_standalone(phi_sample: np.ndarray, x: np.ndarray) -> np.ndarray:
    """phi_sample (ny, nx), x shape (3, N), physical domain x∈[0,4], y∈[0,2]."""
    ny, nx = phi_sample.shape
    xq = np.clip(x[0], 0.0, 4.0)
    yq = np.clip(x[1], 0.0, 2.0)
    fx = (xq / 4.0) * (nx - 1)
    fy = (yq / 2.0) * (ny - 1)
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
    """field_sample (Ny, Nx, 2), interpolate onto the CG target space (trapezoid domain x∈[0,4], y∈[0,2])."""
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
    ix = np.rint(np.clip(coords[:, 0], 0.0, 4.0) / 4.0 * (nx - 1)).astype(np.int32)
    iy = np.rint(np.clip(coords[:, 1], 0.0, 2.0) / 2.0 * (ny - 1)).astype(np.int32)
    vals = field_sample[iy, ix, :]
    arr = f_lin.x.array.reshape((-1, bs))
    arr[:] = vals
    f_lin.x.scatter_forward()
    func.interpolate(f_lin)
    func.x.scatter_forward()

def _build_submesh_V_dx(phi_sample: np.ndarray, nb_cell: int, deg_v: int):
    """Same as the Trapezoid solve: macro mesh + submesh clipped by phi<=0."""
    mesh_macro = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0.0, 0.0], [4.0, 2.0]]),
        np.array([nb_cell, nb_cell]),
        cell_type=CellType.triangle,
    )
    cell_dim = mesh_macro.geometry.dim
    phi_eval = lambda x: _eval_phi_from_matrix_standalone(
        phi_sample, x.reshape((3, -1))
    )
    all_entities = np.arange(
        mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32
    )
    cells_outside = dfx.mesh.locate_entities(
        mesh_macro, cell_dim, lambda x: phi_eval(x) >= -3e-16
    )
    interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
    mesh = dfx.mesh.create_submesh(
        mesh_macro, mesh_macro.topology.dim, interior_entities_macro
    )[0]
    facet_dim = mesh.geometry.dim - 1
    dx_m = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
    V = dfx.fem.functionspace(mesh, ("CG", deg_v, (cell_dim,)))
    left_facets = dfx.mesh.locate_entities_boundary(
        mesh, facet_dim, lambda x: np.isclose(x[0], 0.0)
    )
    boundary_dofs_left = fem.locate_dofs_topological(V, facet_dim, left_facets)
    return mesh, V, dx_m, boundary_dofs_left

def compute_errors(
    phi_sample: np.ndarray,
    u_reference: np.ndarray,
    u_predicted: np.ndarray,
    nb_cell: int,
    deg_v: int = 2,
):
    """
    Compute on the Trapezoid submesh (relative to ``u_reference``):
    1) H1 seminorm error; 2) tangent-stiffness energy-norm error sqrt(e^T K e).
    Returns (relative H1 error, relative energy-norm error).
    """
    if phi_sample.ndim != 2:
        raise ValueError(f"phi_sample must be 2D, got {phi_sample.shape}")
    if u_reference.shape != u_predicted.shape or u_reference.shape[-1] != 2:
        raise ValueError(
            f"u_reference/u_predicted must be (Ny,Nx,2), got {u_reference.shape}"
        )

    mesh, V, dx_m, boundary_dofs_left = _build_submesh_V_dx(phi_sample, nb_cell, deg_v)
    comm = mesh.comm

    u_ref = fem.Function(V)
    u_pred = fem.Function(V)
    _assign_vector_field_from_grid_linear_to_cg(u_ref, u_reference.astype(np.float64))
    _assign_vector_field_from_grid_linear_to_cg(u_pred, u_predicted.astype(np.float64))

    e_fn = fem.Function(V)
    e_fn.x.array[:] = u_pred.x.array - u_ref.x.array
    e_fn.x.scatter_forward()

    form_h1 = fem.form(inner(grad(e_fn), grad(e_fn)) * dx_m)
    h1_sq = comm.allreduce(fem.assemble_scalar(form_h1), op=MPI.SUM)
    h1_err = float(np.sqrt(max(h1_sq, 0.0)))

    form_h1_ref = fem.form(inner(grad(u_ref), grad(u_ref)) * dx_m)
    h1_ref_sq = comm.allreduce(fem.assemble_scalar(form_h1_ref), op=MPI.SUM)
    h1_ref = float(np.sqrt(max(h1_ref_sq, 0.0)))

    u_zero = np.array([0.0, 0.0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_zero, boundary_dofs_left, V)
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
    def __init__(self, y0, y1, y2, y3):
        self.y0 = y0
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

    def eval(self, x):
        # x shape: (3, N_points)
        B_x = self.y0 + (self.y3 - self.y0) / 4.0 * x[0]
        T_x = self.y1 + (self.y2 - self.y1) / 4.0 * x[0]
        # Interior test of the trapezoid
        return (x[1] - B_x) * (x[1] - T_x)

    def omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) <= 3e-16

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) >= -3e-16

class GExpr:
    def __init__(self, boundary_values_1d, load_factor):
        self.values = boundary_values_1d
        self.factor = load_factor
        # Distributed along the y-axis on [0, 2]
        self.y_grid = np.linspace(0.0, 2.0, len(boundary_values_1d))

    def eval(self, x):
        # Interpolate along the y coordinate (x[1])
        val_interp = np.interp(x[1], self.y_grid, self.values)
        # Mask: keep values only at x=4.0, set everything else to 0
        # mask = (np.abs(x[0] - 4.0) < 1e-4).astype(float)
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
        params,
        G_data,
        linear_solver="lu",
        ksp_gmres_restart=200,
        ksp_rtol_gmres=1e-5,
        ksp_max_it_gmres=3000,
    ):
        self.N_cells = nb_cell  # here N_cells_x = N_cells_y = nb_cell
        self.N_nodes = nb_cell + 1 # e.g. 64
        self.params = params
        # Unify input format: (batch, Ny, Nx, 2)
        self.G_data = G_data
        self.linear_solver = str(linear_solver).lower()
        if self.linear_solver not in ("lu", "gmres"):
            raise ValueError("linear_solver must be 'lu' or 'gmres'")
        self.ksp_gmres_restart = int(ksp_gmres_restart)
        self.ksp_rtol_gmres = float(ksp_rtol_gmres)
        self.ksp_max_it_gmres = int(ksp_max_it_gmres)

        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0, 0], [4, 2]]),
            np.array([self.N_cells, self.N_cells]),
            cell_type=CellType.triangle # change cell type
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

        expr_sorted = res_sorted.reshape(self.N_nodes, self.N_nodes, bs)
        expr_sorted = np.transpose(expr_sorted, (2, 1, 0)) # output (bs, Nx, Ny)
        return expr_sorted

    def _sorted_vector_to_matrix(self, vector_array, bs):
        """Internal helper: reshape a flat vector and sort it into a matrix"""
        # Reshape to (Num_Nodes, Block_Size)
        res_reshaped = vector_array.reshape((-1, bs))
        res_sorted = res_reshaped[self.sorted_indices]

        # Likewise, reshape as a rectangle
        expr_sorted = res_sorted.reshape(self.N_nodes, self.N_nodes, bs)
        return np.transpose(expr_sorted, (2, 1, 0))

    def get_analytical_matrix(self, func_eval, V_target):
        """Interpolate an analytic function directly onto the macro mesh"""
        u_target = dolfinx.fem.Function(V_target)
        u_target.interpolate(func_eval)
        u_target.x.scatter_forward()

        return self._sorted_vector_to_matrix(u_target.x.array, V_target.dofmap.index_map_bs)

    def solve_one(self, i):
        self.index = i
        param = self.params[i]
        
        # 1. Unpack parameters
        gamma_G, y0, y1, y2, y3, amplitude_f = (
            param[0],
            param[1],
            param[2],
            param[3],
            param[4],
            param[5],
        )
        
        # Extract 1D boundary data of G (right edge x=4.0, load in y)
        g_sample = self.G_data[i]   # g_sample shape: (Ny, Nx, 2)
        boundary_1d = g_sample[:, -1, 1]

        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1
        vertices_dim = 0

        # Define the trapezoid Phi function
        Phi_full = Phi(y0, y1, y2, y3)
        
        # 2. Cut away cells outside the trapezoid
        all_entities = np.arange(self.mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32)
        cells_outside = dfx.mesh.locate_entities(self.mesh_macro, cell_dim, Phi_full.not_omega())
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
        mesh = dfx.mesh.create_submesh(self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro)[0]

        # 3. Build function spaces
        V = dolfinx.fem.functionspace(mesh, ("CG", degV, (cell_dim,)))
        V_phi = dolfinx.fem.functionspace(mesh, ("CG", degPhi))
        Z_N = dolfinx.fem.functionspace(mesh, ("CG", degV, (cell_dim, cell_dim)))
        if degV == 1:
            Q_N = dolfinx.fem.functionspace(mesh, ("DG", degV - 1, (cell_dim,)))
        else:
            Q_N = dolfinx.fem.functionspace(mesh, ("CG", degV - 1, (cell_dim,)))

        dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
        spaces = [V]
        restricts = [dofs_V]
        restriction = [multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)]
        
        # Tag cells on the top/bottom trapezoid edges (including inactive corners)
        mesh.topology.create_connectivity(cell_dim, facet_dim)
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, vertices_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, vertices_dim)
        interior_entities = np.arange(mesh.topology.index_map(cell_dim).size_global, dtype=np.int32)
        c_to_v_map = np.reshape(c_to_v.array, (-1, 3))
        
        # Evaluate phi at all mesh vertices
        points = mesh.geometry.x
        phi_values = Phi_full.eval(points.T) 
        phi_cells = phi_values[c_to_v_map]
        
        # Tag a cell as a boundary cell if any vertex pair crosses or hits phi=0 (the two slanted edges)
        cells_boundary_all = (
            ((phi_cells[:, 0] * phi_cells[:, 1]) <= 0.0)
            | ((phi_cells[:, 0] * phi_cells[:, 2]) <= 0.0)
            | ((phi_cells[:, 1] * phi_cells[:, 2]) <= 0.0)
            | (near(phi_cells[:, 0] * phi_cells[:, 1], 0.0))
            | (near(phi_cells[:, 0] * phi_cells[:, 2], 0.0))
            | (near(phi_cells[:, 1] * phi_cells[:, 2], 0.0))
        )
        
        neumann_cells_all = np.unique(np.where(cells_boundary_all == True)[0])
        
        neumann_cells = neumann_cells_all
        
        c2f_map = np.reshape(c_to_f.array, (-1, 3))
        neumann_facets = np.unique(c2f_map[neumann_cells].flatten()) if len(neumann_cells) > 0 else np.array([], dtype=np.int32)
        
        # Extract ghost-penalty facets
        if len(neumann_cells_all) > 0:
            omega_1_small_cells = np.setdiff1d(interior_entities, neumann_cells_all)
            omega_1_small_facets = c2f_map[omega_1_small_cells].flatten()
            neumann_stab_facets = np.unique(np.intersect1d(omega_1_small_facets, neumann_facets))
        else:
            neumann_stab_facets = np.array([], dtype=np.int32)

        # Assign tag values on the mesh
        TAG_TRAP = 4   # regular top/bottom slanted edges of the trapezoid
        TAG_STAB = 30  # stabilization facets
        TAG_RIGHT = 1  # right boundary x=4 (loaded face)

        # Build meshtags for cells
        values_cells = np.full(len(neumann_cells), TAG_TRAP, dtype=np.intc)
        entities_cells = neumann_cells
        sorted_cells = np.argsort(entities_cells)
        subdomains_cell = dfx.mesh.meshtags(mesh, cell_dim, entities_cells[sorted_cells], values_cells[sorted_cells])
        
        # Extract the loaded right face
        def right_boundary_physical(x):
            on_right = np.isclose(x[0], 4.0)
            phi_val = Phi_full.eval(x) 
            in_domain = phi_val <= 1e-10
            return on_right & in_domain
        #############################################################################################################################################################################
        right_facets_loc = dfx.mesh.locate_entities_boundary(mesh, facet_dim, right_boundary_physical)
        # right_facets_loc = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 4.0))
        left_facets_loc = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 0.0))
        neumann_facets = np.setdiff1d(neumann_facets, right_facets_loc)
        neumann_facets = np.setdiff1d(neumann_facets, left_facets_loc)
        neumann_stab_facets = np.setdiff1d(neumann_stab_facets, left_facets_loc)

        # Build facet tags via a safe tag array to avoid backend conflicts from duplicate facets
        facet_tags = np.zeros(mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts, dtype=np.int32)
        # Priority: right edge > stabilization > cut-boundary terms
        facet_tags[neumann_facets] = TAG_TRAP
        facet_tags[neumann_stab_facets] = TAG_STAB
        facet_tags[right_facets_loc] = TAG_RIGHT 

        tagged_facets = np.where(facet_tags > 0)[0].astype(np.int32)
        tagged_values = facet_tags[tagged_facets].astype(np.intc)
        subdomains_facet = dfx.mesh.meshtags(mesh, facet_dim, tagged_facets, tagged_values)

        # Restrict the dof spaces
        mesh_neumann = dfx.mesh.create_submesh(mesh, cell_dim, neumann_cells)[0]
        restr_Neumann_Z_N = dolfinx.fem.locate_dofs_topological(Z_N, cell_dim, list(neumann_cells))
        restr_Neumann_Q_N = dolfinx.fem.locate_dofs_topological(Q_N, cell_dim, list(neumann_cells))
        restricts.append(restr_Neumann_Z_N)
        restricts.append(restr_Neumann_Q_N)
        restriction.append(multiphenicsx.fem.DofMapRestriction(Z_N.dofmap, restr_Neumann_Z_N))
        restriction.append(multiphenicsx.fem.DofMapRestriction(Q_N.dofmap, restr_Neumann_Q_N))
        spaces.append(Z_N)
        spaces.append(Q_N)

        # Define measures
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains_cell, metadata={"quadrature_degree": 4})
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=subdomains_facet, metadata={"quadrature_degree": 4}) 
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=subdomains_facet, metadata={"quadrature_degree": 4})

        # Physical and numerical parameters
        nb_incr = 4
        gamma_div, gamma_u, gamma_p = 0.01, 0.001, 0.01
        sigma_N = 0.1
        sigma_D = 20

        uyp_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
        vzq_split = [ufl.TestFunction(VVV) for VVV in spaces]
        duyp_split = [ufl.TrialFunction(VVV) for VVV in spaces]
        
        # Interpolate the single Phi_full onto V_phi (no loop)
        phi_func = dolfinx.fem.Function(V_phi)
        phi_func.interpolate(Phi_full.eval)
        
        current_load = 0.0
        target_load = 1.0 
        d_load = 1.0 / nb_incr 
        min_d_load = 1e-2 
        solved_successfully = False

        while current_load < target_load:
            next_load_factor = current_load + d_load
            if next_load_factor > target_load:
                next_load_factor = target_load
                
            print(f"Trying load factor: {next_load_factor:.4f} (Step size: {d_load:.4f})")
            
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
            
            # Base variational terms and external load
            F[0] += ufl.inner(Pu1, ufl.grad(v1)) * dx - ufl.inner(g, v1) * ds(TAG_RIGHT)

            index_ = 1
            dx_N = dx(TAG_TRAP)
            ds_N = ds(TAG_TRAP)
            dS_N_stab = dS(TAG_STAB)

            Gh1 = sigma_N * ufl.avg(h) * ufl.inner(ufl.jump(Pu1, n), ufl.jump(Pv1, n)) * dS_N_stab
            F[0] += gamma_u * ufl.inner(Pu1, Pv1) * dx_N + Gh1

            F[0] += (ufl.inner(ufl.dot(y, n), v1) * ds_N + gamma_u * ufl.inner(y, Pv1) * dx_N)

            F[index_] += gamma_u * ufl.inner(Pu1, z) * dx_N
            F[index_] += (
                gamma_u * ufl.inner(y, z) * dx_N
                + gamma_div * ufl.inner(ufl.div(y), ufl.div(z)) * dx_N
                + gamma_p * h ** (-2) * ufl.inner(ufl.dot(y, ufl.grad(phi_func)), ufl.dot(z, ufl.grad(phi_func))) * dx_N
            )
            F[index_] += (gamma_p * h ** (-3) * ufl.inner(p_N * phi_func, ufl.dot(z, ufl.grad(phi_func))) * dx_N)

            F[index_ + 1] += (gamma_p * h ** (-3) * ufl.inner(ufl.dot(y, ufl.grad(phi_func)), q_N * phi_func) * dx_N)
            F[index_ + 1] += (gamma_p * h ** (-4) * ufl.inner(p_N * phi_func, q_N * phi_func) * dx_N)

            # Assemble the Jacobian
            J = [[ufl.derivative(F[i], uyp_split[j], duyp_split[j]) for j in range(len(uyp_split))] for i in range(len(F))]
            
            # Left Dirichlet fixed boundary (x=0)
            zeros_1d = np.zeros_like(boundary_1d) 
            g_expr_null = GExpr(zeros_1d, load_factor=1.0)
            g_null = dolfinx.fem.Function(V)
            g_null.interpolate(g_expr_null.eval)
            lower_facets = dfx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[0], 0.0))
            boundary_dofs_low = fem.locate_dofs_topological(V, 1, lower_facets)
            bc_low = fem.dirichletbc(g_null, boundary_dofs_low)
            
            problem = NonLinearPhiFEM(F, J, tuple(uyp_split), [bc_low], restriction, spaces)
            F_vec = mphx.fem.petsc.create_vector_block(problem._F, restriction=restriction)
            J_mat = mphx.fem.petsc.create_matrix_block(problem._J, restriction=(restriction, restriction))
            
            snes = petsc4py.PETSc.SNES().create(mesh.comm)
            snes.setTolerances(max_it=50)
            snes.setType("newtonls") 
            ksp = snes.getKSP()
            if self.linear_solver == "lu":
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.getPC().setFactorSolverType("mumps")
            else:
                ksp.setType("gmres")
                try:
                    ksp.setGMRESRestart(int(self.ksp_gmres_restart))
                except Exception:
                    pass
                pc = ksp.getPC()
                try:
                    pc.setType("ilu")
                except Exception:
                    pc.setType("jacobi")
                ksp.setTolerances(
                    rtol=float(self.ksp_rtol_gmres),
                    atol=1e-12,
                    max_it=int(self.ksp_max_it_gmres),
                )
            opts = PETSc.Options()
            opts["snes_linesearch_type"] = "bt"
            snes.setObjective(problem.obj)
            snes.setFunction(problem.F, F_vec)
            snes.setJacobian(problem.J, J=J_mat, P=None)
            
            solution = problem.create_snes_solution()
            try:
                snes.solve(None, solution)
                converged_reason = snes.getConvergedReason()
            except Exception as e:
                converged_reason = -1
                print(f"Solver crashed: {e}")

            if converged_reason > 0:
                print(f"Converged at {next_load_factor:.4f}")
                current_load = next_load_factor
                problem.update_solutions(solution)
                if d_load < (1.0 / nb_incr):
                    d_load = min(d_load * 1.5, 1.0 / nb_incr)
                if current_load >= target_load:
                    solved_successfully = True
                    break
            else:
                print(f"Failed at {next_load_factor:.4f}. Reducing step size.")
                d_load /= 2.0
                if d_load < min_d_load:
                    print("Step size too small. Aborting calculation for this sample.")
                    break
        
        if not solved_successfully:
            print(f"Sample {self.index} failed to reach full load. Max load: {current_load}")
            return None, None, None, None, None

        # 7. Extract results onto the macro mesh (unchanged)
        u_sol = uyp_split[0]
        u_mat = self.make_matrix(u_sol, V, self.V_macro)
        y_sol = uyp_split[1]
        y_mat = self.make_matrix(y_sol, Z_N, self.V_macro_tensor)
        p_sol = uyp_split[2]
        p_mat = self.make_matrix(p_sol, Q_N, self.V_macro)

        phi_mat = self.get_analytical_matrix(Phi_full.eval, self.V_macro_scalar)
        g_expr_wrapper = GExpr(boundary_1d, 1)
        g_mat = self.get_analytical_matrix(g_expr_wrapper.eval, self.V_macro)
        
        return u_mat, y_mat, p_mat, phi_mat, g_mat

    def solve_several(self):
        U_list = []
        Y_list = []
        P_list = []
        S_list = []
        Phi_list = []
        G_list = []
        nb = len(self.params)
        for i in range(nb):
            print(f"Data : {i}/{nb}")
            u_mat, y_mat, p_mat, phi_mat, g_mat = self.solve_one(i)
            if u_mat is None:
                continue
            U_list.append(u_mat)
            Y_list.append(y_mat)
            P_list.append(p_mat)
            G_list.append(g_mat)
            Phi_list.append(phi_mat)
            # if len(U_list) == 1100:
            #     break

        return (np.stack(U_list), np.stack(Y_list), np.stack(P_list),
                np.stack(Phi_list), np.stack(G_list))

def create_parameters(nb_vert=51, nb_training_shapes=300):
    """
    Generate trapezoid inputs (in memory only); uniformly return:
      - params: (N, 5)
      - phi_out: (N, Ny, Nx, 1)
      - g_out: (N, Ny, Nx, 2)
    """
    nb_data = nb_training_shapes
    _, phi, G, params = create_FG_numpy(nb_data=nb_data, nb_vert=nb_vert)
    phi_out = phi[:, :, :, None]
    # Unify to (batch, Ny, Nx, 2)
    g_out = G.transpose((0, 3, 2, 1))
    return params, phi_out, g_out

def go(
    params,
    G,
    nb_vert=51,
    linear_solver="lu",
    ksp_gmres_restart=120,
    ksp_rtol_gmres=1e-5,
    ksp_max_it_gmres=500,
    n_train=1000,
    n_test=100,
):
    """
    Main function to generate data.
    Save only two npz files (train then test in batch order: first n_train, then n_test).
    """
    save = True
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(base_dir), "data")

    ti0 = time.time()
    # PhiFemSolver inputs:
    #   params: (N, 5)
    #   G: (N, Ny, Nx, 2)
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1,
        params=params,
        G_data=G,
        linear_solver=linear_solver,
        ksp_gmres_restart=ksp_gmres_restart,
        ksp_rtol_gmres=ksp_rtol_gmres,
        ksp_max_it_gmres=ksp_max_it_gmres,
    )
    U, Y, P, Phi, G_out = solver.solve_several()
    # Unify to (batch, Ny, Nx, channels)
    U = U.transpose((0, 3, 2, 1))
    Y = Y.transpose((0, 3, 2, 1))
    P = P.transpose((0, 3, 2, 1))
    Phi = Phi.transpose((0, 3, 2, 1))
    G_out = G_out.transpose((0, 3, 2, 1))
    duration = time.time() - ti0
    print("duration to solve u:", duration)

    if save:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        model_inputs = np.concatenate([Phi, G_out], axis=-1)
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
            data_dir,
            f"Hyperelasticity_Trapezoid_G_u_s{nb_vert}_n{n_tr}_train.npz",
        )
        path_test = os.path.join(
            data_dir,
            f"Hyperelasticity_Trapezoid_G_u_s{nb_vert}_n{n_te}_test.npz",
        )
        if n_tr > 0:
            np.savez_compressed(
                path_train,
                inputs=model_inputs[:n_tr],
                targets=model_targets[:n_tr],
            )
            print(f"saved: {path_train}")

        np.savez_compressed(
            path_test,
            inputs=model_inputs[n_tr : n_tr + n_te],
            targets=model_targets[n_tr : n_tr + n_te],
        )
        print(f"saved: {path_test}")

if __name__ == "__main__":
    nb_vert = 51
    params, _, G = create_parameters(nb_vert=nb_vert, nb_training_shapes=1100)
    go(
        params,
        G,
        nb_vert=nb_vert,
        n_train=1000,
        n_test=100,
    )
    # go_val()
    # go_test()
    # pass
