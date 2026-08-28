"""
    Second order, incremental iteration, triangular elements, N*N, quarter-ellipse pressure vessel, Q9 output, Ein/Eout inputs, follower force p
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
# degPhi = 1
E_val = 200

def call_F(xy, amplitude):
    return np.array([0 * xy[0] + amplitude, amplitude + 0 * xy[1]])

def generate_random_boundary(y_grid, mean_val, variance, length_scale=0.4):
    """Generate a 1D Gaussian random field along the y-axis"""
    N = len(y_grid)
    N_control = 32
    jitter = 1e-8

    y_min, y_max = y_grid.min(), y_grid.max()
    y_control = np.linspace(y_min, y_max, N_control)

    diffs = y_control[:, None] - y_control[None, :]
    K = np.exp(-0.5 * (diffs**2) / (length_scale**2))
    L = np.linalg.cholesky(K + jitter * np.eye(N_control))

    z = np.random.normal(size=N_control)
    val_control = np.dot(L, z)
    val_final = np.interp(y_grid, y_control, val_control)

    return mean_val + np.sqrt(variance) * val_final

def create_FG_numpy(nb_data, nb_vert):
    Nx = Ny = nb_vert
    L_max = 1.0
    xx = np.linspace(0.0, L_max, Nx)
    yy = np.linspace(0.0, L_max, Ny)
    XX, YY = np.meshgrid(xx, yy)
    XX_flat = XX.flatten()
    YY_flat = YY.flatten()
    XXYY = np.stack([XX_flat, YY_flat])

    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(nb_data)
    low_bounds = [10, -1.0]
    up_bounds = [10.000001, 1.0]
    params_cont = qmc.scale(sample, low_bounds, up_bounds)

    thick_arr = np.random.uniform(0.2, 0.25, size=nb_data)
    a_out_arr = np.random.uniform(0.9, 1, size=nb_data)
    b_out_arr = np.random.uniform(0.9, 1, size=nb_data)
    a_in_arr = a_out_arr - thick_arr
    b_in_arr = b_out_arr - thick_arr

    gamma_samples = params_cont[:, 0:1]
    amplitude_f = params_cont[:, 1:2]

    params = np.column_stack([gamma_samples, amplitude_f, a_in_arr, b_in_arr, a_out_arr, b_out_arr])

    print(f"Generating Phi...")
    Phi = []
    Phi_dx = []
    Phi_dy = []

    for n in range(nb_data):
        a_in, b_in = a_in_arr[n], b_in_arr[n]
        a_out, b_out = a_out_arr[n], b_out_arr[n]

        E_in = np.sqrt((XX_flat**2 / a_in**2) + (YY_flat**2 / b_in**2)) - 1.0
        E_out = np.sqrt((XX_flat**2 / a_out**2) + (YY_flat**2 / b_out**2)) - 1.0

        E_in_grid = E_in.reshape((Ny, Nx))
        E_out_grid = E_out.reshape((Ny, Nx))

        epsilon = 1e-12
        Ein_dx = XX / (a_in**2 * (np.sqrt((XX**2 / a_in**2) + (YY**2 / b_in**2)) + epsilon))
        Ein_dy = YY / (b_in**2 * (np.sqrt((XX**2 / a_in**2) + (YY**2 / b_in**2)) + epsilon))
        Eout_dx = XX / (a_out**2 * (np.sqrt((XX**2 / a_out**2) + (YY**2 / b_out**2)) + epsilon))
        Eout_dy = YY / (b_out**2 * (np.sqrt((XX**2 / a_out**2) + (YY**2 / b_out**2)) + epsilon))

        Phi.append(np.stack([E_in_grid, E_out_grid], axis=0))
        Phi_dx.append(np.stack([Ein_dx, Eout_dx], axis=0))
        Phi_dy.append(np.stack([Ein_dy, Eout_dy], axis=0))
    Phi = np.array(Phi)
    Phi_dx = np.array(Phi_dx)
    Phi_dy = np.array(Phi_dy)
    Phi = np.permute_dims(Phi, (0, 2, 3, 1))
    E_dx = np.permute_dims(Phi_dx, (0, 2, 3, 1))
    E_dy = np.permute_dims(Phi_dy, (0, 2, 3, 1))

    print(f"Generating F...")
    F = call_F(XXYY, amplitude_f)
    F = np.reshape(F, [2, nb_data, Ny, Nx]).transpose(1, 0, 3, 2)

    print(f"Generating G ...")
    G_list = []
    theta_grid = np.linspace(0, np.pi / 2, Ny)

    for n in range(nb_data):
        target_mean = gamma_samples[n, 0]
        boundary_profile = generate_random_boundary(theta_grid, mean_val=target_mean, variance=0.1)
        g_scalar_field = np.tile(boundary_profile[:, None], (1, Nx))
        g_sample = np.expand_dims(g_scalar_field, axis=0)
        G_list.append(g_sample)

    G = np.array(G_list).transpose(0, 1, 3, 2)
    return F, Phi, E_dx, E_dy, G, params

def tensors(u):
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))
    E = E_val
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
    """field_sample (Ny, Nx, 2), interpolate onto the CG target space."""
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

    Note: ``K`` contains only the Neo-Hookean volume term, not the full Phi-FEM block Jacobian.
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

class Phi_vessel:
    """Level set of the quarter-ellipse pressure vessel"""
    def __init__(self, a_in, b_in, a_out, b_out):
        self.a_in = a_in
        self.b_in = b_in
        self.a_out = a_out
        self.b_out = b_out

    def eval(self, x):
        E_in = np.sqrt((x[0]**2 / self.a_in**2) + (x[1]**2 / self.b_in**2)) - 1.0
        E_out = np.sqrt((x[0]**2 / self.a_out**2) + (x[1]**2 / self.b_out**2)) - 1.0
        # thickness = (self.a_out - self.a_in + self.b_out - self.b_in) / 2.0
        return E_in * E_out

    # def eval(self, x):
    #     # here x includes x[0], x[1]
    #     r = np.sqrt(x[0]**2 + x[1]**2)
        
    #     # Compute the scale factor; even if r=0, scale is still meaningful
    #     scale_in  = np.sqrt((x[0]/self.a_in)**2 + (x[1]/self.b_in)**2) + 1e-12
    #     scale_out = np.sqrt((x[0]/self.a_out)**2 + (x[1]/self.b_out)**2) + 1e-12
        
    #     r_in  = r / scale_in
    #     r_out = r / scale_out
        
    #     # thickness at the origin converges to some mean of (a_out - a_in) or (b_out - b_in)
    #     thick = r_out - r_in + 1e-12
        
    #     return (r - r_in) * (r - r_out) / thick

    def not_omega(self):
        return lambda x: self.eval(x.reshape((3, -1))) >= -3e-16

def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol

def _system_cpu_info() -> dict:
    """Host logical CPU count (best effort)."""
    logical = os.cpu_count() or 1
    info = {"logical_cores": int(logical)}
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm is not None:
        try:
            info["slurm_cpus_on_node"] = int(slurm)
        except ValueError:
            pass
    return info

def _parallel_runtime_info(comm) -> dict:
    """MPI ranks, OpenMP threads, and host CPU/core counts."""
    try:
        mpi_size = int(comm.Get_size())
        mpi_rank = int(comm.Get_rank())
    except Exception:
        mpi_size, mpi_rank = 1, 0
    omp_env = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS") or ""
    if omp_env:
        try:
            omp_threads: int | str = int(omp_env)
        except ValueError:
            omp_threads = omp_env
    else:
        omp_threads = 1
    return {
        "mpi_size": mpi_size,
        "mpi_rank": mpi_rank,
        "omp_threads": omp_threads,
        **_system_cpu_info(),
    }

def _print_parallel_runtime(comm, *, header: str = "Parallel runtime") -> None:
    if comm.rank != 0:
        return
    info = _parallel_runtime_info(comm)
    print(f"\n=== {header} ===")
    print(f"  host logical cores (os.cpu_count): {info['logical_cores']}")
    if "slurm_cpus_on_node" in info:
        print(f"  SLURM_CPUS_ON_NODE: {info['slurm_cpus_on_node']}")
    print(f"  MPI processes (ranks): {info['mpi_size']}")
    print(f"  OpenMP threads (OMP_NUM_THREADS / MKL_NUM_THREADS): {info['omp_threads']}")
    omp_n = info["omp_threads"] if isinstance(info["omp_threads"], int) else 1
    print(f"  effective parallel units (MPI x OpenMP): {info['mpi_size'] * omp_n}")

def _mesh_activity_report(mesh) -> dict:
    """Global cell / geometry-vertex counts on the phi-cut submesh."""
    dim = mesh.topology.dim
    return {
        "active_cells": int(mesh.topology.index_map(dim).size_global),
        "active_nodes": int(mesh.topology.index_map(0).size_global),
    }

def _count_block_dof_breakdown(spaces, restriction) -> list[int]:
    """Per-block scalar DOF counts (local, before MPI sum)."""
    blocks: list[int] = []
    for sp, restr in zip(spaces, restriction):
        bs = int(sp.dofmap.index_map_bs)
        if isinstance(restr, mphx.fem.DofMapRestriction):
            blocks.append(int(restr.index_map.size_local) * bs)
        else:
            blocks.append(int(np.asarray(restr, dtype=np.int64).size))
    return blocks

def _count_active_block_dofs(spaces, restriction) -> int:
    """Scalar DOFs in the multiphenicsx block system (after restriction)."""
    return int(sum(_count_block_dof_breakdown(spaces, restriction)))

def _active_dof_report(comm, spaces, restriction, block_names=None) -> dict:
    """Local breakdown + MPI-global totals for the SNES block unknown."""
    local = _count_block_dof_breakdown(spaces, restriction)
    if block_names is None:
        block_names = ["u", "y_N", "p_N"][: len(local)]
    global_blocks = [comm.allreduce(n, op=MPI.SUM) for n in local]
    return {
        "active_dofs": int(sum(global_blocks)),
        "dof_breakdown": dict(zip(block_names, global_blocks)),
    }

def _print_solve_profile(
    label: str,
    comm,
    active_dofs: int,
    profile: dict,
    *,
    indent: str = "  ",
) -> None:
    if comm.rank != 0:
        return
    runtime = _parallel_runtime_info(comm)
    print(f"{indent}[{label}]")
    print(f"{indent}  host logical cores: {runtime['logical_cores']}")
    print(f"{indent}  MPI processes (ranks): {runtime['mpi_size']}")
    print(f"{indent}  OpenMP threads: {runtime['omp_threads']}")
    if profile.get("active_cells") is not None:
        print(f"{indent}  active cells (submesh): {profile['active_cells']}")
    if profile.get("active_nodes") is not None:
        print(f"{indent}  active nodes (submesh vertices): {profile['active_nodes']}")
    print(f"{indent}  active DOFs (block, global): {active_dofs}")
    breakdown = profile.get("dof_breakdown")
    if breakdown:
        for name, n in breakdown.items():
            print(f"{indent}    - {name}: {n}")
    petsc_n = profile.get("active_dofs_petsc")
    if petsc_n is not None and int(petsc_n) != int(active_dofs):
        print(f"{indent}  active DOFs (PETSc Vec global size): {petsc_n}")
    print(f"{indent}  assembly time: {profile.get('assembly_time', 0.0):.4f} s")
    print(f"{indent}  factorization time: {profile.get('factorization_time', 0.0):.4f} s")
    print(f"{indent}  linear solve time: {profile.get('linear_solve_time', 0.0):.4f} s")
    print(f"{indent}  I/O time: {profile.get('io_time', 0.0):.4f} s")
    if "newton_iters" in profile:
        print(f"{indent}  Newton iterations: {profile['newton_iters']}")

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
        self._timing = {
            "assembly_F": 0.0,
            "assembly_J": 0.0,
            "lu_factor": 0.0,
        }
        self._track_lu_factor = True

    def reset_timing(self) -> None:
        for key in self._timing:
            self._timing[key] = 0.0

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
        t0 = time.perf_counter()
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
        self._timing["assembly_F"] += time.perf_counter() - t0

    def J(  # type: ignore[no-any-unimported]
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        J_mat: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat,
    ) -> None:
        """Assemble the jacobian."""
        t0 = time.perf_counter()
        J_mat.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix_block(
            J_mat,
            self._J,
            self._bcs,
            diagonal=1.0,  # type: ignore[arg-type]
            restriction=(self._restriction, self._restriction),
        )
        J_mat.assemble()
        self._timing["assembly_J"] += time.perf_counter() - t0
        if self._track_lu_factor:
            try:
                pc = snes.getKSP().getPC()
                t1 = time.perf_counter()
                pc.setUp()
                self._timing["lu_factor"] += time.perf_counter() - t1
            except Exception:
                pass
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
    def __init__(self, nb_cell, params, G_data, nb_incr=3):
        self.N_cells = nb_cell
        self.N_nodes = nb_cell + 1
        self.params = params
        self.G_data = G_data
        self.nb_incr = nb_incr
        self.L_max = 1
        self.mesh_macro = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0.0, 0.0], [self.L_max, self.L_max]]),
            np.array([self.N_cells, self.N_cells]),
            cell_type=CellType.triangle
        )
        # Linear space for loading the 64x64 g_sample
        self.V_macro_cg1_vector = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 1, (self.mesh_macro.geometry.dim,)))
        coords_cg1 = self.V_macro_cg1_vector.tabulate_dof_coordinates()[:, :2]
        self.cg1_indices = np.lexsort((np.round(coords_cg1[:, 0], 5), np.round(coords_cg1[:, 1], 5)))
        
        # Quadratic space for sampling 127x127 Q9 data
        self.N_q9 = 2 * self.N_cells + 1 
        cell_dim = self.mesh_macro.geometry.dim
        self.V_macro_scalar = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 2))
        self.V_macro_vector = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 2, (cell_dim,)))
        self.V_macro_tensor = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 2, (cell_dim, cell_dim)))

        coords_q9 = self.V_macro_scalar.tabulate_dof_coordinates()[:, :2]
        self.q9_indices = np.lexsort((np.round(coords_q9[:, 0], 6), np.round(coords_q9[:, 1], 6)))

        # --- Space 1: CG1 (64x64), receive raw input ---
        self.V_macro_cg1 = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 1, (2,)))
        self.V_macro_cg1_scalar = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 1))
        coords_cg1 = self.V_macro_cg1.tabulate_dof_coordinates()[:, :2]
        self.cg1_indices = np.lexsort((np.round(coords_cg1[:, 0], 5), np.round(coords_cg1[:, 1], 5)))

        # --- Space 2: CG2 (127x127), produce biquadratic interpolated output ---
        self.V_macro_cg2 = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 2, (2,)))
        self.V_macro_cg2_scalar = dolfinx.fem.functionspace(self.mesh_macro, ("CG", 2))
        coords_cg2 = self.V_macro_cg2.tabulate_dof_coordinates()[:, :2]
        self.q9_indices = np.lexsort((np.round(coords_q9[:, 0], 6), np.round(coords_q9[:, 1], 6)))
        
        self.padding = 1e-14

    def make_matrix(self, expr, V, V_target):
        expr.x.scatter_forward()
        u2 = dolfinx.fem.Function(V_target)
        u1_2_u2_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(u2.function_space.mesh, u2.function_space.element, V.mesh, padding=self.padding)
        u2.interpolate(expr, nmm_interpolation_data=u1_2_u2_nmm_data)
        u2.x.scatter_forward()
        
        bs = V_target.dofmap.index_map_bs
        res = u2.x.array.reshape((-1, bs))[self.sorted_indices]
        expr_sorted = res.reshape(self.N_nodes, self.N_nodes, bs)
        return np.transpose(expr_sorted, (2, 1, 0))

    def _sorted_vector_to_matrix(self, vector_array, bs):
        res_sorted = vector_array.reshape((-1, bs))[self.sorted_indices]
        expr_sorted = res_sorted.reshape(self.N_nodes, self.N_nodes, bs)
        return np.transpose(expr_sorted, (2, 1, 0))

    def get_analytical_matrix(self, func_eval, V_target):
        u_target = dolfinx.fem.Function(V_target)
        u_target.interpolate(func_eval)
        u_target.x.scatter_forward()
        return self._sorted_vector_to_matrix(u_target.x.array, V_target.dofmap.index_map_bs)

    def matrix_to_function(self, mat, V_target):
        bs = V_target.dofmap.index_map_bs
        mat_T = np.transpose(mat, (2, 1, 0)) 
        res_sorted = mat_T.reshape((-1, bs))
        
        # Must use the 64x64-specific sorting indices
        inverse_indices = np.argsort(self.cg1_indices)
        res_reshaped = res_sorted[inverse_indices]
        
        u_target = dolfinx.fem.Function(V_target)
        u_target.x.array[:] = res_reshaped.flatten()
        u_target.x.scatter_forward()
        return u_target

    def solve_one(self, i):
        self.index = i
        param = self.params[i]
        t_io = time.perf_counter()

        a_in, b_in, a_out, b_out = param[2], param[3], param[4], param[5]
        g_sample = self.G_data[i] 
        
        cell_dim = self.mesh_macro.geometry.dim
        facet_dim = self.mesh_macro.geometry.dim - 1

        Phi_full = Phi_vessel(a_in, b_in, a_out, b_out)
        
        # 2. Cut away cells outside the trapezoid
        all_entities = np.arange(self.mesh_macro.topology.index_map(cell_dim).size_global, dtype=np.int32)
        cells_outside = dfx.mesh.locate_entities(self.mesh_macro, cell_dim, Phi_full.not_omega())
        interior_entities_macro = np.setdiff1d(all_entities, cells_outside)
        mesh = dfx.mesh.create_submesh(self.mesh_macro, self.mesh_macro.topology.dim, interior_entities_macro)[0]

        mesh.topology.create_connectivity(cell_dim, cell_dim)
        mesh.topology.create_connectivity(facet_dim, cell_dim)
        mesh.topology.create_connectivity(cell_dim, facet_dim)
        mesh.topology.create_connectivity(cell_dim, 0)

        V = dolfinx.fem.functionspace(mesh, ("CG", degV, (cell_dim,)))
        V_phi = dolfinx.fem.functionspace(mesh, ("CG", degPhi))
        Z_N = dolfinx.fem.functionspace(mesh, ("CG", degV, (cell_dim, cell_dim)))
        Q_N = dolfinx.fem.functionspace(mesh, ("CG", degV - 1, (cell_dim,))) if degV > 1 else dolfinx.fem.functionspace(mesh, ("DG", 0, (cell_dim,)))

        dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)
        spaces, restricts = [V], [dofs_V]
        restriction = [multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)]
        
        c_to_f = mesh.topology.connectivity(cell_dim, facet_dim)
        c_to_v = mesh.topology.connectivity(cell_dim, 0)
        interior_entities = np.arange(mesh.topology.index_map(cell_dim).size_global, dtype=np.int32)
        c_to_v_map = np.reshape(c_to_v.array, (-1, 3))
        
        points = mesh.geometry.x
        phi_values = Phi_full.eval(points.T) 
        phi_cells = phi_values[c_to_v_map]
        
        cells_boundary_all = (
            ((phi_cells[:, 0] * phi_cells[:, 1]) <= 0.0) | ((phi_cells[:, 0] * phi_cells[:, 2]) <= 0.0) | ((phi_cells[:, 1] * phi_cells[:, 2]) <= 0.0) |
            (near(phi_cells[:, 0] * phi_cells[:, 1], 0.0)) | (near(phi_cells[:, 0] * phi_cells[:, 2], 0.0)) | (near(phi_cells[:, 1] * phi_cells[:, 2], 0.0))
        )
        neumann_cells_all = np.unique(np.where(cells_boundary_all == True)[0]).astype(np.int32)

        # Drop fake cut-cells near x=0 or y=0 that lie fully inside the solid (phi <= 0)
        # Condition A: cells near a symmetry axis (any vertex with x or y near 0)
        at_x_zero_cells = (points[c_to_v_map][:, :, 0] <= 1e-10).any(axis=1)
        at_y_zero_cells = (points[c_to_v_map][:, :, 1] <= 1e-10).any(axis=1)
        near_axis_cells = at_x_zero_cells | at_y_zero_cells
        
        # Condition B: cells fully inside the solid (phi <= 0 at all three vertices)
        in_domain_cells = (phi_cells <= 1e-10).all(axis=1)
        
        # Cells satisfying both A and B are the ones to exclude
        cells_to_exclude = np.where(near_axis_cells & in_domain_cells)[0].astype(np.int32)
        
        # Remove them entirely from neumann_cells_all
        if len(cells_to_exclude) > 0:
            neumann_cells_all = np.setdiff1d(neumann_cells_all, cells_to_exclude)

        at_x_zero = np.isclose(points[:, 0], 0.0)
        at_y_zero = np.isclose(points[:, 1], 0.0)
        cells_at_x_zero = np.where(at_x_zero[c_to_v_map].any(axis=1))[0].astype(np.int32)
        cells_at_y_zero = np.where(at_y_zero[c_to_v_map].any(axis=1))[0].astype(np.int32)
        boundary_axis_cells = np.union1d(cells_at_x_zero, cells_at_y_zero)
        
        # corner_cells = np.intersect1d(neumann_cells_all, boundary_axis_cells)
        # neumann_cells = np.setdiff1d(neumann_cells_all, corner_cells)
        neumann_cells = neumann_cells_all
        
        # Split inner vs outer Neumann boundaries with the mid ellipse
        a_mid = (a_in + a_out) / 2.0
        b_mid = (b_in + b_out) / 2.0
        
        pts_x = points[c_to_v_map][:, :, 0]
        pts_y = points[c_to_v_map][:, :, 1]
        
        # Evaluate the mid-ellipse equation at cell vertices
        E_mid_cells = (pts_x**2 / a_mid**2) + (pts_y**2 / b_mid**2) - 1.0
        
        # If the mean over vertices is < 0, the cell leans toward the inner wall
        inner_mask = np.mean(E_mid_cells, axis=1) < 0
        outer_mask = np.mean(E_mid_cells, axis=1) >= 0
        
        neumann_cells_inner = np.intersect1d(neumann_cells, np.where(inner_mask)[0]).astype(np.int32)
        neumann_cells_outer = np.intersect1d(neumann_cells, np.where(outer_mask)[0]).astype(np.int32)

        c2f_map = np.reshape(c_to_f.array, (-1, 3))
        neumann_facets_inner = np.unique(c2f_map[neumann_cells_inner].flatten()) if len(neumann_cells_inner) > 0 else np.array([], dtype=np.int32)
        neumann_facets_outer = np.unique(c2f_map[neumann_cells_outer].flatten()) if len(neumann_cells_outer) > 0 else np.array([], dtype=np.int32)
        # corner_facets = np.unique(c2f_map[corner_cells].flatten()) if len(corner_cells) > 0 else np.array([], dtype=np.int32)
        # Strip the x=0 and y=0 symmetry faces from Neumann facets
        # Locate all facets that lie exactly on the geometric boundaries x=0 and y=0
        facets_x0_all = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 0.0))
        facets_y0_all = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], 0.0))
        
        # Merge facets on these two symmetry axes
        axis_facets = np.union1d(facets_x0_all, facets_y0_all)
        
        # Cut these facets out of the loaded boundary with setdiff1d
        if len(neumann_facets_inner) > 0:
            neumann_facets_inner = np.setdiff1d(neumann_facets_inner, axis_facets)
            
        if len(neumann_facets_outer) > 0:
            neumann_facets_outer = np.setdiff1d(neumann_facets_outer, axis_facets)

        if len(neumann_cells_all) > 0:
            omega_1_small_cells = np.setdiff1d(interior_entities, neumann_cells_all)
            omega_1_small_facets = c2f_map[omega_1_small_cells].flatten()
            # all_bnd_facets = np.concatenate([neumann_facets_inner, neumann_facets_outer, corner_facets])
            all_bnd_facets = np.concatenate([neumann_facets_inner, neumann_facets_outer])
            neumann_stab_facets = np.unique(np.intersect1d(omega_1_small_facets, all_bnd_facets))
        else:
            neumann_stab_facets = np.array([], dtype=np.int32)

        TAG_OMEGA, TAG_INNER, TAG_OUTER, TAG_CORNER, TAG_STAB = 1, 4, 6, 5, 30

        num_cells = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
        cell_tags = np.full(num_cells, TAG_OMEGA, dtype=np.int32)
        cell_tags[neumann_cells_inner] = TAG_INNER
        cell_tags[neumann_cells_outer] = TAG_OUTER
        # cell_tags[corner_cells] = TAG_CORNER
        subdomains_cell = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype=np.int32), cell_tags.astype(np.intc))
        
        facet_tags = np.zeros(mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts, dtype=np.int32)
        facet_tags[neumann_facets_inner] = TAG_INNER
        facet_tags[neumann_facets_outer] = TAG_OUTER
        # facet_tags[corner_facets] = TAG_CORNER
        facet_tags[neumann_stab_facets] = TAG_STAB

        tagged_facets = np.where(facet_tags > 0)[0].astype(np.int32)
        subdomains_facet = dfx.mesh.meshtags(mesh, facet_dim, tagged_facets, facet_tags[tagged_facets].astype(np.intc))

        safe_neumann_cells = neumann_cells.astype(np.int32)
        restr_Neumann_Z_N = dolfinx.fem.locate_dofs_topological(Z_N, cell_dim, safe_neumann_cells)
        restr_Neumann_Q_N = dolfinx.fem.locate_dofs_topological(Q_N, cell_dim, safe_neumann_cells)
        
        restricts.extend([restr_Neumann_Z_N, restr_Neumann_Q_N])
        restriction.extend([multiphenicsx.fem.DofMapRestriction(Z_N.dofmap, restr_Neumann_Z_N), 
                            multiphenicsx.fem.DofMapRestriction(Q_N.dofmap, restr_Neumann_Q_N)])
        spaces.extend([Z_N, Q_N])

        h, n = ufl.CellDiameter(mesh), ufl.FacetNormal(mesh)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains_cell, metadata={"quadrature_degree": 4})
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=subdomains_facet, metadata={"quadrature_degree": 4}) 
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=subdomains_facet, metadata={"quadrature_degree": 4})

        nb_incr = self.nb_incr
        
        gamma_div = 0.001
        gamma_u   = 0.001
        gamma_p   = 0.001
        sigma_N   = 1  # stabilization jump term, prevent facet folding

        uyp_split = [dolfinx.fem.Function(VVV) for VVV in spaces]
        vzq_split = [ufl.TestFunction(VVV) for VVV in spaces]
        duyp_split = [ufl.TrialFunction(VVV) for VVV in spaces]
        
        phi_func = dolfinx.fem.Function(V_phi)
        phi_func.interpolate(Phi_full.eval)

        V_x, _ = V.sub(0).collapse()
        V_y, _ = V.sub(1).collapse()
        u_zero_x, u_zero_y = fem.Function(V_x), fem.Function(V_y)
        u_zero_x.x.array[:], u_zero_y.x.array[:] = 0.0, 0.0

        facets_x0 = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 0.0))
        facets_y0 = dfx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], 0.0))
        dofs_x0 = fem.locate_dofs_topological((V.sub(0), V_x), facet_dim, facets_x0)
        dofs_y0 = fem.locate_dofs_topological((V.sub(1), V_y), facet_dim, facets_y0)
        bcs = [fem.dirichletbc(u_zero_x, dofs_x0, V.sub(0)), fem.dirichletbc(u_zero_y, dofs_y0, V.sub(1))]

        p_ext_base_macro = self.matrix_to_function(g_sample, self.V_macro_cg1_scalar) 
        p_ext_base = dolfinx.fem.Function(V_phi) # map onto the local scalar space
        p_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            p_ext_base.function_space.mesh, p_ext_base.function_space.element, 
            p_ext_base_macro.function_space.mesh, padding=self.padding
        )
        p_ext_base.interpolate(p_ext_base_macro, nmm_interpolation_data=p_nmm_data)
        p_ext_base.x.scatter_forward()

        current_load = 0.0
        target_load = 1.0 
        d_load = 1.0 / nb_incr 
        min_d_load = 1e-3 
        solved_successfully = False
        prof_newton_iters = 0
        prof_assembly_time = 0.0
        prof_factorization_time = 0.0
        prof_linear_solve_time = 0.0
        prof_io_time = time.perf_counter() - t_io
        dof_report = _active_dof_report(mesh.comm, spaces, restriction)
        mesh_report = _mesh_activity_report(mesh)
        active_dofs = int(dof_report["active_dofs"])
        active_dofs_petsc = None
        if mesh.comm.rank == 0:
            print(
                f"  [solve_one {self.index}] "
                f"active cells = {mesh_report['active_cells']}, "
                f"active nodes = {mesh_report['active_nodes']}, "
                f"active DOFs = {active_dofs}"
            )
            for name, n_dof in dof_report["dof_breakdown"].items():
                print(f"    block {name}: {n_dof} DOFs")

        while current_load < target_load:
            next_load_factor = min(current_load + d_load, target_load)
            print(f"Trying load factor: {next_load_factor:.4f} (Step size: {d_load:.4f})")
            
            p_scalar = dolfinx.fem.Function(V_phi)
            p_scalar.x.array[:] = p_ext_base.x.array[:] * next_load_factor

            u1, v1 = uyp_split[0], vzq_split[0]
            y, p_N = uyp_split[1], uyp_split[2]
            z, q_N = vzq_split[1], vzq_split[2]
            
            Pu1 = tensors(u1)
            Pv1 = ufl.derivative(Pu1, u1, v1)

            F = [0.0 for _ in range(len(spaces))]
            
            # dx_inner, dx_outer, dx_C = dx(TAG_INNER), dx(TAG_OUTER), dx(TAG_CORNER)
            # ds_inner, ds_outer, ds_C = ds(TAG_INNER), ds(TAG_OUTER), ds(TAG_CORNER)
            dx_inner, dx_outer = dx(TAG_INNER), dx(TAG_OUTER)
            ds_inner, ds_outer = ds(TAG_INNER), ds(TAG_OUTER)
            dS_N_stab = dS(TAG_STAB)

            F[0] += ufl.inner(Pu1, ufl.grad(v1)) * dx
            Gh1 = sigma_N * ufl.avg(h) * ufl.inner(ufl.jump(Pu1, n), ufl.jump(Pv1, n)) * dS_N_stab
            F[0] += gamma_u * ufl.inner(Pu1, Pv1) * (dx_inner + dx_outer) + Gh1
            F[0] += (ufl.inner(ufl.dot(y, n), v1) * (ds_inner + ds_outer) + gamma_u * ufl.inner(y, Pv1) * (dx_inner + dx_outer))
            # F[0] += -ufl.inner(ufl.dot(Pu1, n), v1) * ds_C
            # F[0] += sigma_N * h ** 2 * ufl.inner(ufl.div(Pu1), ufl.div(Pv1)) * dx_C

            # Normalize Phi toward a signed-distance function
            grad_phi_raw = ufl.grad(phi_func)
            # add 1e-12 to avoid division by zero
            norm_grad_phi = ufl.sqrt(ufl.inner(grad_phi_raw, grad_phi_raw)) + 1e-12 
            # norm_grad_phi = 1
            
            # Normalize the normal and distance field (make the vessel metric equivalent to the trapezoid)
            n_phi = grad_phi_raw / norm_grad_phi
            phi_d = phi_func / norm_grad_phi

            F_def = ufl.Identity(cell_dim) + ufl.grad(u1)
            J_def = ufl.det(F_def)
            F_inv_T = ufl.inv(F_def).T

            # Deformation-coupled traction (pull-back to the reference config): P.n = -p * J * F^{-T} * n
            # Sign note: if p is an outward-pushing internal pressure (positive) and n_phi points outward from the center
            # then the physical force points outward (same as n_phi). Flip the sign if your normal convention is the opposite.
            g_vec = -p_scalar * J_def * ufl.dot(F_inv_T, n_phi)

            F[1] += gamma_u * ufl.inner(Pu1, z) * dx_inner
            F[1] += (
                gamma_u * ufl.inner(y, z) * dx_inner
                + gamma_div * ufl.inner(ufl.div(y), ufl.div(z)) * dx_inner
                + gamma_p * h ** (-2) * ufl.inner(ufl.dot(y, n_phi), ufl.dot(z, n_phi)) * dx_inner
            )
            F[1] += gamma_p * h ** (-3) * ufl.inner(p_N * phi_d, ufl.dot(z, n_phi)) * dx_inner
            # Neumann load term: g_ext is already the physical force; take the inner product with n_phi
            F[1] += gamma_p * h ** (-2) * ufl.inner(g_vec, ufl.dot(z, n_phi)) * dx_inner

            F[1] += gamma_u * ufl.inner(Pu1, z) * dx_outer
            F[1] += (
                gamma_u * ufl.inner(y, z) * dx_outer
                + gamma_div * ufl.inner(ufl.div(y), ufl.div(z)) * dx_outer
                + gamma_p * h ** (-2) * ufl.inner(ufl.dot(y, n_phi), ufl.dot(z, n_phi)) * dx_outer
            )
            F[1] += gamma_p * h ** (-3) * ufl.inner(p_N * phi_d, ufl.dot(z, n_phi)) * dx_outer

            F[2] += gamma_p * h ** (-3) * ufl.inner(ufl.dot(y, n_phi), q_N * phi_d) * dx_inner
            F[2] += gamma_p * h ** (-4) * ufl.inner(p_N * phi_d, q_N * phi_d) * dx_inner
            F[2] += gamma_p * h ** (-3) * ufl.inner(g_vec, q_N * phi_d) * dx_inner 

            F[2] += gamma_p * h ** (-3) * ufl.inner(ufl.dot(y, n_phi), q_N * phi_d) * dx_outer
            F[2] += gamma_p * h ** (-4) * ufl.inner(p_N * phi_d, q_N * phi_d) * dx_outer

            J = [[ufl.derivative(F[k], uyp_split[j], duyp_split[j]) for j in range(len(uyp_split))] for k in range(len(F))]
            
            problem = NonLinearPhiFEM(F, J, tuple(uyp_split), bcs, restriction, spaces)
            problem._track_lu_factor = True
            problem.reset_timing()
            t_setup0 = time.perf_counter()
            F_vec = mphx.fem.petsc.create_vector_block(problem._F, restriction=restriction)
            J_mat = mphx.fem.petsc.create_matrix_block(problem._J, restriction=(restriction, restriction))
            if active_dofs_petsc is None:
                active_dofs_petsc = int(F_vec.getSize())
            prof_assembly_time += time.perf_counter() - t_setup0
            problem.reset_timing()

            snes = petsc4py.PETSc.SNES().create(mesh.comm)
            snes.setTolerances(max_it=50, rtol=1e-7, atol=1e-8)
            snes.setType("newtonls")
            snes.getKSP().setType("preonly")
            snes.getKSP().getPC().setType("lu")
            snes.getKSP().getPC().setFactorSolverType("mumps")
            
            # opts = PETSc.Options()
            # opts["snes_linesearch_type"] = "bt"
            # opts["snes_linesearch_maxstep"] = 0.5
            # opts["snes_max_it"] = 50
            # snes.setFromOptions()
            
            snes.setObjective(problem.obj)
            snes.setFunction(problem.F, F_vec)
            snes.setJacobian(problem.J, J=J_mat, P=None)
            
            solution = problem.create_snes_solution()
            problem.reset_timing()
            t_snes0 = time.perf_counter()
            try:
                snes.solve(None, solution)
                converged_reason = snes.getConvergedReason()
            except Exception as e:
                converged_reason = -1
                print(f"Solver crashed: {e}")
            t_snes = time.perf_counter() - t_snes0

            try:
                prof_newton_iters += int(snes.getIterationNumber())
            except Exception:
                pass
            step_assembly = (
                problem._timing["assembly_F"] + problem._timing["assembly_J"]
            )
            step_factor = problem._timing["lu_factor"]
            prof_assembly_time += step_assembly
            prof_factorization_time += step_factor
            prof_linear_solve_time += max(0.0, t_snes - step_assembly - step_factor)

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
        
        t_io0 = time.perf_counter()
        self._last_solve_profile = {
            "newton_iters": prof_newton_iters,
            "assembly_time": prof_assembly_time,
            "factorization_time": prof_factorization_time,
            "linear_solve_time": prof_linear_solve_time,
            "io_time": prof_io_time,
            "active_cells": mesh_report["active_cells"],
            "active_nodes": mesh_report["active_nodes"],
            "active_dofs": active_dofs,
            "dof_breakdown": dof_report["dof_breakdown"],
            "active_dofs_petsc": active_dofs_petsc,
        }

        if not solved_successfully:
            prof_io_time += time.perf_counter() - t_io0
            self._last_solve_profile["io_time"] = prof_io_time
            _print_solve_profile(
                f"solve_one {self.index} (failed)",
                mesh.comm,
                active_dofs,
                self._last_solve_profile,
            )
            print(f"Sample {self.index} failed to reach full load.")
            return None, None, None, None, None

        def extract_q9_matrix(expr, V_source, V_target_macro, shape_dim):
            expr.x.scatter_forward()
            func_macro = dolfinx.fem.Function(V_target_macro)
            
            # Source and target share triangle topology
            nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                func_macro.function_space.mesh, func_macro.function_space.element, V_source.mesh, padding=self.padding
            )
            func_macro.interpolate(expr, nmm_interpolation_data=nmm_data)
            func_macro.x.scatter_forward()
            
            res_sorted = func_macro.x.array.reshape((-1, shape_dim))[self.q9_indices]
            mat = res_sorted.reshape((self.N_q9, self.N_q9, shape_dim))
            return np.transpose(mat, (2, 1, 0))

        u_q9_mat = extract_q9_matrix(uyp_split[0], V, self.V_macro_vector, 2)
        y_q9_mat = extract_q9_matrix(uyp_split[1], Z_N, self.V_macro_tensor, 4)
        p_q9_mat = extract_q9_matrix(uyp_split[2], Q_N, self.V_macro_scalar, 1)

        g_macro_cg1 = self.matrix_to_function(g_sample, self.V_macro_cg1_scalar)
        g_q9 = dolfinx.fem.Function(self.V_macro_cg2_scalar)
        g_q9.interpolate(g_macro_cg1)
        
        # Sort and reshape to (2, 127, 127)
        g_q9_sorted = g_q9.x.array.reshape((-1, 1))[self.q9_indices]
        g_mat = g_q9_sorted.reshape((self.N_q9, self.N_q9, 1))
        g_q9_mat = np.transpose(g_mat, (2, 1, 0))

        phi_fine = dolfinx.fem.Function(self.V_macro_scalar)
        phi_fine.interpolate(Phi_full.eval)
        phi_fine.x.scatter_forward()
        phi_sorted = phi_fine.x.array.reshape((-1, 1))[self.q9_indices]
        phi_q9_mat = np.transpose(phi_sorted.reshape((self.N_q9, self.N_q9, 1)), (2, 1, 0))

        # g_q9_mat[:, phi_q9_mat[0] > 1e-8] = 0.0

        prof_io_time += time.perf_counter() - t_io0
        self._last_solve_profile["io_time"] = prof_io_time
        _print_solve_profile(
            f"solve_one {self.index}",
            mesh.comm,
            active_dofs,
            self._last_solve_profile,
        )

        return u_q9_mat, y_q9_mat, p_q9_mat, phi_q9_mat, g_q9_mat

    def solve_several(self):
        U_list = []
        Y_list = []
        P_list = []
        S_list = []
        Phi_list = []
        G_list = []
        nb = len(self.params)
        agg = {
            "assembly_time": 0.0,
            "factorization_time": 0.0,
            "linear_solve_time": 0.0,
            "io_time": 0.0,
            "newton_iters": 0,
            "active_cells": 0,
            "active_nodes": 0,
            "active_dofs": 0,
            "n_ok": 0,
        }
        comm = MPI.COMM_WORLD
        _print_parallel_runtime(comm, header="generate_data_Vessel parallel / CPU info")
        for i in range(nb):
            if comm.rank == 0:
                print(f"Data : {i}/{nb}")
            start_time = time.perf_counter()
            u_mat, y_mat, p_mat, phi_mat, g_mat = self.solve_one(i)
            wall_time = time.perf_counter() - start_time
            if u_mat is None:
                continue
            U_list.append(u_mat)
            Y_list.append(y_mat)
            P_list.append(p_mat)
            G_list.append(g_mat)
            Phi_list.append(phi_mat)
            prof = getattr(self, "_last_solve_profile", {})
            agg["n_ok"] += 1
            for key in ("assembly_time", "factorization_time", "linear_solve_time", "io_time", "newton_iters"):
                agg[key] += float(prof.get(key, 0.0))
            for key in ("active_cells", "active_nodes", "active_dofs"):
                if prof.get(key) is not None:
                    agg[key] = int(prof[key])
            if comm.rank == 0:
                print(f"  wall-clock (sample {i}): {wall_time:.4f} s")
            # if len(U_list) == 600:
            #     break

        if not U_list:
            raise RuntimeError("solve_several: no successful samples.")

        if comm.rank == 0 and agg["n_ok"] > 0:
            n_ok = agg["n_ok"]
            summary = {
                "assembly_time": agg["assembly_time"] / n_ok,
                "factorization_time": agg["factorization_time"] / n_ok,
                "linear_solve_time": agg["linear_solve_time"] / n_ok,
                "io_time": agg["io_time"] / n_ok,
                "newton_iters": int(round(agg["newton_iters"] / n_ok)),
                "active_cells": agg["active_cells"],
                "active_nodes": agg["active_nodes"],
                "active_dofs": agg["active_dofs"],
            }
            print(f"\n=== solve_several average over {n_ok} samples ===")
            _print_solve_profile("batch mean", comm, summary["active_dofs"], summary)

        return (
            np.stack(U_list),
            np.stack(Y_list),
            np.stack(P_list),
            np.stack(Phi_list),
            np.stack(G_list),
        )

def create_parameters(nb_vert=51, nb_training_shapes=300):
    """
    Generate pressure-vessel inputs (in memory only); return phi / gradient fields / G / geometry params from create_FG_numpy.
    """
    nb_data = nb_training_shapes
    _, phi, E_dx, E_dy, G, params = create_FG_numpy(
        nb_data=nb_data, nb_vert=nb_vert * 2 - 1
    )
    return params, phi, E_dx, E_dy, G

def go(
    params,
    phi,
    E_dx,
    E_dy,
    G,
    nb_vert=51,
    n_train=400,
    n_test=100,
    nb_incr=3
):
    """
    Solve and write to NOWS/data; save only the train and test npz files.
    Convention: **H = y direction, W = x direction**, both (B, H, W, C).
    In batch order: first n_train is train, then n_test is test.

    - inputs: after meshgrid(xx,yy,'xy') and permute, phi is (B, Ny, Nx, ·), i.e. dim1=y, dim2=x;
      After transpose, G is (B, Ny, Nx, 1). Concatenation gives C_in=7.

    - targets: extract_q9_matrix in solve_one yields (C, W_x, H_y) per sample (mat is y×x then
      transposed to C,j,i), stacked as (B, C, W, H); to get (B, H_y, W_x, C) use
      transpose(0, 3, 2, 1); do not use (0, 2, 3, 1) (that would give B, W, H, C).
    """
    save = True
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(base_dir), "data")

    ti0 = time.perf_counter()
    solver = PhiFemSolver(
        nb_cell=nb_vert - 1, params=params, G_data=G[:, :, ::2, ::2], nb_incr=nb_incr
    )
    U, Y, P, _, _ = solver.solve_several()
    t_after_solve = time.perf_counter()

    # After stack: (B, C, W_x, H_y); convert to (B, H_y, W_x, C) i.e. B,H,W,C with H=y, W=x
    U = U.transpose((0, 3, 2, 1))
    Y = Y.transpose((0, 3, 2, 1))
    P = P.transpose((0, 3, 2, 1))

    # In create_FG_numpy, G is (N, 1, Nx, Ny); align to (N, Ny, Nx, 1) with dim1=y, dim2=x
    G = G.transpose((0, 3, 2, 1))
    model_inputs = np.concatenate([phi, E_dx, E_dy, G], axis=-1)
    model_targets = U

    t_io_disk = 0.0
    if save:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        batch_num = model_inputs.shape[0]
        s_grid = nb_vert * 2 - 1
        need = int(n_train) + int(n_test)
        if batch_num < need:
            print(
                f"Warning: batch size {batch_num} < train+test need {need}, "
                "train/test will be truncated to the available count."
            )
        n_tr = min(int(n_train), batch_num)
        rest = batch_num - n_tr
        n_te = min(int(n_test), rest)
        if n_tr > 0:
            path_train = os.path.join(
                data_dir,
                f"Hyperelasticity_Vessel_PhiEdxEdyG_u_s{s_grid}_n{n_tr}_train.npz",
            )
            t0 = time.perf_counter()
            np.savez_compressed(
                path_train,
                inputs=model_inputs[:n_tr],
                targets=model_targets[:n_tr],
            )
            t_io_disk += time.perf_counter() - t0
            print(f"saved: {path_train}")
        path_test = os.path.join(
            data_dir,
            f"Hyperelasticity_Vessel_PhiEdxEdyG_u_s{s_grid}_n{n_te}_test.npz",
        )
        # np.savez_compressed(
        #     path_test,
        #     inputs=model_inputs[n_tr : n_tr + n_te],
        #     targets=model_targets[n_tr : n_tr + n_te],
        # )
        # 
        print(f"saved: {path_test}")

    duration = time.perf_counter() - ti0
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(f"\n=== go() total wall time: {duration:.4f} s ===")
        print(f"  disk I/O time (np.savez): {t_io_disk:.4f} s")
        print(f"  solve wall time (excl. disk): {t_after_solve - ti0:.4f} s")

if __name__ == "__main__":
    nb_vert = 51
    # 500 = 400 train + 100 test
    params, phi, E_dx, E_dy, G = create_parameters(
        nb_vert=nb_vert, nb_training_shapes=1100
    )
    go(
        params,
        phi,
        E_dx,
        E_dy,
        G,
        nb_vert=nb_vert,
        n_train=1000,
        n_test=100,
        nb_incr=1
    )
    # go_val()
    # go_test()
    # pass
