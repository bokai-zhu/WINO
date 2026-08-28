#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the Fourier Neural Operator
Adapted from https://github.com/neuraloperator/neuraloperator/blob/master/utilities3.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import operator
from functools import reduce
from timeit import default_timer
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c

def get_cell_masks(phi_h):
    """
    Build cell masks from nodal level-set values phi_h.
    phi_h shape: (B, H, W, 1)
    """
    # Phi at the four corners of each (H-1) x (W-1) cell
    phi_00 = phi_h[:, :-1, :-1, :]  # (i, j)
    phi_10 = phi_h[:, :-1, 1:, :]  # (i+1, j)
    phi_01 = phi_h[:, 1:, :-1, :]  # (i, j+1)
    phi_11 = phi_h[:, 1:, 1:, :]  # (i+1, j+1)

    # Min / max phi in each cell
    min_phi_in_cell = torch.minimum(torch.minimum(phi_00, phi_10), torch.minimum(phi_01, phi_11))
    max_phi_in_cell = torch.maximum(torch.maximum(phi_00, phi_10), torch.maximum(phi_01, phi_11))

    # Omega_h (T_h): cell T intersects {phi_h < 0}
    # (at least one node of the cell has phi < 0)
    mask_Cell_Omega = (min_phi_in_cell < -1e-10).double()

    # Cut Cells (T_h^Gamma): cell T intersects {phi_h = 0}
    # (the cell has both positive and negative nodal phi)
    mask_Cut_Cell = (min_phi_in_cell < -1e-10) & (max_phi_in_cell >= -1e-10)
    mask_Cut_Cell = mask_Cut_Cell.double()

    # Both masks have shape (B, H-1, W-1, 1)
    return mask_Cell_Omega, mask_Cut_Cell

def get_face_masks(mask_Cut_Cell, mask_Cell_Omega):
    """
    Facet masks between cut cells T_h^Gamma and non-cut material cells in Omega_h.
    Edges between a cut cell and a neighbouring bulk cell (mask_Cell_Omega=1, not cut).

    mask_Cut_Cell, mask_Cell_Omega: (B, H-1, W-1, 1)

    Returns:
    mask_Face_V: vertical faces (B, H-1, W-2, 1)
    mask_Face_H: horizontal faces (B, H-2, W-1, 1)
    """
    mask_bulk = mask_Cell_Omega * (1.0 - mask_Cut_Cell)
    # Vertical face E_{i+0.5,j}: one side cut, the other bulk Omega
    mask_Face_V = (
        mask_Cut_Cell[:, :, :-1, :] * mask_bulk[:, :, 1:, :]
        + mask_Cut_Cell[:, :, 1:, :] * mask_bulk[:, :, :-1, :]
    )
    # Horizontal face E_{i, j+0.5}
    mask_Face_H = (
        mask_Cut_Cell[:, :-1, :, :] * mask_bulk[:, 1:, :, :]
        + mask_Cut_Cell[:, 1:, :, :] * mask_bulk[:, :-1, :, :]
    )
    return mask_Face_V, mask_Face_H

def get_boundary_masks(mask_Cell_Omega):
    """
    Boundary masks including the outer boundary. +1: up/right normal, -1: down/left normal.
    Args:
    mask_Cell_Omega: (B, H-1, W-1, 1)
    Returns:
    mask_Bound_V: vertical interior-face mask (B, H-1, W-2, 1)
    mask_Bound_H: horizontal interior-face mask (B, H-2, W-1, 1)
    """
    # vertical (B, H-1, W, 1)
    mask_Bound_V = mask_Cell_Omega[:, :, :-1, :] - mask_Cell_Omega[:, :, 1:, :]

    # horizontal (B, H-2, W-1, 1)
    mask_Bound_H = mask_Cell_Omega[:, :-1, :, :] - mask_Cell_Omega[:, 1:, :, :]

    return mask_Bound_V, mask_Bound_H

def get_node_masks(phi_h):
    """
    Build a nodal mask (PyTorch).

    phi_h shape: (B, H, W, 1)
    Returns node_mask of shape (B, H, W, 1).
    """

    # 1. Binary condition mask
    condition_mask = (phi_h < -1e-10).double()

    # 2. Convert to (B, C, H, W) for pooling
    # (B, H, W, 1) -> (B, 1, H, W)
    condition_mask_chw = condition_mask.permute(0, 3, 1, 2)

    # 3. 3x3 max pooling
    # kernel_size=3, stride=1, padding=1 (same as JAX 'SAME')
    node_mask_chw = F.max_pool2d(
        condition_mask_chw,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # 4. Convert back to (B, H, W, 1)
    node_mask = node_mask_chw.permute(0, 2, 3, 1)

    return node_mask == 1

def get_node_mask_from_cell_mask(mask_cut_cell):
    """
    Args: mask_cut_cell, shape [B, H-1, W-1, 1] (cell mask)
    Returns: mask_cut_node, shape [B, H, W, 1] (nodal mask)
    """
    # 1. Zero nodal mask, one node larger than the cell mask
    # Example input shape [1, 63, 63, 1]
    B, H, W, C = mask_cut_cell.shape
    mask_nodes = torch.zeros((B, H + 1, W + 1, C), 
                             device=mask_cut_cell.device, 
                             dtype=mask_cut_cell.dtype)
    
    # 2. Scatter each cell mask onto its four nodes
    # Element[i, j] contributes to Node[i, j], [i, j+1], [i+1, j], [i+1, j+1]
    
    # top-left node
    mask_nodes[:, 0:-1, 0:-1, :] += mask_cut_cell
    # top-right node
    mask_nodes[:, 0:-1, 1:  , :] += mask_cut_cell
    # bottom-left node
    mask_nodes[:, 1:  , 0:-1, :] += mask_cut_cell
    # bottom-right node
    mask_nodes[:, 1:  , 1:  , :] += mask_cut_cell
    
    # 3. Positive entries belong to at least one cut cell
    # Binarize to a 0/1 mask:
    mask_nodes = (mask_nodes > 0).double()
    
    return mask_nodes

# First Piola-Kirchhoff stress P for Neo-Hookean, shape (..., 2, 2)
def get_P(dudx, dudy, model_data):
    # Numerical safeguards (hard-coded, not from model_data)
    # softplus keeps J >= DET_EPS so log(J) is defined.
    DET_EPS = 1e-6
    DET_BETA = 30.0  # used by the softplus map
    INV_EPS = 1e-6  # regularize F before inversion

    F = torch.stack([
        torch.stack([dudx[..., 0] + 1, dudy[..., 0]], dim=-1),
        torch.stack([dudx[..., 1], dudy[..., 1] + 1], dim=-1)
    ], dim=-2)
    # F: (batch, H, W, 2, 2)
    detF_raw = torch.det(F).unsqueeze(-1).unsqueeze(-1)
    # J_safe = eps + softplus(beta * J_raw) / beta  (>= eps), smooth
    detF = DET_EPS + torch.nn.functional.softplus(DET_BETA * detF_raw) / DET_BETA

    I2 = torch.eye(2, dtype=F.dtype, device=F.device).view(1, 1, 1, 2, 2)
    F_reg = F + INV_EPS * I2
    F_inv_T = torch.inverse(F_reg).transpose(-2, -1)
    # P = mu * (F - F^{-T}) + lambda * ln(J) * F^{-T}
    P = model_data["mu"] * (F - F_inv_T) + model_data["lambda"] * torch.log(detF) * F_inv_T
    return P

def get_Pv(dudx, dudy, dvdx, dvdy, vi, model_data):
    """
    Linearized increment of P: D_u(P(F))[v].
    Directional derivative of P at F along grad_v.

    Args:
        dudx, dudy:  (..., 2)        gradients of u (du/dx, du/dy)
        dvdx, dvdy:  (..., 1)        gradients of scalar v (dv/dx, dv/dy)
        vi        : 0 or 1; 0 -> vector field (v, 0), 1 -> (0, v)
        model_data: must contain 'mu' and 'lambda'

    Returns:
        delta_P: linearized stress increment (..., 2, 2)
    """
    mu = model_data['mu']
    lmbda = model_data['lambda']

    F = torch.stack([
        torch.stack([dudx[..., 0] + 1.0, dudy[..., 0]], dim=-1),
        torch.stack([dudx[..., 1],        dudy[..., 1] + 1.0], dim=-1)
    ], dim=-2)

    # Build vector field v = (v, 0) or (0, v) from vi
    # dvdx, dvdy have shape (..., 1); squeeze then scatter into two components
    dvx = dvdx[..., 0]
    dvy = dvdy[..., 0]
    zero = torch.zeros_like(dvx)
    if vi == 0:
        dvdx_vec = torch.stack([dvx, zero], dim=-1)
        dvdy_vec = torch.stack([dvy, zero], dim=-1)
    else:
        dvdx_vec = torch.stack([zero, dvx], dim=-1)
        dvdy_vec = torch.stack([zero, dvy], dim=-1)

    grad_v = torch.stack([
        torch.stack([dvdx_vec[..., 0], dvdy_vec[..., 0]], dim=-1),
        torch.stack([dvdx_vec[..., 1], dvdy_vec[..., 1]], dim=-1)
    ], dim=-2)

    # 1. Kinematic quantities
    # F^{-T} (inverse transpose)
    # torch.inverse is batched
    F_inv = torch.inverse(F)
    F_inv_T = F_inv.transpose(-2, -1)  # F^{-T}

    detF = torch.det(F).unsqueeze(-1).unsqueeze(-1)
    lnJ = torch.log(torch.clamp(detF, min=1e-6))

    # 2. Variations of each term

    # term 1: delta(F) = grad_v
    delta_F = grad_v

    # term 2: delta(ln J) = F^{-T} : delta_F (double contraction)
    # sum(A * B) is the Frobenius inner product
    delta_lnJ = torch.sum(F_inv_T * delta_F, dim=(-2, -1), keepdim=True)

    # term 3: delta(F^{-T}) = - F^{-T} * (delta_F)^T * F^{-T}
    # Product order: - F^{-T} @ grad_v.T @ F^{-T}
    grad_v_T = grad_v.transpose(-2, -1)
    # @ is batched matrix multiply
    delta_F_inv_T = -1.0 * (F_inv_T @ grad_v_T @ F_inv_T)

    # 3. Assemble delta_P
    # P = mu*F + (lambda*lnJ - mu)*F^{-T}
    # delta_P = mu*delta_F + lambda*delta_lnJ*F^{-T} + (lambda*lnJ - mu)*delta_F^{-T}

    term_1 = mu * delta_F
    term_2 = lmbda * delta_lnJ * F_inv_T
    term_3 = (lmbda * lnJ - mu) * delta_F_inv_T

    delta_P = term_1 + term_2 + term_3

    return delta_P

def get_Pv_batched(dudx, dudy, dvdx, dvdy, model_data):
    """
    Factor F^{-T} once for a (dudx, dudy) field and evaluate vi=0/1 at all shape-function nodes.
    dvdx, dvdy: (ngy, ngx, Nnode, 1, 1, 1)
    Returns Pv_vi0, Pv_vi1 with shape (ngy, ngx, Nnode, *spatial, 2, 2).
    """
    mu = model_data['mu']
    lmbda = model_data['lambda']
    nnode = dvdx.shape[2]

    F = torch.stack([
        torch.stack([dudx[..., 0] + 1.0, dudy[..., 0]], dim=-1),
        torch.stack([dudx[..., 1], dudy[..., 1] + 1.0], dim=-1)
    ], dim=-2)
    F_inv_T = torch.inverse(F).transpose(-2, -1)
    lnJ = torch.log(torch.clamp(torch.det(F).unsqueeze(-1).unsqueeze(-1), min=1e-6))

    dvx = dvdx[:, :, :nnode, 0, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    dvy = dvdy[:, :, :nnode, 0, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    zero = torch.zeros_like(dvx)

    grad_v0 = torch.stack([
        torch.stack([dvx, dvy], dim=-1),
        torch.stack([zero, zero], dim=-1),
    ], dim=-2)
    grad_v1 = torch.stack([
        torch.stack([zero, zero], dim=-1),
        torch.stack([dvx, dvy], dim=-1),
    ], dim=-2)

    F_inv_T_n = F_inv_T.unsqueeze(2)
    lnJ_n = lnJ.unsqueeze(2)

    def _delta_P(grad_v):
        delta_lnJ = torch.sum(F_inv_T_n * grad_v, dim=(-2, -1), keepdim=True)
        delta_F_inv_T = -(F_inv_T_n @ grad_v.transpose(-2, -1) @ F_inv_T_n)
        return (
            mu * grad_v
            + lmbda * delta_lnJ * F_inv_T_n
            + (lmbda * lnJ_n - mu) * delta_F_inv_T
        )

    return _delta_P(grad_v0), _delta_P(grad_v1)

def add_Gh_vform_to_res_v(
    res_v, U, V, mask_Face_V, mask_Face_H, xgrid, w1d, dx, dy, model_data, *,
    h_face_yx=True,
):
    """
    G_h(u,v) = h * int [P(F) n] · [D_u P[v] n], accumulated into res_v.
    h_face_yx: True uses meshgrid(-1, xgrid) on horizontal faces; False uses meshgrid(xgrid, -1).
    """
    sigma_N = model_data["sigma_N"]
    ones = torch.ones_like(xgrid[0])
    neg_ones = -ones

    # Vertical faces (n = ±x)
    W1d_v = w1d.view(3, 1, 1, 1, 1, 1)
    mV = mask_Face_V[..., 0]
    Y_l, X_l = torch.meshgrid(xgrid, neg_ones, indexing='ij')
    Y_r, X_r = torch.meshgrid(xgrid, ones, indexing='ij')
    dudx_r, dudy_r = d_shape_functions_Q4(U[:, :, :, :, :-1, :], Y_r, X_r, dx, dy)
    dudx_l, dudy_l = d_shape_functions_Q4(U[:, :, :, :, 1:, :], Y_l, X_l, dx, dy)
    dvdx_r, dvdy_r = d_shape_functions_Q4(V[:, :, 1::2], Y_r, X_r, dx, dy)
    dvdx_l, dvdy_l = d_shape_functions_Q4(V[:, :, ::2], Y_l, X_l, dx, dy)
    jump_v = get_P(dudx_r, dudy_r, model_data)[..., 0] - get_P(dudx_l, dudy_l, model_data)[..., 0]
    Pv0_r, Pv1_r = get_Pv_batched(dudx_r, dudy_r, dvdx_r, dvdy_r, model_data)
    Pv0_l, Pv1_l = get_Pv_batched(dudx_l, dudy_l, dvdx_l, dvdy_l, model_data)
    scale_v = mV * sigma_N * dy ** 2 / 2
    # inner*: (ngy, ngx, Nnode, b, Hf, Wf)
    inner0 = (jump_v.unsqueeze(2) * (Pv0_r[..., 0] - Pv0_l[..., 0])).sum(dim=-1)
    inner1 = (jump_v.unsqueeze(2) * (Pv1_r[..., 0] - Pv1_l[..., 0])).sum(dim=-1)
    c0 = (inner0 * W1d_v).sum(dim=(0, 1))
    c1 = (inner1 * W1d_v).sum(dim=(0, 1))
    res_v[:, 0:-1, 1:-1, 0] += scale_v * c0[0]
    res_v[:, 0:-1, 1:-1, 1] += scale_v * c1[0]
    res_v[:, 1:, 1:-1, 0] += scale_v * c0[1]
    res_v[:, 1:, 1:-1, 1] += scale_v * c1[1]

    # Horizontal faces (n = ±y)
    W1d_h = w1d.view(1, 3, 1, 1, 1, 1)
    mH = mask_Face_H[..., 0]
    if h_face_yx:
        Y_d, X_d = torch.meshgrid(neg_ones, xgrid, indexing='ij')
        Y_u, X_u = torch.meshgrid(ones, xgrid, indexing='ij')
    else:
        X_d, Y_d = torch.meshgrid(xgrid, neg_ones, indexing='ij')
        X_u, Y_u = torch.meshgrid(xgrid, ones, indexing='ij')
    dudx_d, dudy_d = d_shape_functions_Q4(U[:, :, :, 1:, :], Y_d, X_d, dx, dy)
    dudx_u, dudy_u = d_shape_functions_Q4(U[:, :, :, :-1, :], Y_u, X_u, dx, dy)
    dvdx_d, dvdy_d = d_shape_functions_Q4(V[:, :, 0:2], Y_d, X_d, dx, dy)
    dvdx_u, dvdy_u = d_shape_functions_Q4(V[:, :, 2:4], Y_u, X_u, dx, dy)
    jump_h = get_P(dudx_d, dudy_d, model_data)[..., 1] - get_P(dudx_u, dudy_u, model_data)[..., 1]
    Pv0_d, Pv1_d = get_Pv_batched(dudx_d, dudy_d, dvdx_d, dvdy_d, model_data)
    Pv0_u, Pv1_u = get_Pv_batched(dudx_u, dudy_u, dvdx_u, dvdy_u, model_data)
    scale_h = mH * sigma_N * dx ** 2 / 2
    inner0 = (jump_h.unsqueeze(2) * (Pv0_d[..., 1] - Pv0_u[..., 1])).sum(dim=-1)
    inner1 = (jump_h.unsqueeze(2) * (Pv1_d[..., 1] - Pv1_u[..., 1])).sum(dim=-1)
    c0 = (inner0 * W1d_h).sum(dim=(0, 1))
    c1 = (inner1 * W1d_h).sum(dim=(0, 1))
    res_v[:, 1:-1, 0:-1, 0] += scale_h * c0[0]
    res_v[:, 1:-1, 0:-1, 1] += scale_h * c1[0]
    res_v[:, 1:-1, 1:, 0] += scale_h * c0[1]
    res_v[:, 1:-1, 1:, 1] += scale_h * c1[1]

def d_shape_functions_Q9(u, y, x, dx, dy):
    """
    u: (3, 3, b, h, w, c)
    y, x: (y, x)
    Returns: (y, x, b, h, w, c)
    """
    y = y.view(y.shape[0], y.shape[1], 1, 1, 1, 1)    # （y,x,1,1,1,1)
    x = x.view(x.shape[0], x.shape[1], 1, 1, 1, 1)
    u = u.unsqueeze(2).unsqueeze(2)   # (3,3,1,1,b,h,w,2)
    dudx = (2*(u[1,1]*(x - 1)*(y - 1)*(y + 1) - (u[1,0]*(x - 1)*(y - 1)*(y + 1))/2 + u[1,1]*(x + 1)*(y - 1)*(y + 1)\
            - (u[1,2]*(x + 1)*(y - 1)*(y + 1))/2 + (u[0,0]*x*y*(y - 1))/4 + (u[0,2]*x*y*(y - 1))/4 + (u[2,0]*x*y*(y + 1))/4\
            + (u[2,2]*x*y*(y + 1))/4 + (u[0,0]*y*(x - 1)*(y - 1))/4 - (u[0,1]*y*(x - 1)*(y - 1))/2 - (u[0,1]*y*(x + 1)*(y - 1))/2\
            + (u[0,2]*y*(x + 1)*(y - 1))/4 - (u[1,0]*x*(y - 1)*(y + 1))/2 - (u[1,2]*x*(y - 1)*(y + 1))/2\
            + (u[2,0]*y*(x - 1)*(y + 1))/4 - (u[2,1]*y*(x - 1)*(y + 1))/2 - (u[2,1]*y*(x + 1)*(y + 1))/2\
            + (u[2,2]*y*(x + 1)*(y + 1))/4))/dx

    dudy = (2*(u[1,1]*(x - 1)*(x + 1)*(y - 1) - (u[0,1]*(x - 1)*(x + 1)*(y - 1))/2 + u[1,1]*(x - 1)*(x + 1)*(y + 1)\
            - (u[2,1]*(x - 1)*(x + 1)*(y + 1))/2 + (u[0,0]*x*y*(x - 1))/4 + (u[0,2]*x*y*(x + 1))/4 + (u[2,0]*x*y*(x - 1))/4\
            + (u[2,2]*x*y*(x + 1))/4 + (u[0,0]*x*(x - 1)*(y - 1))/4 - (u[0,1]*y*(x - 1)*(x + 1))/2 + (u[0,2]*x*(x + 1)*(y - 1))/4\
            - (u[1,0]*x*(x - 1)*(y - 1))/2 - (u[1,0]*x*(x - 1)*(y + 1))/2 - (u[1,2]*x*(x + 1)*(y - 1))/2\
            - (u[1,2]*x*(x + 1)*(y + 1))/2 + (u[2,0]*x*(x - 1)*(y + 1))/4 - (u[2,1]*y*(x - 1)*(x + 1))/2\
            + (u[2,2]*x*(x + 1)*(y + 1))/4))/dy
    return dudx, dudy

def shape_functions(x, y):
    """
    x, y: (...,)
    return: N (3,3,...)
    """
    Lx = torch.stack([
        0.5 * x * (x - 1),
        1 - x**2,
        0.5 * x * (x + 1)
    ], dim=0)

    Ly = torch.stack([
        0.5 * y * (y - 1),
        1 - y**2,
        0.5 * y * (y + 1)
    ], dim=0)

    return Lx[:, None] * Ly[None, :]

def shape_functions_Q9(u, y, x):
    """
    u: (3,3,b,h,w,d)
    x, y: (y,x) in [-1,1]
    return: (y,x,b,h,w,d)
    """
    x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
    y = y.view(y.shape[0], y.shape[1], 1, 1, 1)

    N = shape_functions(x, y)          # (3,3,y,x,1,1,1)
    N = N.unsqueeze(-1)                # (...,1)

    u = u.unsqueeze(2).unsqueeze(2)    # (3,3,1,1,b,h,w,d)

    u_xy = (N * u).sum(dim=(0,1))      # (y,x,b,h,w,d)

    return u_xy

def d_shape_functions_Q4(u, y, x, dx, dy):
    """
    u: (2,2,b,h,w,2)
    x, y: (y,x) in [-1,1]
    return: dudx, dudy -> (y,x,b,h,w,2)
    """
    y = y.view(y.shape[0], y.shape[1], 1, 1, 1, 1)
    x = x.view(x.shape[0], x.shape[1], 1, 1, 1, 1)

    u = u.unsqueeze(0).unsqueeze(0)    # (1,1,2,2,b,h,w,2)

    # dN/dx
    dNdx = torch.stack([
        torch.stack([
            -0.25 * (1 - y),   0.25 * (1 - y)
        ], dim=2),
        torch.stack([
            -0.25 * (1 + y),   0.25 * (1 + y)
        ], dim=2)
    ], dim=2)   # (y,x,2,2,1,1,1,1)

    # dN/dy
    dNdy = torch.stack([
        torch.stack([
            -0.25 * (1 - x),  -0.25 * (1 + x)
        ], dim=2),
        torch.stack([
             0.25 * (1 - x),   0.25 * (1 + x)
        ], dim=2)
    ], dim=2)

    dudx = (dNdx * u).sum(dim=(2,3)) / (dx/2)
    dudy = (dNdy * u).sum(dim=(2,3)) / (dy/2)

    return dudx, dudy

def shape_functions_Q4(u, y, x):
    """
    u: (2,2,b,h,w,2)
    x, y: (y,x) in [-1,1]
    return: u_interp -> (y,x,b,h,w,2)
    """
    y = y.view(y.shape[0], y.shape[1], 1, 1, 1, 1)
    x = x.view(x.shape[0], x.shape[1], 1, 1, 1, 1)

    u = u.unsqueeze(0).unsqueeze(0)    # (1,1,2,2,b,h,w,2)

    # shape functions N
    N = torch.stack([
        torch.stack([
            0.25 * (1 - x) * (1 - y),   # N00
            0.25 * (1 + x) * (1 - y)    # N10
        ], dim=2),
        torch.stack([
            0.25 * (1 - x) * (1 + y),   # N01
            0.25 * (1 + x) * (1 + y)    # N11
        ], dim=2)
    ], dim=2)   # (y,x,2,2,1,1,1,1)

    u_interp = (N * u).sum(dim=(2,3))

    return u_interp

def stack_Q4_data(tensor):
    """
    Reshape (b, H, W, C) to (2, 2, b, H-1, W-1, C) for Q4 quadrature.
    """
    t00 = tensor[:, 0:-1, 0:-1, :]
    t10 = tensor[:, 1:  , 0:-1, :]
    t11 = tensor[:, 1:  , 1:  , :]
    t01 = tensor[:, 0:-1, 1:  , :]
    return torch.stack([
        torch.stack([t00, t01], dim=0),
        torch.stack([t10, t11], dim=0)
    ], dim=0)

def stack_Q9_data(tensor):
    """
    Reshape (b, H, W, C) to (3, 3, b, H-1, W-1, C) for Q9 quadrature.
    """
    t00 = tensor[:, 0:-2:2, 0:-2:2, :]  # bottom-left, (batch, H-1, W-1, 2)
    t20 = tensor[:, 2::2, 0:-2:2, :]  # top-left
    t22 = tensor[:, 2::2, 2::2, :]  # top-right
    t02 = tensor[:, 0:-2:2, 2::2, :]  # bottom-right
    t10 = tensor[:, 1:-1:2, 0:-2:2, :]  # left mid-side
    t21 = tensor[:, 2::2, 1:-1:2, :]  # top mid-side
    t12 = tensor[:, 1:-1:2, 2::2, :]  # right mid-side
    t01 = tensor[:, 0:-2:2, 1:-1:2, :]  # bottom mid-side
    t11 = tensor[:, 1:-1:2, 1:-1:2, :]  # center
    T = torch.stack([
        torch.stack([t00, t01, t02], dim=0),
        torch.stack([t10, t11, t12], dim=0),
        torch.stack([t20, t21, t22], dim=0)
    ], dim=0)   # (3, 3, b, H-1, W-1, C)
    return T

def get_mask_cut_g(mask_Cell_Omega):
    """
    Find inner-side cut cells by a shift: material cells whose left or bottom neighbour is empty.
    
    Args:
    mask_Cell_Omega: array (b, h-1, w-1, 1), 1 = material, 0 = void
    
    Returns:
    mask_Cut_g: array (b, h-1, w-1, 1)
    """
    # 1. Drop the channel axis -> (b, H, W)
    m = mask_Cell_Omega[..., 0]
    
    # 2. Left neighbour by shifting right by one cell
    left_neighbor = torch.ones_like(m)
    left_neighbor[:, :, 1:] = m[:, :, :-1]
    
    # 3. Bottom neighbour by shifting up by one cell
    bottom_neighbor = torch.ones_like(m)
    bottom_neighbor[:, 1:, :] = m[:, :-1, :]

    lb_neighbor = torch.ones_like(m)
    lb_neighbor[:, 1:, 1:] = m[:, :-1, :-1]
    
    # 4. Vectorized predicate
    # Cell is material (1) and left or bottom neighbour is empty (0)
    mask_Cut_g = (m == 1) & ((left_neighbor == 0) | (bottom_neighbor == 0) | (lb_neighbor == 0))
    
    # 5. Restore shape (b, h-1, w-1, 1)
    return mask_Cut_g.unsqueeze(-1).double()

def train_rvino_hyperelasticity_Guass(model, model_data, train_loader, test_loader,
                                      loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    model.eval()
    init_loss_sums = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for x_init, y_init in train_loader:
            x_init, y_init = x_init.to(model_data["device"]), y_init.to(model_data["device"])
            out = model(x_init)
            if model_data["normalized"] is True:
                y_init = normalizer_y.decode(y_init)
                x_init = normalizer_x.decode(x_init)
            phi_h = x_init[:, :, :, 0:1]
            g_h = x_init[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["E"]
            p_h *= model_data["E"]
            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity(
                u_h, y_h, p_h, phi_h, g_h, model_data
            )
            init_loss_sums[0] += (torch.sum(loss_vij) / len(u_h)).item()
            init_loss_sums[1] += (torch.sum(loss_yij) / len(u_h)).item()
            init_loss_sums[2] += (torch.sum(loss_pij) / len(u_h)).item()
            init_loss_sums[3] += (torch.sum(loss_dyij) / len(u_h)).item()
    initial_loss = [s / n_train for s in init_loss_sums]
    Lam = model_data["lambda_loss"]
    print("Initial loss:", initial_loss)
    print("Initial weighted loss:",
          initial_loss[0] * Lam[0] + initial_loss[1] * Lam[1]
          + initial_loss[2] * Lam[2] + initial_loss[3] * Lam[3])
    model.train()

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep
            

        train_loss_epoch = 0.0
        train_l2 = 0.0
        
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["E"]
            p_h *= model_data["E"]
            u_true = y_true[:, :, :, 0:2]
            mask_node = get_node_masks(phi_h)
            Lam = model_data["lambda_loss"]

            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity(u_h, y_h, p_h, phi_h, g_h, model_data)
            loss_v = torch.sum(loss_vij) / len(u_h)
            loss_y = torch.sum(loss_yij) / len(u_h)
            loss_p = torch.sum(loss_pij) / len(u_h)
            loss_dy = torch.sum(loss_dyij) / len(u_h) 
            loss = loss_v * Lam[0] + loss_y * Lam[1] + loss_p * Lam[2] + loss_dy * Lam[3]

            train_l2_batch = loss_func(u_h * mask_node, u_true * mask_node)
            train_l2_batch /= len(x)
            train_l2 += train_l2_batch

            if model_data["fno"]["use_data"] is True:
                loss = loss + train_l2_batch

            loss.backward()

            optimizer.step()
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / n_train
        avg_train_l2 = train_l2 / n_train

        train_loss_log.append(avg_train_l2.item())

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, Loss: {avg_train_loss:.4f}, "
                  f"train_l2: {avg_train_l2:.4f}, "
                  f"test:{test_l2:.4f}, time:{t2-t1:.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

# Weak form + least squares + G_h(u, v) face term
def train_rvino_Gh_hyperelasticity_Guass(model, model_data, train_loader, test_loader,
                                      loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep
            

        train_loss_epoch = 0.0
        train_l2 = 0.0
        
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["E"]
            p_h *= model_data["E"]
            u_true = y_true[:, :, :, 0:2]
            mask_node = get_node_masks(phi_h)
            Lam = model_data["lambda_loss"]

            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity_Gh(u_h, y_h, p_h, phi_h, g_h, model_data)
            loss_v = torch.sum(loss_vij) / len(u_h)
            loss_y = torch.sum(loss_yij) / len(u_h)
            loss_p = torch.sum(loss_pij) / len(u_h)
            loss_dy = torch.sum(loss_dyij) / len(u_h) 
            loss = loss_v * Lam[0] + loss_y * Lam[1] + loss_p * Lam[2] + loss_dy * Lam[3]

            train_l2_batch = loss_func(u_h * mask_node, u_true * mask_node)
            train_l2_batch /= len(x)
            train_l2 += train_l2_batch

            if model_data["fno"]["use_data"] is True:
                loss = loss + train_l2_batch

            loss.backward()

            optimizer.step()
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / n_train
        avg_train_l2 = train_l2 / n_train

        train_loss_log.append(avg_train_l2.item())

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, Loss: {avg_train_loss:.4f}, "
                  f"train_l2: {avg_train_l2:.4f}, "
                  f"test:{test_l2:.4f}, time:{t2-t1:.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

def train_fno_hyperelasticity_Guass(model, model_data, train_loader, test_loader,
                                      loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep
            

        train_loss_epoch = 0.0
        train_l2 = 0.0
        
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["E"]
            p_h *= model_data["E"]
            u_true = y_true[:, :, :, 0:2]
            mask_node = get_node_masks(phi_h)
            loss = loss_func(u_h * mask_node, u_true * mask_node)
            if model_data["fno"].get("use_grad", False) is True:
                h = 1 / (model_data["grid_point_num"] - 1)
                mask_node_y = mask_node[:, 1:-1, :] * mask_node[:, :-2, :] * mask_node[:, 2:, :]
                mask_node_x = mask_node[:, :, 1:-1] * mask_node[:, :, :-2] * mask_node[:, :, 2:]
                loss_grad_y = mask_node_y * ((u_h[:, 2:, :] - u_h[:, :-2, :]) - (u_true[:, 2:, :] - u_true[:, :-2, :])) / (2 * h)
                loss_grad_x = mask_node_x * ((u_h[:, :, 2:] - u_h[:, :, :-2]) - (u_true[:, :, 2:] - u_true[:, :, :-2])) / (2 * h)
                loss_grad = (
                    (loss_grad_y.pow(2).sum(dim=-1).mean() + loss_grad_x.pow(2).sum(dim=-1).mean()) / 2
                )
                loss = loss + loss_grad
            loss /= len(x)
            train_l2 += loss

            loss.backward()

            optimizer.step()
            train_loss_epoch += loss.item()

        avg_train_l2 = train_l2 / n_train

        train_loss_log.append(avg_train_l2)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                  f"train_l2: {avg_train_l2:.4f}, "
                  f"test:{test_l2:.4f}, time:{t2-t1:.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

def train_rvino_hyperelasticity_trapezoid_Guass(model, model_data, train_loader, test_loader,
                                                       loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    loss_v_log = []
    loss_y_log = []
    loss_p_log = []
    loss_dy_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    model.eval()
    init_loss_sums = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for x_init, y_init in train_loader:
            x_init, y_init = x_init.to(model_data["device"]), y_init.to(model_data["device"])
            out = model(x_init)
            if model_data["normalized"] is True:
                y_init = normalizer_y.decode(y_init)
                x_init = normalizer_x.decode(x_init)
            phi_h = x_init[:, :, :, 0:1]
            g_h = x_init[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["norm_y"]
            p_h *= model_data["norm_p"]
            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity_trapezoid(
                u_h, y_h, p_h, phi_h, g_h, model_data
            )
            mask_node = get_node_masks(phi_h)
            tot_v = mask_node.sum(dim=(1, 2, 3), keepdim=True)
            tot_y = torch.sum((loss_yij >= 1e-12).double(), dim=(1, 2), keepdim=True)
            tot_p = torch.sum((loss_pij >= 1e-12).double(), dim=(1, 2, 3), keepdim=True)
            tot_dy = torch.sum((loss_dyij >= 1e-12).double(), dim=(1, 2, 3), keepdim=True)
            init_loss_sums[0] += (torch.sum(loss_vij / tot_v) / len(u_h)).item()
            init_loss_sums[1] += (torch.sum(loss_yij / tot_y) / len(u_h)).item()
            init_loss_sums[2] += (torch.sum(loss_pij / tot_p) / len(u_h)).item()
            init_loss_sums[3] += (torch.sum(loss_dyij / tot_dy) / len(u_h)).item()
    initial_loss = [s / n_train for s in init_loss_sums]
    Lam = model_data["lambda_loss"]
    print("Initial loss:", initial_loss)
    print("Initial weighted loss:",
          initial_loss[0] * Lam[0] + initial_loss[1] * Lam[1]
          + initial_loss[2] * Lam[2] + initial_loss[3] * Lam[3])
    model.train()

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep

        train_loss_epoch = 0.0
        train_l2 = 0
        train_loss_v_epoch = 0.0
        train_loss_y_epoch = 0.0
        train_loss_p_epoch = 0.0
        train_loss_dy_epoch = 0.0
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["norm_y"]
            p_h *= model_data["norm_p"]                
            u_true = y_true[:, :, :, 0:2]
            Lam = model_data["lambda_loss"]

            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity_trapezoid(u_h, y_h, p_h, phi_h, g_h, model_data)
            mask_node = get_node_masks(phi_h)
            tot_v = mask_node.sum(dim=(1,2,3), keepdim=True)
            tot_y = torch.sum((loss_yij>=1e-12).double(), dim=(1,2), keepdim=True)
            tot_p = torch.sum((loss_pij>=1e-12).double(), dim=(1,2,3), keepdim=True)
            tot_dy = torch.sum((loss_dyij>=1e-12).double(), dim=(1,2,3), keepdim=True)
            loss_v = torch.sum(loss_vij / tot_v) / len(u_h)
            loss_y = torch.sum(loss_yij / tot_y) / len(u_h)
            loss_p = torch.sum(loss_pij / tot_p) / len(u_h)
            loss_dy = torch.sum(loss_dyij / tot_dy) / len(u_h) 
            loss = loss_v * Lam[0] + loss_y * Lam[1] + loss_p * Lam[2] + loss_dy * Lam[3]

            mask_node = get_node_masks(phi_h)
            train_l2_batch = loss_func(u_h * mask_node, u_true * mask_node)
            train_l2_batch /= len(x)
            if model_data["fno"]["use_data"] is True:
                loss += train_l2_batch
            loss.backward()

            optimizer.step()
            train_loss_epoch += loss.item()
            train_loss_v_epoch += loss_v.item()
            train_loss_y_epoch += loss_y.item()
            train_loss_p_epoch += loss_p.item()
            train_loss_dy_epoch += loss_dy.item()

            train_l2 += train_l2_batch.item()
        avg_train_loss = train_loss_epoch / n_train
        avg_train_l2 = train_l2 / n_train
        avg_loss_v = train_loss_v_epoch / n_train
        avg_loss_y = train_loss_y_epoch / n_train
        avg_loss_p = train_loss_p_epoch / n_train
        avg_loss_dy = train_loss_dy_epoch / n_train
        loss_v_log.append(avg_loss_v)
        loss_y_log.append(avg_loss_y)
        loss_p_log.append(avg_loss_p)
        loss_dy_log.append(avg_loss_dy)

        if scheduler_adam is not None:
            if isinstance(scheduler_adam, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_adam.step(loss)
            else:
                scheduler_adam.step()

        train_loss_log.append(avg_train_l2)

        model.eval()
        test_l2 = 0.0
        test_l2_f = 0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep >= 15000:
            model_data["step_size"] = 1
        if not np.isfinite(avg_train_loss) or not np.isfinite(avg_train_l2) or not np.isfinite(test_l2):
            print(f"Epoch {ep}: NaN or Inf in train/test metrics; stopping early.")
            break
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"train_l2: {avg_train_l2:.4f}, "
                f"test_l2: {test_l2:.4f}, "
                f"time: {(t2 - t1):.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log, loss_v_log, loss_y_log, loss_p_log, loss_dy_log

def train_fno_hyperelasticity_trapezoid_Guass(model, model_data, train_loader, test_loader,
                                            loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep

        train_loss_epoch = 0.0
        train_l2 = 0
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["norm_y"]
            p_h *= model_data["norm_p"]                
            u_true = y_true[:, :, :, 0:2]
            Lam = model_data["lambda_loss"]

            mask_node = get_node_masks(phi_h)
            loss = loss_func(u_h * mask_node, u_true * mask_node)
            if model_data["fno"]["use_grad"] is True:
                h = 1 / (model_data["grid_point_num"] - 1)
                mask_node_y = mask_node[:, 1:-1, :] * mask_node[:, :-2, :] * mask_node[:, 2:, :]
                mask_node_x = mask_node[:, :, 1:-1] * mask_node[:, :, :-2] * mask_node[:, :, 2:]
                loss_grad_y = mask_node_y * ((u_h[:, 2:, :] - u_h[:, :-2, :]) - (u_true[:, 2:, :] - u_true[:, :-2, :])) / (2 * h)
                loss_grad_x = mask_node_x * ((u_h[:, :, 2:] - u_h[:, :, :-2]) - (u_true[:, :, 2:] - u_true[:, :, :-2])) / (2 * h)
                # (n, h-1, w, 2) / (n, h, w-1, 2) -> squared L2 on last axis, then mean over space and batch
                loss_grad = (
                    (loss_grad_y.pow(2).sum(dim=-1).mean() + loss_grad_x.pow(2).sum(dim=-1).mean()) / 2
                )
                loss = loss + loss_grad
            loss /= len(x)
            loss.backward()
            optimizer.step()

            train_l2 += loss.item()

            
        avg_train_l2 = train_l2 / n_train

        

        if scheduler_adam is not None:
            if isinstance(scheduler_adam, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_adam.step(loss)
            else:
                scheduler_adam.step()

        train_loss_log.append(avg_train_l2)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)

        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                f"train_l2: {avg_train_l2:.4f}, "
                f"test_l2: {test_l2:.4f}, "
                f"time: {(t2 - t1):.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

# E_in / E_out vessel geometry with follower traction
def train_rvino_hyperelasticity_vessel_Guass(model, model_data, train_loader, test_loader,
                                              loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    model.eval()
    init_loss_sums = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for x_init, y_init in train_loader:
            x_init, y_init = x_init.to(model_data["device"]), y_init.to(model_data["device"])
            out = model(x_init)
            if model_data["normalized"] is True:
                y_init = normalizer_y.decode(y_init)
                x_init = normalizer_x.decode(x_init)
            Ein = x_init[..., 0:1]
            Eout = x_init[..., 1:2]
            Edx = x_init[..., 2:4]
            Edy = x_init[..., 4:6]
            g_h = x_init[:, :, :, 6:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]
            y_h *= model_data["norm_y"]
            p_h *= model_data["norm_p"]
            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity_vessel(
                u_h, y_h, p_h, Ein, Eout, Edx, Edy, g_h, model_data
            )
            init_loss_sums[0] += (torch.sum(loss_vij) / len(u_h)).item()
            init_loss_sums[1] += (torch.sum(loss_yij) / len(u_h)).item()
            init_loss_sums[2] += (torch.sum(loss_pij) / len(u_h)).item()
            init_loss_sums[3] += (torch.sum(loss_dyij) / len(u_h)).item()
    initial_loss = [s / n_train for s in init_loss_sums]
    Lam = model_data["lambda_loss"]
    print("Initial loss:", initial_loss)
    print("Initial weighted loss:",
          initial_loss[0] * Lam[0] + initial_loss[1] * Lam[1]
          + initial_loss[2] * Lam[2] + initial_loss[3] * Lam[3])
    model.train()

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep

        train_loss_epoch = 0.0
        train_l2 = 0
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            Ein = x[..., 0:1]
            Eout = x[..., 1:2]
            Edx = x[..., 2:4]
            Edy = x[..., 4:6]
            g_h = x[:, :, :, 6:]
            u_h = out[:, :, :, 0:2]
            y_h = out[:, :, :, 2:6]
            p_h = out[:, :, :, 6:8]

            phi_h = Ein * Eout
            mask_node = get_node_masks(phi_h[:, ::2, ::2])
            tot_v = mask_node.sum(dim=(1,2,3), keepdim=True)
            y_h *= model_data["norm_y"]
            p_h *= model_data["norm_p"]    
            u_true = y_true[:, :, :, 0:2]
            Lam = model_data["lambda_loss"]

            loss_vij, loss_yij, loss_pij, loss_dyij = get_loss_hyperelasticity_vessel(u_h, y_h, p_h, Ein, Eout, Edx, Edy, g_h, model_data)

            tot_y = torch.sum((loss_yij>=1e-12).double(), dim=(1,2), keepdim=True)
            tot_p = torch.sum((loss_pij>=1e-12).double(), dim=(1,2,3), keepdim=True)
            tot_dy = torch.sum((loss_dyij>=1e-12).double(), dim=(1,2,3), keepdim=True)
            loss_v = torch.sum(loss_vij / tot_v) / len(u_h)
            loss_y = torch.sum(loss_yij / tot_y) / len(u_h)
            loss_p = torch.sum(loss_pij / tot_p) / len(u_h)
            loss_dy = torch.sum(loss_dyij / tot_dy) / len(u_h) 

            loss_v = torch.sum(loss_vij) / len(u_h)
            loss_y = torch.sum(loss_yij) / len(u_h)
            loss_p = torch.sum(loss_pij) / len(u_h)
            loss_dy = torch.sum(loss_dyij) / len(u_h) 
            loss = loss_v * Lam[0] + loss_y * Lam[1] + loss_p * Lam[2] + loss_dy * Lam[3]

            phi_grid = phi_h[:, ::2, ::2]
            mask_node = get_node_masks(phi_grid)
            train_l2_batch = loss_func(u_h[:, ::2, ::2] * mask_node, u_true[:, ::2, ::2] * mask_node)
            train_l2_batch /= len(x)
            train_l2 += train_l2_batch

            if model_data["fno"]["use_data"] is True:
                loss += train_l2_batch                
                
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            
        avg_train_loss = train_loss_epoch / n_train
        avg_train_l2 = train_l2 / n_train

        if scheduler_adam is not None:
            if isinstance(scheduler_adam, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_adam.step(loss)
            else:
                scheduler_adam.step()

        train_loss_log.append(avg_train_l2.item())

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                Ein = x[..., 0:1]
                Eout = x[..., 1:2]
                Edx = x[..., 2:4]
                Edy = x[..., 4:6]
                g_h = x[:, :, :, 6:]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                phi_h = Ein * Eout
                phi_grid = phi_h[:, ::2, ::2]
                mask_node = get_node_masks(phi_grid)
                test_l2_batch = loss_func(u_h[:, ::2, ::2] * mask_node, u_true[:, ::2, ::2] * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"train_l2: {avg_train_l2:.4f}, "
                f"test_l2: {test_l2:.4f}, "
                f"time: {(t2 - t1):.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

# E_in / E_out vessel geometry with follower traction
def train_fno_hyperelasticity_vessel_Guass(model, model_data, train_loader, test_loader,
                                           loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep

        train_loss_epoch = 0.0
        train_l2 = 0
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                x = normalizer_x.decode(x)
            Ein = x[..., 0:1]
            Eout = x[..., 1:2]
            Edx = x[..., 2:4]
            Edy = x[..., 4:6]
            g_h = x[:, :, :, 6:]
            u_h = out[:, :, :, 0:2]
            u_true = y_true[:, :, :, 0:2]

            phi_h = Ein * Eout
            mask_node = get_node_masks(phi_h[:, ::2, ::2])
            loss = loss_func(u_h[:, ::2, ::2] * mask_node, u_true[:, ::2, ::2] * mask_node)
            if model_data["fno"]["use_grad"] is True:
                h = 1 / (model_data["grid_point_num"] - 1)
                mask_node_y = mask_node[:, 1:-1, :] * mask_node[:, :-2, :] * mask_node[:, 2:, :]
                mask_node_x = mask_node[:, :, 1:-1] * mask_node[:, :, :-2] * mask_node[:, :, 2:]
                uh = u_h[:, ::2, ::2]
                utrue = u_true[:, ::2, ::2]
                loss_grad_y = mask_node_y * ((uh[:, 2:, :] - uh[:, :-2, :]) - (utrue[:, 2:, :] - utrue[:, :-2, :])) / (2 * h)
                loss_grad_x = mask_node_x * ((uh[:, :, 2:] - uh[:, :, :-2]) - (utrue[:, :, 2:] - utrue[:, :, :-2])) / (2 * h)
                # (n, h-1, w, 2) / (n, h, w-1, 2) -> squared L2 on last axis, then mean over space and batch
                loss_grad = (
                    (loss_grad_y.pow(2).sum(dim=-1).mean() + loss_grad_x.pow(2).sum(dim=-1).mean()) / 2
                )
                loss = loss + loss_grad
            loss /= len(x)

            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        avg_train_loss = train_loss_epoch / n_train

        if scheduler_adam is not None:
            if isinstance(scheduler_adam, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_adam.step(loss)
            else:
                scheduler_adam.step()

        train_loss_log.append(avg_train_loss)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    x = normalizer_x.decode(x)
                Ein = x[..., 0:1]
                Eout = x[..., 1:2]
                Edx = x[..., 2:4]
                Edy = x[..., 4:6]
                g_h = x[:, :, :, 6:]
                u_h = out[:, :, :, 0:2]
                u_true = y_true[:, :, :, 0:2]
                phi_h = Ein * Eout
                phi_grid = phi_h[:, ::2, ::2]
                mask_node = get_node_masks(phi_grid)
                test_l2_batch = loss_func(u_h[:, ::2, ::2] * mask_node, u_true[:, ::2, ::2] * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                f"Loss: {avg_train_loss:.4f}, "
                f"test_l2: {test_l2:.4f}, "
                f"time: {(t2 - t1):.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

# Dirichlet boundary-value problem
def train_rvino_hyperelasticity_DBC_Guass(model, model_data, train_loader, test_loader,
                                      loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep
            

        train_loss_epoch = 0.0
        train_l2 = 0.0
        
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            # Standard Adam step
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                out = normalizer_y.decode(out)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:3]
            f_h = x[:, :, :, 3:]
            u_h = out[:, :, :, 0:2] * phi_h + g_h
            u_true = y_true[:, :, :, 0:2] * phi_h + g_h
            mask_node = get_node_masks(phi_h)
            Lam = model_data["lambda_loss"]

            loss = get_loss_hyperelasticity_DBC(u_h, phi_h, f_h, model_data) * Lam

            train_l2_batch = loss_func(u_h * mask_node, u_true * mask_node)
            train_l2_batch /= len(x)
            train_l2 += train_l2_batch

            if model_data["fno"]["use_data"] is True:
                loss = loss + train_l2_batch

            loss.backward()

            optimizer.step()
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / n_train
        avg_train_l2 = train_l2 / n_train

        train_loss_log.append(avg_train_l2.item())

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    out = normalizer_y.decode(out)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                g_h = x[:, :, :, 1:3]
                u_h = out[:, :, :, 0:2] * phi_h + g_h
                u_true = y_true[:, :, :, 0:2] * phi_h + g_h
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, Loss: {avg_train_loss:.4f}, "
                  f"train_l2: {avg_train_l2:.4f}, "
                  f"test:{test_l2:.4f}, time:{t2-t1:.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

def train_fno_hyperelasticity_DBC_Guass(model, model_data, train_loader, test_loader,
                                      loss_func, optimizer_adam, scheduler_adam, normalizers):
    model.train()
    n_train = len(train_loader)
    train_loss_log = []
    test_loss_log = []
    if model_data["normalized"] is True:
        normalizer_x = normalizers[0]
        normalizer_y = normalizers[1]

    # Use the Adam optimizer
    optimizer = optimizer_adam

    t1 = default_timer()
    ep1 = -1
    es = model_data["patience"]
    for ep in range(model_data["num_epoch"]):
        if (ep - ep1) % es == 0:
            if model_data["learning_rate_adam"] >= model_data["min_lr"]:
                model_data["learning_rate_adam"] *= model_data["gamma"]
                optimizer.param_groups[0]['lr'] = model_data["learning_rate_adam"]
                print("learning rate:", model_data["learning_rate_adam"])
                ep1 = ep

        train_l2 = 0
        for x, y_true in train_loader:
            x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
            optimizer.zero_grad()
            out = model(x)
            if model_data["normalized"] is True:
                y_true = normalizer_y.decode(y_true)
                out = normalizer_y.decode(out)
                x = normalizer_x.decode(x)
            phi_h = x[:, :, :, 0:1]
            g_h = x[:, :, :, 1:3]
            f_h = x[:, :, :, 3:]
            u_h = out[:, :, :, 0:2] * phi_h + g_h
            u_true = y_true[:, :, :, 0:2] * phi_h + g_h
            mask_node = get_node_masks(phi_h)

            loss = loss_func(u_h * mask_node, u_true * mask_node)
            if model_data["fno"].get("use_grad", False) is True:
                h = 1 / (model_data["grid_point_num"] - 1)
                mask_node_y = mask_node[:, 1:-1, :] * mask_node[:, :-2, :] * mask_node[:, 2:, :]
                mask_node_x = mask_node[:, :, 1:-1] * mask_node[:, :, :-2] * mask_node[:, :, 2:]
                loss_grad_y = mask_node_y * ((u_h[:, 2:, :] - u_h[:, :-2, :]) - (u_true[:, 2:, :] - u_true[:, :-2, :])) / (2 * h)
                loss_grad_x = mask_node_x * ((u_h[:, :, 2:] - u_h[:, :, :-2]) - (u_true[:, :, 2:] - u_true[:, :, :-2])) / (2 * h)
                loss_grad = (
                    (loss_grad_y.pow(2).sum(dim=-1).mean() + loss_grad_x.pow(2).sum(dim=-1).mean()) / 2
                )
                loss = loss + loss_grad
            loss /= len(x)

            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        avg_train_l2 = train_l2 / n_train

        train_loss_log.append(avg_train_l2)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(model_data["device"]), y_true.to(model_data["device"])
                out = model(x)
                if model_data["normalized"] is True:
                    y_true = normalizer_y.decode(y_true)
                    out = normalizer_y.decode(out)
                    x = normalizer_x.decode(x)
                phi_h = x[:, :, :, 0:1]
                g_h = x[:, :, :, 1:3]
                u_h = out[:, :, :, 0:2] * phi_h + g_h
                u_true = y_true[:, :, :, 0:2] * phi_h + g_h
                mask_node = get_node_masks(phi_h)
                test_l2_batch = loss_func(u_h * mask_node, u_true * mask_node).item()
                test_l2_batch /= len(x)
                test_l2 += test_l2_batch

        test_l2 /= len(test_loader)
        if ep % model_data["step_size"] == 0:
            t2 = default_timer()
            print(f"Epoch {ep}, "
                f"train_l2: {avg_train_l2:.4f}, "
                f"test_l2: {test_l2:.4f}, "
                f"time: {(t2 - t1):.4f}")
            t1 = t2
        test_loss_log.append(test_l2)

    return train_loss_log, test_loss_log

# Weak form + least squares
def get_loss_hyperelasticity(u_h, y_h, p_h, phi_h, g_h, model_data):
    gamma_div, gamma_u, gamma_p = 1, 1, 1

    mask_Cell_Omega, mask_Cut_Cell = get_cell_masks(phi_h)
    mask_Bound_V, mask_Bound_H = get_boundary_masks(mask_Cell_Omega)
    mask_Node_Cut_Cell = get_node_mask_from_cell_mask(mask_Cut_Cell)
    mask_node = get_node_masks(phi_h)
    torch.set_printoptions(threshold=torch.inf)

    # Cell-corner displacements for Q4 gradients
    u00 = u_h[:, 0:-1, 0:-1, :]  # bottom-left, (batch, H-1, W-1, 2)
    u10 = u_h[:, 1:, 0:-1, :]  # top-left
    u11 = u_h[:, 1:, 1:, :]  # top-right
    u01 = u_h[:, 0:-1, 1:, :]  # bottom-right
    h = 1 / (model_data["grid_point_num"] - 1)
    U = torch.stack([
        torch.stack([u00, u01], dim=0),
        torch.stack([u10, u11], dim=0),
    ], dim=0)   # (2, 2, b, H-1, W-1, 2)
    xgrid = torch.tensor(
        [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)],
        dtype=torch.float64, device=model_data["device"]
    )
    Y, X = torch.meshgrid(xgrid, xgrid, indexing='ij')
    w1d = torch.tensor(
        [5/9, 8/9, 5/9],
        dtype=torch.float64, device=model_data["device"]
    )

    W = (w1d[:, None] * w1d[None, :]).view(3, 3, 1, 1, 1, 1)

    dudx, dudy = d_shape_functions_Q4(U, Y, X, h, h)   # (3, 3, batch, H-1, W-1, 2)
    Sigma_list = get_P(dudx, dudy, model_data)  # (3, 3, batch, H-1, W-1, 2, 2)
    V = torch.zeros(2, 2, 4, 1, 1, 1).to(model_data["device"])   # nodal test-function basis

    V[0, 0, 0, 0, 0, 0] = 1.0  # bottom-left
    V[0, 1, 1, 0, 0, 0] = 1.0  # bottom-right
    V[1, 0, 2, 0, 0, 0] = 1.0  # top-left
    V[1, 1, 3, 0, 0, 0] = 1.0  # top-right
    dvdx, dvdy = d_shape_functions_Q4(V, Y, X, h, h)   # (3, 3, 4, 1, 1, 1)

    # Residual of test function v, shape (batch, H-1, W-1, 2)
    res_v = torch.zeros_like(u_h)
    # P : nabla(v)
    # Use the same spacing in x and y (dx = dy = h)
    # i=0, j=0
    res_v[:, 0:-1, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 0:1] + Sigma_list[..., 1] * dvdy[:, :, 0:1]) * W, dim=(0, 1)
    )
    # i=0, j=1
    res_v[:, 0:-1, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 1:2] + Sigma_list[..., 1] * dvdy[:, :, 1:2]) * W, dim=(0, 1)
    )
    # i=1, j=0
    res_v[:, 1:, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 2:3] + Sigma_list[..., 1] * dvdy[:, :, 2:3]) * W, dim=(0, 1)
    )
    # i=1, j=1
    res_v[:, 1:, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 3:4] + Sigma_list[..., 1] * dvdy[:, :, 3:4]) * W, dim=(0, 1)
    )

    # y n · v on interior faces
    yh00 = y_h[:, 0:-1, 0:-1, :]
    yh01 = y_h[:, 0:-1, 1:, :]
    yh10 = y_h[:, 1:, 0:-1, :]
    yh11 = y_h[:, 1:, 1:, :]
    Yh_ij = torch.stack([
        torch.stack([yh00, yh01], dim=0),
        torch.stack([yh10, yh11], dim=0),
    ], dim=0)   # (2, 2, batch, H-1, W-1, 4)
    # Vertical faces
    Y_l, X_l = torch.meshgrid(xgrid, -torch.ones_like(xgrid[0]), indexing='ij')
    Yh_list = shape_functions_Q4(Yh_ij[:, :, :, :, 1:, :], Y_l, X_l)   # (3, 1, batch, H-1, W-2, 4)

    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 1, batch, H-1, W-2, 2, 2)

    Yh_10 = Yh_bound_list[0, 0][..., 0]  # (batch, H-1, W-2, 2)
    Yh_20 = Yh_bound_list[1, 0][..., 0]
    Yh_30 = Yh_bound_list[2, 0][..., 0]
    # i=0, j=0
    res_v[:, 0:-1, 1:-1, :] += mask_Bound_V * h/2 * (
        w1d[0] * Yh_10 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 - np.sqrt(15)/10)
    )
    # i=1, j=0
    res_v[:, 1:, 1:-1, :] += mask_Bound_V * h/2 * (
        w1d[0] * Yh_10 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 + np.sqrt(15)/10)
    )
    # Horizontal faces
    Y_d, X_d = torch.meshgrid(-torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    # down
    Yh_list = shape_functions_Q4(Yh_ij[:, :, :, 1:, :, :], Y_d, X_d) # (1, 3, batch, H-2, W-1, 4)
    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (1, 3, batch, H-1, W-2, 2, 2)
    
    Yh_01 = Yh_bound_list[0, 0][..., 1]  # (batch, H-2, W-1, 2)
    Yh_02 = Yh_bound_list[0, 1][..., 1]
    Yh_03 = Yh_bound_list[0, 2][..., 1]
    # i=0, j=0
    res_v[:, 1:-1, 0:-1, :] += mask_Bound_H * h/2 * (
        w1d[0] * Yh_01 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 - np.sqrt(15)/10)
    )
    # i=0, j=1
    res_v[:, 1:-1, 1:, :] += mask_Bound_H * h/2 * (
        w1d[0] * Yh_01 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 + np.sqrt(15)/10)
    )

    # Traction g · v, Simpson rule
    res_v[:, -1, 0:-1, :] -= (g_h[:, -1, 0:-1, :] * h / 6 + (g_h[:, -1, 0:-1, :] + g_h[:, -1, 1:, :]) * h / 6)
    res_v[:, -1, 1:, :] -= (g_h[:, -1, 1:, :] * h / 6 + (g_h[:, -1, 0:-1, :] + g_h[:, -1, 1:, :]) * h / 6)

    res_v[:, 0, :, :] = 0

    res_v = torch.where(mask_Node_Cut_Cell>0, model_data["lambda_vin"] * res_v, res_v)
    loss_vij = res_v ** 2
    loss_v = torch.sum(loss_vij) / len(u_h)

    # ||Y + P(F)||^2 on cut cells
    Yh_list = shape_functions_Q4(Yh_ij, Y, X)   # (3, 3, batch, H-1, W-1, 4)
    Yh_int_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 3, batch, H-1, W-1, 2, 2)

    loss_yij = torch.sum(gamma_u * mask_Cut_Cell.unsqueeze(-1) * (h/2) ** 2 * torch.sum(
        (Yh_int_list + Sigma_list) ** 2 * W.unsqueeze(-1), dim=(0, 1)
    ), dim=(-2, -1))  # (batch, H-1, W-1)
    loss_y = torch.sum(loss_yij) / len(u_h)

    # ||Y nabla phi + p phi / h||^2 on cut cells
    Phi = stack_Q4_data(phi_h)   # (2, 2, batch, H-1, W-1, 1)
    P = stack_Q4_data(p_h)   # (2, 2, batch, H-1, W-1, 2)
    dphidx, dphidy = d_shape_functions_Q4(Phi, Y, X, h, h) # (3, 3, batch, H-1, W-1, 1)
    phi_list = shape_functions_Q4(Phi, Y, X)    # (3, 3, batch, H-1, W-1, 1)
    p_list = shape_functions_Q4(P, Y, X)    # (3, 3, batch, H-1, W-1, 2)
    loss_pij = gamma_p * mask_Cut_Cell / 4 * torch.sum(
        (Yh_int_list[..., 0] * dphidx + Yh_int_list[..., 1] * dphidy + 1/h * p_list * phi_list) ** 2 * W,
        dim=(0, 1)
    )# (b, H-1, W-1, 2)

    loss_p = torch.sum(loss_pij) / len(u_h)

    # ||div y||^2 on cut cells
    dYdx, dYdy = d_shape_functions_Q4(Yh_ij, Y, X, h, h) # (3, 3, batch, H-1, W-1, 4)
    res_dy = dYdx[..., 0::2] + dYdy[..., 1::2]    # (3, 3, batch, H-1, W-1, 2)
    loss_dyij = gamma_div * mask_Cut_Cell * (h/2) ** 2 * torch.sum(res_dy ** 2 * W, dim=(0, 1))
    loss_dy = torch.sum(loss_dyij) / len(u_h)

    g_mean = g_h.norm(dim=-1).mean(dim=(1, 2), keepdim=True).unsqueeze(-1) ** 2
    loss_vij /= g_mean
    loss_yij /= g_mean.squeeze(-1)
    loss_pij /= g_mean
    loss_dyij /= g_mean
    return loss_vij, loss_yij, loss_pij, loss_dyij

# Weak form + least squares + G_h(u, v) face term
def get_loss_hyperelasticity_Gh(u_h, y_h, p_h, phi_h, g_h, model_data):
    gamma_div, gamma_u, gamma_p = 1, 1, 1

    mask_Cell_Omega, mask_Cut_Cell = get_cell_masks(phi_h)
    mask_Face_V, mask_Face_H = get_face_masks(mask_Cut_Cell, mask_Cell_Omega)
    mask_Bound_V, mask_Bound_H = get_boundary_masks(mask_Cell_Omega)
    mask_Node_Cut_Cell = get_node_mask_from_cell_mask(mask_Cut_Cell)
    mask_node = get_node_masks(phi_h)
    torch.set_printoptions(threshold=torch.inf)

    # Cell-corner displacements for Q4 gradients
    u00 = u_h[:, 0:-1, 0:-1, :]  # bottom-left, (batch, H-1, W-1, 2)
    u10 = u_h[:, 1:, 0:-1, :]  # top-left
    u11 = u_h[:, 1:, 1:, :]  # top-right
    u01 = u_h[:, 0:-1, 1:, :]  # bottom-right
    h = 1 / (model_data["grid_point_num"] - 1)
    U = torch.stack([
        torch.stack([u00, u01], dim=0),
        torch.stack([u10, u11], dim=0),
    ], dim=0)   # (2, 2, b, H-1, W-1, 2)
    xgrid = torch.tensor(
        [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)],
        dtype=torch.float64, device=model_data["device"]
    )
    Y, X = torch.meshgrid(xgrid, xgrid, indexing='ij')
    w1d = torch.tensor(
        [5/9, 8/9, 5/9],
        dtype=torch.float64, device=model_data["device"]
    )

    W = (w1d[:, None] * w1d[None, :]).view(3, 3, 1, 1, 1, 1)

    dudx, dudy = d_shape_functions_Q4(U, Y, X, h, h)   # (3, 3, batch, H-1, W-1, 2)
    Sigma_list = get_P(dudx, dudy, model_data)  # (3, 3, batch, H-1, W-1, 2, 2)
    V = torch.zeros(2, 2, 4, 1, 1, 1).to(model_data["device"])   # nodal test-function basis

    V[0, 0, 0, 0, 0, 0] = 1.0  # bottom-left
    V[0, 1, 1, 0, 0, 0] = 1.0  # bottom-right
    V[1, 0, 2, 0, 0, 0] = 1.0  # top-left
    V[1, 1, 3, 0, 0, 0] = 1.0  # top-right
    dvdx, dvdy = d_shape_functions_Q4(V, Y, X, h, h)   # (3, 3, 4, 1, 1, 1)

    # Residual of test function v, shape (batch, H-1, W-1, 2)
    res_v = torch.zeros_like(u_h)
    # P : nabla(v)
    # Use the same spacing in x and y (dx = dy = h)
    # i=0, j=0
    res_v[:, 0:-1, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 0:1] + Sigma_list[..., 1] * dvdy[:, :, 0:1]) * W, dim=(0, 1)
    )
    # i=0, j=1
    res_v[:, 0:-1, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 1:2] + Sigma_list[..., 1] * dvdy[:, :, 1:2]) * W, dim=(0, 1)
    )
    # i=1, j=0
    res_v[:, 1:, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 2:3] + Sigma_list[..., 1] * dvdy[:, :, 2:3]) * W, dim=(0, 1)
    )
    # i=1, j=1
    res_v[:, 1:, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 3:4] + Sigma_list[..., 1] * dvdy[:, :, 3:4]) * W, dim=(0, 1)
    )

    # y n · v on interior faces
    yh00 = y_h[:, 0:-1, 0:-1, :]
    yh01 = y_h[:, 0:-1, 1:, :]
    yh10 = y_h[:, 1:, 0:-1, :]
    yh11 = y_h[:, 1:, 1:, :]
    Yh_ij = torch.stack([
        torch.stack([yh00, yh01], dim=0),
        torch.stack([yh10, yh11], dim=0),
    ], dim=0)   # (2, 2, batch, H-1, W-1, 4)
    # Vertical faces
    Y_l, X_l = torch.meshgrid(xgrid, -torch.ones_like(xgrid[0]), indexing='ij')
    Yh_list = shape_functions_Q4(Yh_ij[:, :, :, :, 1:, :], Y_l, X_l)   # (3, 1, batch, H-1, W-2, 4)

    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 1, batch, H-1, W-2, 2, 2)

    Yh_10 = Yh_bound_list[0, 0][..., 0]  # (batch, H-1, W-2, 2)
    Yh_20 = Yh_bound_list[1, 0][..., 0]
    Yh_30 = Yh_bound_list[2, 0][..., 0]
    # i=0, j=0
    res_v[:, 0:-1, 1:-1, :] += mask_Bound_V * h/2 * (
        w1d[0] * Yh_10 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 - np.sqrt(15)/10)
    )
    # i=1, j=0
    res_v[:, 1:, 1:-1, :] += mask_Bound_V * h/2 * (
        w1d[0] * Yh_10 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 + np.sqrt(15)/10)
    )
    # Horizontal faces
    Y_d, X_d = torch.meshgrid(-torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    # down
    Yh_list = shape_functions_Q4(Yh_ij[:, :, :, 1:, :, :], Y_d, X_d) # (1, 3, batch, H-2, W-1, 4)
    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (1, 3, batch, H-1, W-2, 2, 2)
    
    Yh_01 = Yh_bound_list[0, 0][..., 1]  # (batch, H-2, W-1, 2)
    Yh_02 = Yh_bound_list[0, 1][..., 1]
    Yh_03 = Yh_bound_list[0, 2][..., 1]
    # i=0, j=0
    res_v[:, 1:-1, 0:-1, :] += mask_Bound_H * h/2 * (
        w1d[0] * Yh_01 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 - np.sqrt(15)/10)
    )
    # i=0, j=1
    res_v[:, 1:-1, 1:, :] += mask_Bound_H * h/2 * (
        w1d[0] * Yh_01 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 + np.sqrt(15)/10)
    )

    # G_h(u,v) = h * int [P(F) n] · [D_u P[v] n]
    add_Gh_vform_to_res_v(
        res_v, U, V, mask_Face_V, mask_Face_H, xgrid, w1d, h, h, model_data,
        h_face_yx=True,
    )

    # Traction g · v, Simpson rule
    res_v[:, -1, 0:-1, :] -= (g_h[:, -1, 0:-1, :] * h / 6 + (g_h[:, -1, 0:-1, :] + g_h[:, -1, 1:, :]) * h / 6)
    res_v[:, -1, 1:, :] -= (g_h[:, -1, 1:, :] * h / 6 + (g_h[:, -1, 0:-1, :] + g_h[:, -1, 1:, :]) * h / 6)

    res_v[:, 0, :, :] = 0

    res_v = torch.where(mask_Node_Cut_Cell>0, model_data["lambda_vin"] * res_v, res_v)
    loss_vij = res_v ** 2
    loss_v = torch.sum(loss_vij) / len(u_h)

    # ||Y + P(F)||^2 on cut cells
    Yh_list = shape_functions_Q4(Yh_ij, Y, X)   # (3, 3, batch, H-1, W-1, 4)
    Yh_int_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 3, batch, H-1, W-1, 2, 2)

    loss_yij = torch.sum(gamma_u * mask_Cut_Cell.unsqueeze(-1) * (h/2) ** 2 * torch.sum(
        (Yh_int_list + Sigma_list) ** 2 * W.unsqueeze(-1), dim=(0, 1)
    ), dim=(-2, -1))  # (batch, H-1, W-1)
    loss_y = torch.sum(loss_yij) / len(u_h)

    # ||Y nabla phi + p phi / h||^2 on cut cells
    Phi = stack_Q4_data(phi_h)   # (2, 2, batch, H-1, W-1, 1)
    P = stack_Q4_data(p_h)   # (2, 2, batch, H-1, W-1, 2)
    dphidx, dphidy = d_shape_functions_Q4(Phi, Y, X, h, h) # (3, 3, batch, H-1, W-1, 1)
    phi_list = shape_functions_Q4(Phi, Y, X)    # (3, 3, batch, H-1, W-1, 1)
    p_list = shape_functions_Q4(P, Y, X)    # (3, 3, batch, H-1, W-1, 2)
    loss_pij = gamma_p * mask_Cut_Cell / 4 * torch.sum(
        (Yh_int_list[..., 0] * dphidx + Yh_int_list[..., 1] * dphidy + 1/h * p_list * phi_list) ** 2 * W,
        dim=(0, 1)
    )# (b, H-1, W-1, 2)

    loss_p = torch.sum(loss_pij) / len(u_h)

    # ||div y||^2 on cut cells
    dYdx, dYdy = d_shape_functions_Q4(Yh_ij, Y, X, h, h) # (3, 3, batch, H-1, W-1, 4)
    res_dy = dYdx[..., 0::2] + dYdy[..., 1::2]    # (3, 3, batch, H-1, W-1, 2)
    loss_dyij = gamma_div * mask_Cut_Cell * (h/2) ** 2 * torch.sum(res_dy ** 2 * W, dim=(0, 1))
    loss_dy = torch.sum(loss_dyij) / len(u_h)

    g_mean = g_h.norm(dim=-1).mean(dim=(1, 2), keepdim=True).unsqueeze(-1) ** 2
    loss_vij /= g_mean
    loss_yij /= g_mean.squeeze(-1)
    loss_pij /= g_mean
    loss_dyij /= g_mean
    return loss_vij, loss_yij, loss_pij, loss_dyij

# Weak form + least squares
def get_loss_hyperelasticity_trapezoid(u_h, y_h, p_h, phi_h, g_h, model_data):
    torch.set_printoptions(threshold=torch.inf)

    gamma_div, gamma_u, gamma_p = 1, 1, 1
    mask_Cell_Omega, mask_Cut_Cell = get_cell_masks(phi_h)
    mask_Cell_Omega_padded = F.pad(
        mask_Cell_Omega,
        (0, 0,   # no pad on the last (channel) axis
        0, 1,   # pad 1 column on the right of W
        1, 1),  # pad 1 row on top and bottom of H
        mode='constant',
        value=0
    )   # (b, H+1, W, 1)
    mask_Bound_V, mask_Bound_H = get_boundary_masks(mask_Cell_Omega_padded)
    mask_Bound_V = mask_Bound_V[:, 1:-1]    # drop padded rows, (b, H-1, W-1, 1)
    mask_Bound_H = mask_Bound_H[:, :, :-1]  # (b, H, W-1)

    mask_Bound_V[:, :, -1] = 0
    mask_Bound_g = mask_Cell_Omega[:, :, -1]

    mask_Node_Cut_Cell = get_node_mask_from_cell_mask(mask_Cut_Cell)
    mask_node = get_node_masks(phi_h)
    mask_laplace = get_node_masks(mask_node)
    

    # Cell-corner displacements for Q4 gradients
    u00 = u_h[:, 0:-1, 0:-1, :]  # bottom-left, (batch, H-1, W-1, 2)
    u10 = u_h[:, 1:, 0:-1, :]  # top-left
    u11 = u_h[:, 1:, 1:, :]  # top-right
    u01 = u_h[:, 0:-1, 1:, :]  # bottom-right
    dx = model_data["W"] / (model_data["grid_point_x"] - 1)
    dy = model_data["H"] / (model_data["grid_point_y"] - 1)
    U = torch.stack([
        torch.stack([u00, u01], dim=0),
        torch.stack([u10, u11], dim=0),
    ], dim=0)   # (2, 2, b, H-1, W-1, 2)
    xgrid = torch.tensor(
        [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)],
        dtype=torch.float64, device=model_data["device"]
    )
    Y, X = torch.meshgrid(xgrid, xgrid, indexing='ij')
    w1d = torch.tensor(
        [5/9, 8/9, 5/9],
        dtype=torch.float64, device=model_data["device"]
    )

    W = (w1d[:, None] * w1d[None, :]).view(3, 3, 1, 1, 1, 1)

    dudx, dudy = d_shape_functions_Q4(U, Y, X, dx, dy)   # (3, 3, batch, H-1, W-1, 2)
    Sigma_list = get_P(dudx, dudy, model_data)  # (3, 3, batch, H-1, W-1, 2, 2)
    V = torch.zeros(2, 2, 4, 1, 1, 1).to(model_data["device"])   # nodal test-function basis

    V[0, 0, 0, 0, 0, 0] = 1.0  # bottom-left
    V[0, 1, 1, 0, 0, 0] = 1.0  # bottom-right
    V[1, 0, 2, 0, 0, 0] = 1.0  # top-left
    V[1, 1, 3, 0, 0, 0] = 1.0  # top-right
    dvdx, dvdy = d_shape_functions_Q4(V, Y, X, dx, dy)   # (3, 3, 4, 1, 1, 1)

    # Residual of test function v, shape (batch, H-1, W-1, 2)
    res_v = torch.zeros_like(u_h)
    # P : nabla(v)
    # i=0, j=0
    res_v[:, 0:-1, 0:-1, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 0:1] + Sigma_list[..., 1] * dvdy[:, :, 0:1]) * W, dim=(0, 1)
    )
    # i=0, j=1
    res_v[:, 0:-1, 1:, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 1:2] + Sigma_list[..., 1] * dvdy[:, :, 1:2]) * W, dim=(0, 1)
    )
    # i=1, j=0
    res_v[:, 1:, 0:-1, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 2:3] + Sigma_list[..., 1] * dvdy[:, :, 2:3]) * W, dim=(0, 1)
    )
    # i=1, j=1
    res_v[:, 1:, 1:, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 3:4] + Sigma_list[..., 1] * dvdy[:, :, 3:4]) * W, dim=(0, 1)
    )

    # y n · v on interior faces
    yh00 = y_h[:, 0:-1, 0:-1, :]
    yh01 = y_h[:, 0:-1, 1:, :]
    yh10 = y_h[:, 1:, 0:-1, :]
    yh11 = y_h[:, 1:, 1:, :]
    Yh_ij = torch.stack([
        torch.stack([yh00, yh01], dim=0),
        torch.stack([yh10, yh11], dim=0),
    ], dim=0)   # (2, 2, batch, H-1, W-1, 4)
    # Vertical faces
    Y_r, X_r = torch.meshgrid(xgrid, torch.ones_like(xgrid[0]), indexing='ij')
    Yh_list = shape_functions_Q4(Yh_ij[:, :, :, :, :, :], Y_r, X_r)   # (3, 1, batch, H-1, W-1, 4)

    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 1, batch, H-1, W-1, 2, 2)

    Yh_10 = Yh_bound_list[0, 0][..., 0]  # (batch, H-1, W-1, 2)
    Yh_20 = Yh_bound_list[1, 0][..., 0]
    Yh_30 = Yh_bound_list[2, 0][..., 0]
    # i=0, j=0
    res_v[:, 0:-1, 1:, :] += mask_Bound_V * dy/2 * (
        w1d[0] * Yh_10 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 - np.sqrt(15)/10)
    )
    # i=1, j=0
    res_v[:, 1:, 1:, :] += mask_Bound_V * dy/2 * (
        w1d[0] * Yh_10 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 + np.sqrt(15)/10)
    )
    # Horizontal faces (padded rows)
    Y_d, X_d = torch.meshgrid(-torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    Y_u, X_u = torch.meshgrid(torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    # down
    Yh_list_d = shape_functions_Q4(Yh_ij[:, :, :, :, :, :], Y_d, X_d) # (1, 3, batch, H-1, W-1, 4)
    Yh_list_u = shape_functions_Q4(Yh_ij[:, :, :, -1:, :, :], Y_u, X_u) # (1, 3, batch, 1, W-1, 4)
    Yh_list = torch.cat([Yh_list_d, Yh_list_u], dim=3)  # (1, 3, batch, H, W-1, 4)
    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (1, 3, batch, H, W-1, 2, 2)
    
    Yh_01 = Yh_bound_list[0, 0][..., 1]  # (batch, H-2, W-1, 2)
    Yh_02 = Yh_bound_list[0, 1][..., 1]
    Yh_03 = Yh_bound_list[0, 2][..., 1]
    # i=0, j=0
    res_v[:, :, 0:-1, :] += mask_Bound_H * dx/2 * (
        w1d[0] * Yh_01 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 - np.sqrt(15)/10)
    )
    # i=0, j=1
    res_v[:, :, 1:, :] += mask_Bound_H * dx/2 * (
        w1d[0] * Yh_01 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 + np.sqrt(15)/10)
    )

    # Traction g · v, Simpson rule
    res_v[:, 0:-1, -1, :] -= mask_Bound_g * (g_h[:, 0:-1, -1, :] * dy / 6 + (g_h[:, 0:-1, -1, :] + g_h[:, 1:, -1, :]) * dy / 6)
    res_v[:, 1:, -1, :] -= mask_Bound_g * (g_h[:, 1:, -1, :] * dy / 6 + (g_h[:, 0:-1, -1, :] + g_h[:, 1:, -1, :]) * dy / 6)

    res_v[:, :, 0, :] = 0

    res_v = torch.where(mask_Node_Cut_Cell>0, model_data["lambda_vcut"] * res_v, res_v)
    loss_vij = res_v ** 2
    loss_v = torch.sum(loss_vij) / len(u_h)

    # ||Y + P(F)||^2 on cut cells
    Yh_list = shape_functions_Q4(Yh_ij, Y, X)   # (3, 3, batch, H-1, W-1, 4)
    Yh_int_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 3, batch, H-1, W-1, 2, 2)

    loss_yij = torch.sum(gamma_u * mask_Cut_Cell.unsqueeze(-1) * (dx/2) * (dy/2) * torch.sum(
        (Yh_int_list + Sigma_list) ** 2 * W.unsqueeze(-1), dim=(0, 1)
    ), dim=(-2, -1))  # (batch, H-1, W-1)
    loss_y = torch.sum(loss_yij) / len(u_h)

    # ||Y nabla phi + p phi / h||^2 on cut cells
    Phi = stack_Q4_data(phi_h)   # (2, 2, batch, H-1, W-1, 1)
    P = stack_Q4_data(p_h)   # (2, 2, batch, H-1, W-1, 2)
    dphidx, dphidy = d_shape_functions_Q4(Phi, Y, X, dx, dy) # (3, 3, batch, H-1, W-1, 1)
    eps = 1e-8
    dphi = torch.sqrt(dphidx ** 2 + dphidy ** 2) + eps  # (3, 3, batch, H-1, W-1, 1)
    phi_list = shape_functions_Q4(Phi, Y, X)    # (3, 3, batch, H-1, W-1, 1)
    p_list = shape_functions_Q4(P, Y, X)    # (3, 3, batch, H-1, W-1, 2)
    loss_pij = gamma_p * mask_Cut_Cell / 4 * torch.sum(
        (Yh_int_list[..., 0] * dphidx + Yh_int_list[..., 1] * dphidy + 1/dx * p_list * phi_list) ** 2 * W,
        dim=(0, 1)
    )# (b, H-1, W-1, 2)
    loss_p = torch.sum(loss_pij) / len(u_h)

    # ||div y||^2 on cut cells
    dYdx, dYdy = d_shape_functions_Q4(Yh_ij, Y, X, dx, dy) # (3, 3, batch, H-1, W-1, 4)
    res_dy = dYdx[..., 0::2] + dYdy[..., 1::2]    # (3, 3, batch, H-1, W-1, 2)
    loss_dyij = gamma_div * mask_Cut_Cell * (dx/2) * (dy/2) * torch.sum(res_dy ** 2 * W, dim=(0, 1))
    loss_dy = torch.sum(loss_dyij) / len(u_h)

    # Exterior (void) Laplace residual
    laplace_in = (
        u_h[:, 2:, 1:-1] + u_h[:, :-2, 1:-1] + u_h[:, 1:-1, 2:] + u_h[:, 1:-1, :-2]
        - 4 * u_h[:, 1:-1, 1:-1]
    ) / (dx ** 2)  # (b, H-2, W-2, 2)
    loss_inij = (1 - mask_laplace[:, 1:-1, 1:-1].double()) * (laplace_in ** 2).sum(dim=-1, keepdim=True)

    g_real = g_h[:, :, -1:, :]
    g_mean = g_real.norm(dim=-1).mean(dim=(1, 2), keepdim=True).unsqueeze(-1) ** 1
    loss_vij /= g_mean
    loss_yij /= g_mean.squeeze(-1)
    loss_pij /= g_mean
    loss_dyij /= g_mean

    return loss_vij, loss_yij, loss_pij, loss_dyij

# E_in / E_out vessel geometry with follower traction
def get_loss_hyperelasticity_vessel(u_h, y_h, p_h, Ein, Eout, Edx, Edy, g_h, model_data):
    torch.set_printoptions(threshold=torch.inf)
    u_h = u_h[:, ::2, ::2]  # (b, H, W, 2)
    y_h = y_h[:, ::2, ::2]
    p_h = p_h[:, ::2, ::2]
    g_h = g_h[:, ::2, ::2]
    phi_h = Ein * Eout
    phi_grid = phi_h[:, ::2, ::2]

    gamma_div, gamma_u, gamma_p = 1, 1, 1
    mask_Cell_Omega, mask_Cut_Cell = get_cell_masks(phi_grid)
    mask_Cell_Omega_padded = F.pad(
        mask_Cell_Omega,
        (0, 0,   # no pad on the last (channel) axis
        1, 1,   # pad 1 column on the right of W
        1, 1),  # pad 1 row on top and bottom of H
        mode='constant',
        value=0
    )   # (b, H+1, W+1, 1)
    mask_Bound_V, mask_Bound_H = get_boundary_masks(mask_Cell_Omega_padded)
    mask_Bound_V = mask_Bound_V[:, 1:-1]    # drop padded rows, (b, H-1, W, 1)
    mask_Bound_H = mask_Bound_H[:, :, 1:-1]  # (b, H, W-1, 1)
    mask_Bound_V[:, :, 0] = 0
    mask_Bound_H[:, 0] = 0
    mask_Node_Cut_Cell = get_node_mask_from_cell_mask(mask_Cut_Cell)
    mask_node = get_node_masks(phi_grid)

    mask_Cut_g = get_mask_cut_g(mask_Cell_Omega)
    mask_Node_Outer_Cut = mask_Node_Cut_Cell - get_node_mask_from_cell_mask(mask_Cut_g)

    # Cell-corner displacements for Q4 gradients
    dx = model_data["W"] / (model_data["grid_point_x"] - 1)
    dy = model_data["H"] / (model_data["grid_point_y"] - 1)
    U = stack_Q4_data(u_h)   # (2, 2, b, H-1, W-1, 2)
    xgrid = torch.tensor(
        [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)],
        dtype=torch.float64, device=model_data["device"]
    )
    Y, X = torch.meshgrid(xgrid, xgrid, indexing='ij')
    w1d = torch.tensor(
        [5/9, 8/9, 5/9],
        dtype=torch.float64, device=model_data["device"]
    )

    W = (w1d[:, None] * w1d[None, :]).view(3, 3, 1, 1, 1, 1)

    dudx, dudy = d_shape_functions_Q4(U, Y, X, dx, dy)   # (3, 3, batch, H-1, W-1, 2)
    Sigma_list = get_P(dudx, dudy, model_data)  # (3, 3, batch, H-1, W-1, 2, 2)
    V = torch.zeros(2, 2, 4, 1, 1, 1).to(model_data["device"])   # nodal test-function basis

    V[0, 0, 0, 0, 0, 0] = 1.0  # bottom-left
    V[0, 1, 1, 0, 0, 0] = 1.0  # bottom-right
    V[1, 0, 2, 0, 0, 0] = 1.0  # top-left
    V[1, 1, 3, 0, 0, 0] = 1.0  # top-right
    dvdx, dvdy = d_shape_functions_Q4(V, Y, X, dx, dy)   # (3, 3, 4, 1, 1, 1)

    # Residual of test function v, shape (batch, H-1, W-1, 2)
    res_v = torch.zeros_like(u_h)
    # P : nabla(v)
    # i=0, j=0
    res_v[:, 0:-1, 0:-1, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 0:1] + Sigma_list[..., 1] * dvdy[:, :, 0:1]) * W, dim=(0, 1)
    )
    # i=0, j=1
    res_v[:, 0:-1, 1:, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 1:2] + Sigma_list[..., 1] * dvdy[:, :, 1:2]) * W, dim=(0, 1)
    )
    # i=1, j=0
    res_v[:, 1:, 0:-1, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 2:3] + Sigma_list[..., 1] * dvdy[:, :, 2:3]) * W, dim=(0, 1)
    )
    # i=1, j=1
    res_v[:, 1:, 1:, :] += mask_Cell_Omega * (dx/2) * (dy/2) * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 3:4] + Sigma_list[..., 1] * dvdy[:, :, 3:4]) * W, dim=(0, 1)
    )

    # y n · v on interior faces
    Yh_ij = stack_Q4_data(y_h)   # (2, 2, batch, H-1, W-1, 4)
    # Vertical faces
    Y_l, X_l = torch.meshgrid(xgrid, -torch.ones_like(xgrid[0]), indexing='ij')
    Y_r, X_r = torch.meshgrid(xgrid, torch.ones_like(xgrid[0]), indexing='ij')
    Yh_list_l = shape_functions_Q4(Yh_ij[:, :, :, :, :, :], Y_l, X_l)   # (3, 1, batch, H-1, W-1, 4)
    Yh_list_r = shape_functions_Q4(Yh_ij[:, :, :, :, -1:, :], Y_r, X_r)   # (3, 1, batch, H-1, 1, 4)
    Yh_list = torch.cat([Yh_list_l, Yh_list_r], dim=4)  # (3, 1, b, H-1, W, 4)

    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 1, batch, H-1, W-1, 2, 2)

    Yh_10 = Yh_bound_list[0, 0][..., 0]  # (batch, H-1, W-1, 2)
    Yh_20 = Yh_bound_list[1, 0][..., 0]
    Yh_30 = Yh_bound_list[2, 0][..., 0]
    # i=0, j=0
    res_v[:, 0:-1, :, :] += mask_Bound_V * dy/2 * (
        w1d[0] * Yh_10 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 - np.sqrt(15)/10)
    )
    # i=1, j=0
    res_v[:, 1:, :, :] += mask_Bound_V * dy/2 * (
        w1d[0] * Yh_10 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_20 * 1/2 +
        w1d[2] * Yh_30 * (1/2 + np.sqrt(15)/10)
    )
    # Horizontal faces
    Y_d, X_d = torch.meshgrid(-torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    Y_u, X_u = torch.meshgrid(torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    Yh_list_d = shape_functions_Q4(Yh_ij[:, :, :, :, :, :], Y_d, X_d) # (1, 3, batch, H-1, W-1, 4)
    Yh_list_u = shape_functions_Q4(Yh_ij[:, :, :, -1:, :, :], Y_u, X_u) # (1, 3, batch, 1, W-1, 4)
    Yh_list = torch.cat([Yh_list_d, Yh_list_u], dim=3)  # (1, 3, batch, H, W-1, 4)
    Yh_bound_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (1, 3, batch, H-1, W-1, 2, 2)
    
    Yh_01 = Yh_bound_list[0, 0][..., 1]  # (batch, H-1, W-1, 2)
    Yh_02 = Yh_bound_list[0, 1][..., 1]
    Yh_03 = Yh_bound_list[0, 2][..., 1]
    # i=0, j=0
    res_v[:, :, 0:-1, :] += mask_Bound_H * dx/2 * (
        w1d[0] * Yh_01 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 - np.sqrt(15)/10)
    )
    # i=0, j=1
    res_v[:, :, 1:, :] += mask_Bound_H * dx/2 * (
        w1d[0] * Yh_01 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * Yh_02 * 1/2 +
        w1d[2] * Yh_03 * (1/2 + np.sqrt(15)/10)
    )

    # Dirichlet boundary-value problem
    # x=0, ux=0
    res_v[:, :, 0, 0] = 0
    # y=0, uy=0
    res_v[:, 0, :, 1] = 0
    res_v = torch.where(mask_Node_Outer_Cut>0, model_data["lambda_vcut"] * res_v, res_v)
    loss_vij = res_v ** 2
    loss_v = torch.sum(loss_vij) / len(u_h)

    # ||Y + P(F)||^2 on cut cells
    Yh_list = shape_functions_Q4(Yh_ij, Y, X)   # (3, 3, batch, H-1, W-1, 4)
    Yh_int_list = torch.stack([
        torch.stack([Yh_list[..., 0], Yh_list[..., 1]], dim=-1),
        torch.stack([Yh_list[..., 2], Yh_list[..., 3]], dim=-1) 
    ], dim=-2)  # (3, 3, batch, H-1, W-1, 2, 2)

    loss_yij = torch.sum(gamma_u * mask_Cut_Cell.unsqueeze(-1) * (dx/2) * (dy/2) * torch.sum(
        (Yh_int_list + Sigma_list) ** 2 * W.unsqueeze(-1), dim=(0, 1)
    ), dim=(-2, -1))  # (batch, H-1, W-1)
    loss_y = torch.sum(loss_yij) / len(u_h)

    # ||Y nabla phi + p phi / h + g |nabla phi||| ^2 on cut cells
    Phi = stack_Q9_data(phi_h)  # (3, 3, batch, H-1, W-1, 1)
    P = stack_Q4_data(p_h)   # (2, 2, batch, H-1, W-1, 2)
    G = stack_Q4_data(g_h)   # (3, 3, batch, H-1, W-1, 1)
    G = G * mask_Cut_g.unsqueeze(0).unsqueeze(0)
    dphidx = Ein * Edx[..., 1:2] + Edx[..., 0:1] * Eout # (b, 2H-1, 2W-1, 1)
    dphidy = Ein * Edy[..., 1:2] + Edy[..., 0:1] * Eout # (b, 2H-1, 2W-1, 1)
    dPhidx = stack_Q9_data(dphidx)  # (3, 3, b, H-1, W-1, 1)
    dPhidy = stack_Q9_data(dphidy)  # (3, 3, b, H-1, W-1, 1)
    norm_dphi = torch.sqrt(dPhidx ** 2 + dPhidy ** 2)   # (3, 3, batch, H-1, W-1, 1)
    phi_list = shape_functions_Q9(Phi, Y, X)    # (3, 3, batch, H-1, W-1, 1)
    p_list = shape_functions_Q4(P, Y, X)    # (3, 3, batch, H-1, W-1, 2)
    g_list = shape_functions_Q4(G, Y, X)    # (3, 3, batch, H-1, W-1, 2)
    dphi = torch.cat([dPhidx, dPhidy], dim=-1)  # (3, 3, b, H-1, W-1, 2)

    # Pressure / traction residual
    F_u = torch.stack([
        torch.stack([1 + dudx[..., 0], dudy[..., 0]], dim=-1),
        torch.stack([dudx[..., 1], 1 + dudy[..., 1]], dim=-1) 
    ], dim=-2)  # (3, 3, batch, H-1, W-1, 2, 2)
    J = torch.det(F_u).unsqueeze(-1)  # (3, 3, batch, H-1, W-1, 1)
    # t * |nabla phi|
    tphi_list = -g_list * J * torch.einsum('...ij,...j->...i', torch.inverse(F_u).transpose(-1, -2), dphi)  # (3, 3, batch, H-1, W-1, 2)
    loss_pij = gamma_p * mask_Cut_Cell / 4 * torch.sum(
        (Yh_int_list[..., 0] * dPhidx + Yh_int_list[..., 1] * dPhidy 
        + 1/dx * p_list * phi_list + tphi_list) ** 2 * W,
        dim=(0, 1)
    )   # (b, H-1, W-1, 2)
    loss_p = torch.sum(loss_pij) / len(u_h)

    # ||div y||^2 on cut cells
    dYdx, dYdy = d_shape_functions_Q4(Yh_ij, Y, X, dx, dy) # (3, 3, batch, H-1, W-1, 4)
    res_dy = dYdx[..., 0::2] + dYdy[..., 1::2]    # (3, 3, batch, H-1, W-1, 2)
    loss_dyij = gamma_div * mask_Cut_Cell * (dx/2) * (dy/2) * torch.sum(res_dy ** 2 * W, dim=(0, 1))
    loss_dy = torch.sum(loss_dyij) / len(u_h)

    g_mean = g_h[:, ::2, ::2].norm(dim=-1).mean(dim=(1, 2), keepdim=True).unsqueeze(-1) ** 2
    loss_vij /= g_mean
    loss_yij /= g_mean.squeeze(-1)
    loss_pij /= g_mean
    loss_dyij /= g_mean

    return loss_vij, loss_yij, loss_pij, loss_dyij

def get_loss_hyperelasticity_DBC(u_h, phi_h, f_h, model_data):
    mask_Cell_Omega, mask_Cut_Cell = get_cell_masks(phi_h)
    mask_Bound_V, mask_Bound_H = get_boundary_masks(mask_Cell_Omega)
    mask_Node_Cut_Cell = get_node_mask_from_cell_mask(mask_Cut_Cell)
    mask_node = get_node_masks(phi_h)
    torch.set_printoptions(threshold=torch.inf)

    # Cell-corner displacements for Q4 gradients
    u00 = u_h[:, 0:-1, 0:-1, :]  # bottom-left, (batch, H-1, W-1, 2)
    u10 = u_h[:, 1:, 0:-1, :]  # top-left
    u11 = u_h[:, 1:, 1:, :]  # top-right
    u01 = u_h[:, 0:-1, 1:, :]  # bottom-right
    h = 1 / (model_data["grid_point_num"] - 1)
    U = torch.stack([
        torch.stack([u00, u01], dim=0),
        torch.stack([u10, u11], dim=0),
    ], dim=0)   # (2, 2, b, H-1, W-1, 2)
    xgrid = torch.tensor(
        [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)],
        dtype=torch.float64, device=model_data["device"]
    )
    Y, X = torch.meshgrid(xgrid, xgrid, indexing='ij')
    w1d = torch.tensor(
        [5/9, 8/9, 5/9],
        dtype=torch.float64, device=model_data["device"]
    )

    W = (w1d[:, None] * w1d[None, :]).view(3, 3, 1, 1, 1, 1)

    dudx, dudy = d_shape_functions_Q4(U, Y, X, h, h)   # (3, 3, batch, H-1, W-1, 2)
    Sigma_list = get_P(dudx, dudy, model_data)  # (3, 3, batch, H-1, W-1, 2, 2)
    V = torch.zeros(2, 2, 4, 1, 1, 1).to(model_data["device"])   # nodal test-function basis

    V[0, 0, 0, 0, 0, 0] = 1.0  # bottom-left
    V[0, 1, 1, 0, 0, 0] = 1.0  # bottom-right
    V[1, 0, 2, 0, 0, 0] = 1.0  # top-left
    V[1, 1, 3, 0, 0, 0] = 1.0  # top-right
    dvdx, dvdy = d_shape_functions_Q4(V, Y, X, h, h)   # (3, 3, 4, 1, 1, 1)

    # Residual of test function v, shape (batch, H-1, W-1, 2)
    res_v = torch.zeros_like(u_h)
    # P : nabla(v)
    # Use the same spacing in x and y (dx = dy = h)
    # i=0, j=0
    res_v[:, 0:-1, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 0:1] + Sigma_list[..., 1] * dvdy[:, :, 0:1]) * W, dim=(0, 1)
    )
    # i=0, j=1
    res_v[:, 0:-1, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 1:2] + Sigma_list[..., 1] * dvdy[:, :, 1:2]) * W, dim=(0, 1)
    )
    # i=1, j=0
    res_v[:, 1:, 0:-1, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 2:3] + Sigma_list[..., 1] * dvdy[:, :, 2:3]) * W, dim=(0, 1)
    )
    # i=1, j=1
    res_v[:, 1:, 1:, :] += mask_Cell_Omega * (h/2) ** 2 * torch.sum(
        (Sigma_list[..., 0] * dvdx[:, :, 3:4] + Sigma_list[..., 1] * dvdy[:, :, 3:4]) * W, dim=(0, 1)
    )

    # y n · v on interior faces
    # Vertical faces
    Y_l, X_l = torch.meshgrid(xgrid, -torch.ones_like(xgrid[0]), indexing='ij')
    Y_r, X_r = torch.meshgrid(xgrid, torch.ones_like(xgrid[0]), indexing='ij')
    # left
    dudx_l, dudy_l = d_shape_functions_Q4(U[:, :, :, :, 1:, :], Y_l, X_l, h, h)
    # right
    dudx_r, dudy_r = d_shape_functions_Q4(U[:, :, :, :, :-1, :], Y_r, X_r, h, h)
    # mask_Bound_V = left_cell - right_cell:
    #   >0: left cell in Omega, use left traces; <0: right cell in Omega, use right traces.
    dudx = torch.where(mask_Bound_V.unsqueeze(0).unsqueeze(0) > 0, dudx_l, dudx_r)
    dudy = torch.where(mask_Bound_V.unsqueeze(0).unsqueeze(0) > 0, dudy_l, dudy_r)
    S_bound_list = get_P(dudx, dudy, model_data)  # (3, 1, batch, H-1, W-2, 2, 2)
    S_10 = S_bound_list[0, 0][..., 0]  # (batch, H-1, W-2, 2)
    S_20 = S_bound_list[1, 0][..., 0]
    S_30 = S_bound_list[2, 0][..., 0]
    # i=0, j=0
    res_v[:, 0:-1, 1:-1, :] -= mask_Bound_V * h/2 * (
        w1d[0] * S_10 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * S_20 * 1/2 +
        w1d[2] * S_30 * (1/2 - np.sqrt(15)/10)
    )
    # i=1, j=0
    res_v[:, 1:, 1:-1, :] -= mask_Bound_V * h/2 * (
        w1d[0] * S_10 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * S_20 * 1/2 +
        w1d[2] * S_30 * (1/2 + np.sqrt(15)/10)
    )
    # Horizontal faces
    Y_d, X_d = torch.meshgrid(-torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    Y_u, X_u = torch.meshgrid(torch.ones_like(xgrid[0]), xgrid, indexing='ij')
    # down
    dudx_d, dudy_d = d_shape_functions_Q4(U[:, :, :, 1:, :, :], Y_d, X_d, h, h)
    # up
    dudx_u, dudy_u = d_shape_functions_Q4(U[:, :, :, :-1, :, :], Y_u, X_u, h, h)
    # mask_Bound_H = down_cell - up_cell:
    #   >0: bottom cell in Omega, use bottom traces; <0: top cell in Omega, use top traces.
    dudx = torch.where(mask_Bound_H.unsqueeze(0).unsqueeze(0) > 0, dudx_d, dudx_u)
    dudy = torch.where(mask_Bound_H.unsqueeze(0).unsqueeze(0) > 0, dudy_d, dudy_u)
    S_bound_list = get_P(dudx, dudy, model_data)  # (1, 3, batch, H-1, W-2, 2, 2)
    S_01 = S_bound_list[0, 0][..., 1]  # (batch, H-2, W-1, 2)
    S_02 = S_bound_list[0, 1][..., 1]
    S_03 = S_bound_list[0, 2][..., 1]
    # i=0, j=0
    res_v[:, 1:-1, 0:-1, :] -= mask_Bound_H * h/2 * (
        w1d[0] * S_01 * (1/2 + np.sqrt(15)/10) +
        w1d[1] * S_02 * 1/2 +
        w1d[2] * S_03 * (1/2 - np.sqrt(15)/10)
    )
    # i=0, j=1
    res_v[:, 1:-1, 1:, :] -= mask_Bound_H * h/2 * (
        w1d[0] * S_01 * (1/2 - np.sqrt(15)/10) +
        w1d[1] * S_02 * 1/2 +
        w1d[2] * S_03 * (1/2 + np.sqrt(15)/10)
    )

    res_v[:, 0:-1, 0:-1, :] -= mask_Cell_Omega * f_h[:, 0:-1, 0:-1, :] / 4 * h ** 2
    res_v[:, 1:, 0:-1, :] -= mask_Cell_Omega * f_h[:, 1:, 0:-1, :] / 4 * h ** 2
    res_v[:, 0:-1, 1:, :] -= mask_Cell_Omega * f_h[:, 0:-1, 1:, :] / 4 * h ** 2
    res_v[:, 1:, 1:, :] -= mask_Cell_Omega * f_h[:, 1:, 1:, :] / 4 * h ** 2

    # Multiply residual by phi (v := phi * v)
    res_v *= phi_h

    loss_vij = res_v ** 2
    loss_v = torch.sum(loss_vij) / len(u_h)

    return loss_v

