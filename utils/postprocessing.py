#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting and post-processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:  # pragma: no cover
    RegularGridInterpolator = None


def _colorbar_decimal_formatter(n_decimals: int) -> FormatStrFormatter:
    """
    Colorbar ticks: fixed to ``n_decimals`` decimal places (e.g. 3 → resolution 0.001, 4 → 0.0001).
    Never use scientific notation (avoids forms like ``5e-5``).
    """
    n = int(n_decimals)
    if n < 0:
        raise ValueError("colorbar_decimals must be >= 0")
    return FormatStrFormatter(f"%.{n}f")


def _style_colorbar_fixed_decimals(cbar, n_decimals: int) -> None:
    """Apply fixed decimal places on the colorbar axis and disable offset/scientific multiplier display."""
    cbar.ax.yaxis.set_major_formatter(_colorbar_decimal_formatter(n_decimals))
    cbar.ax.yaxis.get_offset_text().set_visible(False)


def plot_pred(x, u_exact, u_pred, title):
    fig_font = "DejaVu Serif"
    fig_size = (5, 3)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact, 'b', label='Ground Truth', linewidth=1.5)
    plt.plot(x, u_pred, '-.r', label='Prediction', linewidth=1.5)
    plt.legend()
    plt.title(title)
    title = title.replace('\\', '').replace('$', '')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/' + title + '.png', dpi=300)
    plt.show()
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)

    error_title = f"Relative $L^2$ error = {rel_l2_error:.4f}"
    print(error_title)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact - u_pred, 'b', label='Ground Truth', linewidth=1.5)
    # plt.legend()
    plt.title(error_title)
    # plt.savefig('error_' + title + '.png', dpi=300)
    plt.show()


def plot_pred1(x, y, f, title):
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, f, levels=2, cmap='Purples')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Input ' + title)
    plt.savefig('Input ' + title + '.png', dpi=600)
    plt.show()


def plot_pred2(x, y, u_exact, u_pred, title, saved_title):
    saved_title = saved_title.replace('\\', '').replace('$', '')
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, u_pred, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.savefig(saved_title + ' - Approximate solution' + '.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Exact solution - ' + title)
    plt.savefig(saved_title + ' - Exact solution' + '.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact - u_pred, levels=500, cmap='bwr')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Error - ' + title)
    plt.savefig(saved_title + ' - Error' + '.png', dpi=600)
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)
    plt.show()


def plot_field_2d(
    F,
    L,
    W,
    title,
    folder=None,
    file=None,
    mask=None,
    isError=False,
    colorbar_decimals: int = 3,
):
    """
    Plots a 2D field stored in a 1D tensor F

    colorbar_decimals
        Number of colorbar decimal places; default 3 (same order as 0.001). Pass the same value across figures for a consistent style.
    """
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    fs = plt.rcParams["font.size"]
    num_pts_v = F.shape[0]
    num_pts_u = F.shape[1]
    x = np.linspace(0, L, num_pts_u)
    y = np.linspace(0, W, num_pts_v)
    x_2d, y_2d = np.meshgrid(x, y, indexing='xy')
    plt.figure()
    color_type = 'jet'
    if isError:
        color_type = 'bwr'

    if mask is None:
        plt.contourf(x_2d, y_2d, F, levels=512, cmap=color_type)
    else:
        masked_f = np.ma.array(F, mask=mask).copy()
        plt.contourf(x_2d, y_2d, masked_f, 255, cmap=color_type)
    # cbar = plt.colorbar(orientation='horizontal', pad=0.2, aspect=40)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    _style_colorbar_fixed_decimals(cbar, colorbar_decimals)
    # cbar.locator = ticker.MaxNLocator(nbins=8)
    # cbar.update_ticks()
    if title:
        plt.title(title)
    plt.gca().tick_params(labelsize=fs)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.subplots_adjust(top=0.5, bottom=0.2)
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + '/' + file
        plt.savefig(full_name + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_field_2d_exact(
    F,
    L,
    W,
    title,
    folder=None,
    file=None,
    mask=None,
    phi=None,
    phi_hole_positive=True,
    upsample=8,
    isError=False,
    error_nonnegative_clip=False,
    colorbar_decimals: int = 3,
    colorbar_horizontal: bool = False,
):
    """
    Plot a scalar field F on a regular grid after refinement + **linear** (bilinear) interpolation.

    Nodal values of ``F`` match the intra-element shape of bilinear quadrilateral elements (Q4):
    on a regular rectangular grid, scipy's
    ``RegularGridInterpolator(..., method="linear")`` is equivalent to bilinear extension
    over each rectangle spanned by adjacent nodes, with no cubic-interpolation overshoot inside cells.

    Same as ``plot_field_2d``: F has shape (Ny, Nx), corresponding to rows=y, cols=x under
    ``np.meshgrid(..., indexing='xy')``.

    Parameters
    ----------
    phi : (Ny, Nx), optional
        Level set (e.g. Hole phi: >0 inside the hole, <0 in the solid). After **linear interpolation
        of phi on the fine grid** as well, mask the hole with ``phi_fine > 0``
        (or the opposite if ``phi_hole_positive=False``); the boundary is a piecewise-linear
        curve after refinement, finer than a coarse binary mask, but not an analytically smooth
        boundary from cubic interpolation.
    mask : (Ny, Nx) bool, optional
        Used only when ``phi`` is None; linearly interpolated to the fine grid then thresholded at 0.5.
    upsample : int
        Number of subdivisions per coarse grid spacing (fine grid size is about ``(Ny-1)*upsample+1``).
    error_nonnegative_clip : bool
        If True, apply ``max(·, 0)`` to ``F_f`` after interpolation. Nonnegative nodal values are
        usually already preserved under linear interpolation; still useful as a safeguard in
        extreme cases combined with ``fill_value`` etc. Do not enable for signed errors.
    colorbar_decimals : int
        Fixed number of colorbar decimal places (no scientific notation); default 3.
        Displacement and error plots can share the same value for a consistent style.
    colorbar_horizontal : bool
        Whether to use a horizontal colorbar. Default False (vertical).
    """
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    fs = plt.rcParams["font.size"]
    F = np.asarray(F, dtype=np.float64)
    num_pts_v, num_pts_u = F.shape[0], F.shape[1]
    y_axis = np.linspace(0.0, W, num_pts_v)
    x_axis = np.linspace(0.0, L, num_pts_u)

    Hf = max((num_pts_v - 1) * int(upsample) + 1, num_pts_v)
    Wf = max((num_pts_u - 1) * int(upsample) + 1, num_pts_u)
    y_f = np.linspace(0.0, W, Hf)
    x_f = np.linspace(0.0, L, Wf)
    yy, xx = np.meshgrid(y_f, x_f, indexing="ij")

    if RegularGridInterpolator is not None:
        interp_kw = dict(bounds_error=False, fill_value=np.nan)
        iF = RegularGridInterpolator((y_axis, x_axis), F, method="linear", **interp_kw)
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        F_f = iF(pts).reshape(yy.shape)
        if phi is not None:
            phi = np.asarray(phi, dtype=np.float64)
            iphi = RegularGridInterpolator((y_axis, x_axis), phi, method="linear", **interp_kw)
            phi_f = iphi(pts).reshape(yy.shape)
            if phi_hole_positive:
                mask_f = phi_f > 0.0
            else:
                mask_f = phi_f < 0.0
        elif mask is not None:
            mask = np.asarray(mask, dtype=bool)
            imk = RegularGridInterpolator(
                (y_axis, x_axis), mask.astype(np.float64), method="linear", **interp_kw
            )
            mask_f = imk(pts).reshape(yy.shape) > 0.5
        else:
            mask_f = np.zeros_like(F_f, dtype=bool)
    else:
        try:
            from scipy.ndimage import zoom

            zy = (Hf - 1) / max(num_pts_v - 1, 1)
            zx = (Wf - 1) / max(num_pts_u - 1, 1)
            F_f = zoom(F, (zy, zx), order=1)
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                phi_f = zoom(phi, (zy, zx), order=1)
                if phi_hole_positive:
                    mask_f = phi_f > 0.0
                else:
                    mask_f = phi_f < 0.0
            elif mask is not None:
                mask_f = zoom(mask.astype(np.float64), (zy, zx), order=1) > 0.5
            else:
                mask_f = np.zeros_like(F_f, dtype=bool)
        except ImportError:
            F_f = F
            y_f, x_f = y_axis, x_axis
            if phi is not None:
                ph = np.asarray(phi, dtype=np.float64)
                mask_f = ph > 0.0 if phi_hole_positive else ph < 0.0
            elif mask is not None:
                mask_f = np.asarray(mask, dtype=bool)
            else:
                mask_f = np.zeros_like(F_f, dtype=bool)

    if error_nonnegative_clip:
        F_f = np.maximum(np.asarray(F_f, dtype=np.float64), 0.0)

    X, Y = np.meshgrid(x_f, y_f, indexing="xy")
    color_type = "jet"
    if isError:
        color_type = "bwr"
    F_plot = np.ma.array(F_f, mask=mask_f | np.isnan(F_f))

    plt.figure()
    ax = plt.gca()
    im = ax.pcolormesh(X, Y, F_plot, shading="gouraud", cmap=color_type)
    cbar_orientation = "horizontal" if colorbar_horizontal else "vertical"
    cbar = plt.colorbar(im, orientation=cbar_orientation)
    cbar.ax.tick_params(labelsize=fs)
    _style_colorbar_fixed_decimals(cbar, colorbar_decimals)
    if title:
        plt.title(title)
    ax.tick_params(labelsize=fs)
    ax.set_aspect("equal", adjustable="box")
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + "/" + file
        plt.savefig(full_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_u_matrix_displaced(
    u_mat,
    mask,
    title=None,
    scale=1.0,
    folder=None,
    file=None,
    upsample=8,
    show_plot=True,
    phi=None,
    phi_hole_positive=True,
):
    """
    u_mat: (H, W, 2) displacement; mask: (H, W) bool, True means cut out / hidden (consistent with masked_where).
    Interpolate and refine ux, uy on a regular grid, then apply Gouraud shading to avoid
    polyline cell edges from pcolormesh on a coarse deformed mesh.

    phi: optional, (H, W) continuous level set (e.g. phi in Hole data: >0 inside the hole, <0 in the solid).
    If given, the hole boundary is determined from cubically interpolated phi, yielding a smooth
    discrete approximation of an elliptical hole edge and avoiding staircasing from nearest-neighbor
    upsampling of a binary mask.
    When phi_hole_positive=True, mask the region where phi_fine > 0 (the hole); set False if your convention is the opposite.
    """
    ux = np.asarray(u_mat[..., 0], dtype=np.float64)
    uy = np.asarray(u_mat[..., 1], dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    H, W = ux.shape
    y_axis = np.linspace(0.0, 1.0, H)
    x_axis = np.linspace(0.0, 1.0, W)

    Hf = max((H - 1) * int(upsample) + 1, H)
    Wf = max((W - 1) * int(upsample) + 1, W)
    y_f = np.linspace(0.0, 1.0, Hf)
    x_f = np.linspace(0.0, 1.0, Wf)
    yy, xx = np.meshgrid(y_f, x_f, indexing="ij")

    if RegularGridInterpolator is not None:
        interp_kw = dict(bounds_error=False, fill_value=np.nan)
        iux = RegularGridInterpolator((y_axis, x_axis), ux, method="cubic", **interp_kw)
        iuy = RegularGridInterpolator((y_axis, x_axis), uy, method="cubic", **interp_kw)
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        ux_f = iux(pts).reshape(yy.shape)
        uy_f = iuy(pts).reshape(yy.shape)
        if phi is not None:
            phi = np.asarray(phi, dtype=np.float64)
            iphi = RegularGridInterpolator((y_axis, x_axis), phi, method="cubic", **interp_kw)
            phi_f = iphi(pts).reshape(yy.shape)
            if phi_hole_positive:
                mask_f = phi_f > 0.0
            else:
                mask_f = phi_f < 0.0
        else:
            imk = RegularGridInterpolator(
                (y_axis, x_axis), mask.astype(np.float64), method="linear", **interp_kw
            )
            mask_f = imk(pts).reshape(yy.shape) > 0.5
    else:
        try:
            from scipy.ndimage import zoom

            zy = (Hf - 1) / max(H - 1, 1)
            zx = (Wf - 1) / max(W - 1, 1)
            ux_f = zoom(ux, (zy, zx), order=3)
            uy_f = zoom(uy, (zy, zx), order=3)
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                phi_f = zoom(phi, (zy, zx), order=3)
                if phi_hole_positive:
                    mask_f = phi_f > 0.0
                else:
                    mask_f = phi_f < 0.0
            else:
                mask_f = zoom(mask.astype(np.float64), (zy, zx), order=1) > 0.5
        except ImportError:
            yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
            ux_f, uy_f, mask_f = ux, uy, mask

    X_deformed = xx + scale * ux_f
    Y_deformed = yy + scale * uy_f
    mag = np.sqrt(ux_f ** 2 + uy_f ** 2)
    mag_masked = np.ma.array(mag, mask=np.isnan(ux_f) | np.isnan(uy_f) | mask_f)

    plt.figure(figsize=(7, 8))
    ax = plt.gca()
    im = ax.pcolormesh(
        X_deformed,
        Y_deformed,
        mag_masked,
        cmap="jet",
        shading="gouraud",
    )
    ax.set_axis_off()
    cbar = plt.colorbar(
        im,
        orientation="horizontal",
        fraction=0.04,
        pad=0.05,
        shrink=0.6,
        aspect=35,
    )
    cbar.set_label("Displacement Magnitude", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1.9)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.gca().set_aspect("equal")
    if title:
        plt.title(title)
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + "/" + file
        plt.savefig(full_name + ".png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_u_matrix_displaced_custom_domain(
    u_mat,
    mask,
    title=None,
    scale=1.0,
    folder=None,
    file=None,
    upsample=8,
    show_plot=True,
    phi=None,
    phi_hole_positive=True,
    x_domain=(-1.0, 1.0),
    y_domain=(-1.0, 1.0),
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
):
    """
    Same as plot_u_matrix_displaced, but allows a custom base grid domain and display range.
    Suitable for visualizing the full domain after mirroring and stitching a 1/4 structure.
    """
    ux = np.asarray(u_mat[..., 0], dtype=np.float64)
    uy = np.asarray(u_mat[..., 1], dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    H, W = ux.shape
    x0, x1 = float(x_domain[0]), float(x_domain[1])
    y0, y1 = float(y_domain[0]), float(y_domain[1])
    y_axis = np.linspace(y0, y1, H)
    x_axis = np.linspace(x0, x1, W)

    Hf = max((H - 1) * int(upsample) + 1, H)
    Wf = max((W - 1) * int(upsample) + 1, W)
    y_f = np.linspace(y0, y1, Hf)
    x_f = np.linspace(x0, x1, Wf)
    yy, xx = np.meshgrid(y_f, x_f, indexing="ij")

    if RegularGridInterpolator is not None:
        interp_kw = dict(bounds_error=False, fill_value=np.nan)
        iux = RegularGridInterpolator((y_axis, x_axis), ux, method="cubic", **interp_kw)
        iuy = RegularGridInterpolator((y_axis, x_axis), uy, method="cubic", **interp_kw)
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        ux_f = iux(pts).reshape(yy.shape)
        uy_f = iuy(pts).reshape(yy.shape)
        if phi is not None:
            phi = np.asarray(phi, dtype=np.float64)
            iphi = RegularGridInterpolator((y_axis, x_axis), phi, method="cubic", **interp_kw)
            phi_f = iphi(pts).reshape(yy.shape)
            if phi_hole_positive:
                mask_f = phi_f > 0.0
            else:
                mask_f = phi_f < 0.0
        else:
            imk = RegularGridInterpolator(
                (y_axis, x_axis), mask.astype(np.float64), method="linear", **interp_kw
            )
            mask_f = imk(pts).reshape(yy.shape) > 0.5
    else:
        try:
            from scipy.ndimage import zoom

            zy = (Hf - 1) / max(H - 1, 1)
            zx = (Wf - 1) / max(W - 1, 1)
            ux_f = zoom(ux, (zy, zx), order=3)
            uy_f = zoom(uy, (zy, zx), order=3)
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                phi_f = zoom(phi, (zy, zx), order=3)
                if phi_hole_positive:
                    mask_f = phi_f > 0.0
                else:
                    mask_f = phi_f < 0.0
            else:
                mask_f = zoom(mask.astype(np.float64), (zy, zx), order=1) > 0.5
        except ImportError:
            yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
            ux_f, uy_f, mask_f = ux, uy, mask

    X_deformed = xx + scale * ux_f
    Y_deformed = yy + scale * uy_f
    mag = np.sqrt(ux_f ** 2 + uy_f ** 2)
    mag_masked = np.ma.array(mag, mask=np.isnan(ux_f) | np.isnan(uy_f) | mask_f)

    plt.figure(figsize=(7, 8))
    ax = plt.gca()
    im = ax.pcolormesh(
        X_deformed,
        Y_deformed,
        mag_masked,
        cmap="jet",
        shading="gouraud",
    )
    ax.set_axis_off()
    cbar = plt.colorbar(
        im,
        orientation="horizontal",
        fraction=0.04,
        pad=0.05,
        shrink=0.9,
        aspect=35,
    )
    cbar.set_label("Displacement Magnitude", fontsize=12)
    plt.xlim(float(xlim[0]), float(xlim[1]))
    plt.ylim(float(ylim[0]), float(ylim[1]))
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.gca().set_aspect("equal")
    if title:
        plt.title(title)
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + "/" + file
        plt.savefig(full_name + ".png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

