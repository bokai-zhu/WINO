#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
import time
import os
import sys
_WINO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _WINO_ROOT not in sys.path:
    sys.path.insert(0, _WINO_ROOT)
_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_CASE_DIR, "model")
_PICTURES_DIR = os.path.join(_CASE_DIR, "pictures")
_RESULT_DIR = os.path.join(_CASE_DIR, "result")

from Phi_FEM.generate_data_Vessel import compute_errors
from torch.nn.modules import fold
from torch.utils.data import DataLoader, random_split, Subset
from utils.postprocessing import (
    plot_field_2d_exact,
    plot_u_matrix_displaced,
    plot_u_matrix_displaced_custom_domain,
)
from utils.fno_2d import FNO2d
from utils.database_makers import HyperelasticityDataset
from utils.fno_utils import *
from utils.soap import SOAP
from transformers import get_cosine_schedule_with_warmup
# from utils.scheduler import LR_Scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
torch.set_default_dtype(torch.float64)
os.environ["PYTHONUNBUFFERED"] = "1"

# torch.manual_seed(42)
# np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("this is the device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
################################################################
#  configurations
################################################################
class FNO2d_HyperElasticity_Vessel(FNO2d):
    def __init__(self, model_data, normalizers=None):
        super(FNO2d_HyperElasticity_Vessel, self).__init__(model_data)
        if normalizers is not None:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def forward(self, x):
        out = super().forward(x)

        if model_data["normalized"] is True:
            x = self.normalizer_x.decode(x)
            out[..., 0:2] = self.normalizer_y.decode(out[..., 0:2])
        # u0*x, u1*y
        x_coords = torch.linspace(0, 1, out.shape[2]).to(model_data["device"])
        x_grid = x_coords.reshape(1, 1, out.shape[2], 1)

        y_coords = torch.linspace(0, 1, out.shape[1]).to(model_data["device"])
        y_grid = y_coords.reshape(1, out.shape[1], 1, 1)
        out[..., 0:1] = out[..., 0:1] * x_grid
        out[..., 1:2] = out[..., 1:2] * y_grid
        return out

# Settings dictionary
model_data = dict()
model_data["device"] = device
model_data["n_train"] = 1000
model_data["n_test"] = 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["batch_size"] = 50
model_data["learning_rate_adam"] = 2e-3
model_data["num_epoch"] = 2000
model_data["gamma"] = 0.7
model_data["patience"] = 250
model_data["min_lr"] = 2e-4
model_data["step_size"] = 25
model_data["E"] = 200
model_data["nu"] = 0.3
model_data["mu"] = model_data["E"] / (2 * (1 + model_data["nu"]))
model_data["lambda"] = model_data["E"] * model_data["nu"] / ((1 + model_data["nu"]) * (1 - 2 * model_data["nu"]))
model_data["W"] = 1     # width
model_data["H"] = 1     # height
model_data["grid_point_num"] = 51
model_data["grid_point_num_test"] = 256
model_data["sensor_point_num"] = model_data["grid_point_num"] * 2 - 1
model_data["sensor_point_num_test"] = model_data["grid_point_num_test"] * 2 - 1
model_data["grid_point_x"] = model_data["grid_point_num"]
model_data["grid_point_y"] = model_data["grid_point_num"]
model_data["fno"] = dict()
model_data["fno"]["modes"] = 16
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["input"] = 9
model_data["fno"]["output"] = 8
model_data["fno"]["use_data"] = True
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["normalized"] = True
model_data["dir"] = os.path.join(_WINO_ROOT, 'data')
model_data["path_train"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Vessel_PhiEdxEdyG_u_s' + str(model_data["sensor_point_num"])
                                  + '_n' + str(model_data["n_train"]) + '_train.npz')
if model_data["grid_point_num"] == model_data["grid_point_num_test"]:
    model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Vessel_PhiEdxEdyG_u_s' + str(model_data["sensor_point_num"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
else:
    model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Vessel_PhiEdxEdyG_u_s' + str(model_data["sensor_point_num_test"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
model_data["model_filename"] = "WINO_hyperelasticityVessel_s{}_n{}_ep{}.pth".format(
    model_data["grid_point_num"],
    model_data["n_data"],
    model_data["num_epoch"]
)
plot_fig = True



os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_PICTURES_DIR, exist_ok=True)
os.makedirs(_RESULT_DIR, exist_ok=True)

if model_data["grid_point_num"] // 2 + 1 < model_data["fno"]["modes"]:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

t_start = time.time()
#################################################################
# generate the data
#################################################################
train_dataset = HyperelasticityDataset(model_data, path=model_data["path_train"])
normalizers = (
    [train_dataset.normalizer_x, train_dataset.normalizer_y]
    if model_data["normalized"] is True
    else None
)
test_dataset = HyperelasticityDataset(
    model_data,
    path=model_data["path_test"],
    normalizers=normalizers,
)
# Data convention: npz is a Q9 nodal grid s=2*nb_vert-1 (e.g. nb_vert=201 → 401×401); evaluate on ::2 to get nb_vert×nb_vert
_test_h, _test_w = test_dataset.x.shape[1], test_dataset.x.shape[2]
_test_vert = _test_h // 2 + 1
print(
    f"test set npz grid: H×W={_test_h}×{_test_w} "
    f"(sensor s={model_data['sensor_point_num_test']}); "
    f"vertex grid after ::2 ≈ {_test_vert}×{_test_vert}, "
    f"compute_errors nb_cell={_test_vert - 1}"
)
if _test_vert != model_data["grid_point_num_test"]:
    print(
        f"warning: grid_point_num_test={model_data['grid_point_num_test']} "
        f"does not match inferred vertex count {_test_vert}; check npz or config."
    )

# Making dataloaders
t_start = time.time()
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=model_data["batch_size"],
#     shuffle=True,
#     drop_last=True
# )
test_loader = DataLoader(
    test_dataset,
    batch_size=model_data["batch_size"],
    shuffle=True,
)

model = FNO2d_HyperElasticity_Vessel(model_data, normalizers).to(device)
myLoss = LpLoss(d=1, size_average=False)
save_path = os.path.join(_MODEL_DIR, model_data["model_filename"])
state_dict = torch.load(save_path, map_location=device)
model.load_state_dict(state_dict)

print("model parameters:", count_params(model))

folder = _PICTURES_DIR

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
)
pred = []
index = 0
# x_test2 stores *input f* (H, W) for later plotting
x_test2 = []
# y_test2 stores the *physical* true solution u_true (H, W)
y_test2 = []
# phi_test2 has the same resolution as u_true/u_pred (train ::2 -> 51×51, test ::2 -> 201×201)
phi_test2 = []
rel_l2_set = []
h1_semi_set = []
energy_norm_set = []
w_pred_l = []
w_true_l = []
model_forward_times = []

with torch.no_grad():
    for x, u_true in test_loader:
        x, u_true = x.to(device), u_true.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_model0 = time.perf_counter()
        out = model(x)  # (1, H, W, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_forward_times.append(time.perf_counter() - t_model0)

        if model_data["normalized"] is True:
            u_true = normalizers[1].decode(u_true)
            x = normalizers[0].decode(x)
        phi12 = x[:, ::2, ::2, 0:2]
        phi_h = phi12[..., 0:1] * phi12[..., 1:2]
        u_h = out[:, ::2, ::2, 0:2]
        u_true1 = u_true[:, ::2, ::2, 0:2]
        w_pred_l.append(out[:, ::2, ::2, 0:2])
        w_true_l.append(u_true[:, ::2, ::2])



        # 2. Store f (input) for later plotting
        x_test2.append(x.squeeze())  # store (H, W)

        # 5. Create a physical mask (compute only where phi <= 0)
        # 1.0 = inside domain, 0.0 = outside domain
        # physical_mask = torch.where(phi_h <= 0, 1.0, 0.0)

        # 6. Apply the mask for a fair loss evaluation and plotting
        mask_node = get_node_masks(phi_h).squeeze(-1)
        u_true_masked = u_true1[mask_node]
        u_pred_masked = u_h[mask_node]

        # 7. Store the masked U
        y_test2.append(u_true1)  # (H, W)
        pred.append(u_h)  # (H, W)
        phi_test2.append(phi_h.squeeze())  # (H, W), same as u_h / u_true1
        # 8. Relative L2: LpLoss.__call__ -> rel(pred, true) = ||pred-true||_2 / ||true||_2 (masked nodes flattened)
        rel_l2 = myLoss(u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)).item()
        rel_l2_set.append(rel_l2)

        phi_np = np.squeeze(phi_h.detach().cpu().numpy())
        u_true_np = np.squeeze(u_true1.detach().cpu().numpy())
        u_pred_np = np.squeeze(u_h.detach().cpu().numpy())
        # compute_errors nb_cell must match the ::2 vertex grid (Ny×Nx vertices → Ny-1 cells)
        ny_eval, nx_eval = u_true_np.shape[0], u_true_np.shape[1]
        if ny_eval != nx_eval:
            raise ValueError(f"eval grid must be square, got {u_true_np.shape}")
        nb_cell_eval = ny_eval - 1
        try:
            h1_semi, energy_norm = compute_errors(
                phi_np,
                u_true_np,
                u_pred_np,
                nb_cell=nb_cell_eval,
                deg_v=1,
            )
        except Exception as e:
            print(f"Error computing errors: {e}")
            h1_semi = float("nan")
            energy_norm = float("nan")
        h1_semi_set.append(h1_semi)
        energy_norm_set.append(energy_norm)
        if index == 0:
            print(
                f"  (eval grid {ny_eval}×{nx_eval}, compute_errors nb_cell={nb_cell_eval})"
            )
        print(
            f"[test {index}] rel_L2(myLoss)={rel_l2:.6e}  "
            f"rel_H1_semi={h1_semi:.6e}  rel_energy={energy_norm:.6e}  "
            f"inference={model_forward_times[-1]*1e3:.3f}ms"
        )
        index = index + 1

train_eval_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
)
train_rel_l2_set = []
with torch.no_grad():
    for x, u_true in train_eval_loader:
        x, u_true = x.to(device), u_true.to(device)
        out = model(x)
        if model_data["normalized"] is True:
            u_true = normalizers[1].decode(u_true)
            x = normalizers[0].decode(x)
        phi12 = x[:, ::2, ::2, 0:2]
        phi_h = phi12[..., 0:1] * phi12[..., 1:2]
        u_h = out[:, ::2, ::2, 0:2]
        u_true1 = u_true[:, ::2, ::2, 0:2]
        mask_node = get_node_masks(phi_h).squeeze(-1)
        u_true_masked = u_true1[mask_node]
        u_pred_masked = u_h[mask_node]
        rel_l2_tr = myLoss(
            u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)
        ).item()
        train_rel_l2_set.append(rel_l2_tr)

h1_arr = np.asarray(h1_semi_set, dtype=np.float64)
en_arr = np.asarray(energy_norm_set, dtype=np.float64)
n_ok_h1 = np.sum(np.isfinite(h1_arr))
n_ok_en = np.sum(np.isfinite(en_arr))
print("-" * 60)
rel_l2_arr = np.asarray(rel_l2_set, dtype=np.float64)


def _nanstd_ddof1(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(np.std(a, ddof=1))


train_rel_l2_arr = np.asarray(train_rel_l2_set, dtype=np.float64)
print(
    f"train set relative L2 (myLoss): mean={np.mean(train_rel_l2_arr):.6e}, "
    f"std={_nanstd_ddof1(train_rel_l2_arr):.6e} (N={len(train_rel_l2_arr)})"
)
print(
    f"test set relative L2 (myLoss): mean={np.mean(rel_l2_arr):.6e}, std={_nanstd_ddof1(rel_l2_arr):.6e} "
    f"(N={len(rel_l2_arr)})"
)
print(
    f"relative H1 seminorm (compute_errors): mean={np.nanmean(h1_arr):.6e}, std={np.nanstd(h1_arr, ddof=1):.6e} "
    f"(valid samples {n_ok_h1}/{len(h1_arr)})"
)
print(
    f"relative energy norm (compute_errors): mean={np.nanmean(en_arr):.6e}, std={np.nanstd(en_arr, ddof=1):.6e} "
    f"(valid samples {n_ok_en}/{len(en_arr)})"
)
model_forward_times = np.asarray(model_forward_times, dtype=np.float64)
print(
    f"inference time (model forward): N={len(model_forward_times)}, "
    f"mean={float(model_forward_times.mean()):.6e}s ({float(model_forward_times.mean()*1e3):.3f}ms), "
    f"std={float(model_forward_times.std(ddof=0)):.6e}s, "
    f"total={float(model_forward_times.sum()):.6e}s"
)
print("-" * 60)

if plot_fig == True:
    err_labels = [
        r"Train $L^2$",
        r"Test $L^2$",
        r"Test $H^1$",
        r"Test energy",
    ]
    err_data = [
        train_rel_l2_arr[np.isfinite(train_rel_l2_arr)],
        rel_l2_arr[np.isfinite(rel_l2_arr)],
        h1_arr[np.isfinite(h1_arr)],
        en_arr[np.isfinite(en_arr)],
    ]
    _err_floor = 1e-20
    err_data_plot = [
        np.maximum(np.asarray(d, dtype=np.float64), _err_floor) for d in err_data
    ]

    fig_font = "DejaVu Serif"
    _plot_fs = 10
    plt.rcParams.update(
        {
            "font.family": fig_font,
            "font.size": _plot_fs,
            "axes.titlesize": _plot_fs,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": _plot_fs,
            "legend.fontsize": _plot_fs,
        }
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_err, ax_err = plt.subplots(figsize=(6.8, 5))
    positions = np.arange(1, len(err_labels) + 1)
    bp = ax_err.boxplot(
        err_data_plot,
        positions=positions,
        widths=0.55,
        whis=1.5,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="crimson",
            markersize=5,
            markeredgewidth=1.0,
            linestyle="none",
            alpha=0.85,
        ),
    )
    box_colors = ["#D4C8F0", "#A8C8EC", "#F0B0B8", "#B8E0B8"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
    ax_err.set_xticks(positions)
    ax_err.set_xticklabels(err_labels)
    ax_err.set_ylabel("Relative error")
    ax_err.set_yscale("log")
    # ax_err.set_title(
    #     r"Relative errors"
    # )
    ax_err.grid(True, axis="y", linestyle="--", alpha=0.35, which="both")
    fig_err.tight_layout()
    err_bar_path = os.path.join(folder, "Vessel_error_bars.png")
    fig_err.savefig(err_bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig_err)
    print("saved error bar figure:", err_bar_path)

    worst_idx = int(np.argmax(rel_l2_set))
    print("worst test L2 index:", worst_idx)

    # Plot 9 figures for 1 test sample only (change plot_idx, e.g. worst_idx)
    plot_idx = 98
    if plot_idx >= len(x_test2):
        raise IndexError("plot_idx out of range for x_test2")


    _plot_fs = 15
    plt.rcParams.update(
        {
            "font.family": fig_font,
            "font.size": _plot_fs,
            "axes.titlesize": 18,
            "axes.labelsize": _plot_fs,
            "xtick.labelsize": _plot_fs,
            "ytick.labelsize": _plot_fs,
            "legend.fontsize": _plot_fs,
        }
    )
    if not os.path.exists(folder):
        os.makedirs(folder)

    k = plot_idx
    x_input = x_test2[k].cpu()
    u_exact = y_test2[k].cpu()
    u_pred = pred[k].cpu()
    phi_grid = x_input.unsqueeze(0)[:, :, :, 0:1]
    mask1 = ~(get_node_masks(phi_grid))
    phi_plot_hw = phi_test2[k].detach().cpu().numpy()

    g_hw = x_input[:, :, 6:]
    g_np = g_hw.numpy()
    H, _, _ = g_np.shape
    theta_rad = np.linspace(0.0, np.pi / 2.0, H)
    theta_deg = np.rad2deg(theta_rad)
    pressure_theta = g_np[:, 0, 0]

    plt.figure()
    plt.plot(theta_deg, pressure_theta)
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel("Pressure")
    plt.title("Input", fontsize=20)
    plt.savefig(
        os.path.join(folder, "Vessel_Input.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    norm_exact = torch.sqrt(
        u_exact[0, :, :, 0] ** 2 + u_exact[0, :, :, 1] ** 2
    )
    norm_pred = torch.sqrt(
        u_pred[0, :, :, 0] ** 2 + u_pred[0, :, :, 1] ** 2
    )

    ux_e = u_exact[0, :, :, 0]
    ux_p = u_pred[0, :, :, 0]
    uy_e = u_exact[0, :, :, 1]
    uy_p = u_pred[0, :, :, 1]

    plot_field_2d_exact(
        ux_e,
        1,
        1,
        "Exact X displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_X_exact",
    )
    plot_field_2d_exact(
        ux_p,
        1,
        1,
        "Predict X displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_X_predict",
    )
    plot_field_2d_exact(
        torch.abs(ux_e - ux_p),
        1,
        1,
        "X displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_X_error",
        isError=True,
        error_nonnegative_clip=True,
    )

    plot_field_2d_exact(
        uy_e,
        1,
        1,
        "Exact Y displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_Y_exact",
    )
    plot_field_2d_exact(
        uy_p,
        1,
        1,
        "Predict Y displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_Y_predict",
    )
    plot_field_2d_exact(
        torch.abs(uy_e - uy_p),
        1,
        1,
        "Y displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Vessel_Y_error",
        isError=True,
        error_nonnegative_clip=True,
    )

    mask_np = np.asarray(mask1.squeeze().cpu().numpy(), dtype=bool)
    phi_np = phi_plot_hw
    u_exact_np = u_exact[0].detach().cpu().numpy()
    u_pred_np = u_pred[0].detach().cpu().numpy()

    def build_full_vessel_from_quarter(u_quarter, mask_quarter, phi_quarter=None):
        """
        Mirror-stitch a quarter vessel (the quadrant solution of this script) into a full vessel.
        Displacement components follow geometric symmetry:
          - Mirror about the vertical axis: ux changes sign, uy unchanged
          - Mirror about the horizontal axis: uy changes sign, ux unchanged
        """
        u_q = np.asarray(u_quarter, dtype=np.float64)
        m_q = np.asarray(mask_quarter, dtype=bool)
        p_q = None if phi_quarter is None else np.asarray(phi_quarter, dtype=np.float64)

        # Upper-right (original quarter, physical domain [0,1]x[0,1])
        u_ru = u_q
        m_ru = m_q
        p_ru = p_q

        # Upper-left (mirror of upper-right about the y-axis)
        u_lu = u_q[:, ::-1, :].copy()
        u_lu[..., 0] *= -1.0
        m_lu = m_q[:, ::-1]
        p_lu = None if p_q is None else p_q[:, ::-1]

        # Stitch the upper half (keep shared edge, do not duplicate a column)
        u_up = np.concatenate([u_lu, u_ru[:, 1:, :]], axis=1)
        m_up = np.concatenate([m_lu, m_ru[:, 1:]], axis=1)
        p_up = None if p_q is None else np.concatenate([p_lu, p_ru[:, 1:]], axis=1)

        # Lower half (mirror about the horizontal axis)
        u_dn = u_up[::-1, :, :].copy()
        u_dn[..., 1] *= -1.0
        m_dn = m_up[::-1, :]
        p_dn = None if p_up is None else p_up[::-1, :]

        # Stitch the full domain (keep shared edge, do not duplicate a row)
        # Note: array row 0 corresponds to the lower boundary of y_domain, so place the lower half first, then the upper half
        u_full = np.concatenate([u_dn, u_up[1:, :, :]], axis=0)
        m_full = np.concatenate([m_dn, m_up[1:, :]], axis=0)
        p_full = None if p_up is None else np.concatenate([p_dn, p_up[1:, :]], axis=0)
        return u_full, m_full, p_full

    # plot_u_matrix_displaced(
    #     u_exact_np,
    #     mask_np,
    #     "Exact displaced mesh",
    #     scale=1.0,
    #     folder=folder,
    #     file="Vessel_Displaced_exact",
    #     upsample=10,
    #     phi=phi_np,
    # )
    # plot_u_matrix_displaced(
    #     u_pred_np,
    #     mask_np,
    #     "Predict displaced mesh",
    #     scale=1.0,
    #     folder=folder,
    #     file="Vessel_Displaced_predict",
    #     upsample=10,
    #     phi=phi_np,
    # )

    u_exact_full, mask_full, phi_full = build_full_vessel_from_quarter(
        u_exact_np, mask_np, phi_np
    )
    u_pred_full, _, _ = build_full_vessel_from_quarter(
        u_pred_np, mask_np, phi_np
    )
    plot_u_matrix_displaced_custom_domain(
        u_pred_full,
        mask_full,
        title="Predict displaced field (full vessel)",
        scale=1.0,
        folder=folder,
        file="Vessel_Displaced_full_predict",
        upsample=10,
        phi=phi_full,
        x_domain=[-1, 1],
        y_domain=[-1, 1],
        xlim=[-1.3, 1.3],
        ylim=[-1.3, 1.3],
    )
