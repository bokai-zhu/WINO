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

from Phi_FEM.generate_data_ellip import compute_errors, solve_only, call_G
from torch.nn.modules import fold
from torch.utils.data import DataLoader, random_split, Subset
from utils.postprocessing import plot_field_2d, plot_field_2d_exact, plot_u_matrix_displaced
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
# Settings dictionary
model_data = dict()
model_data["device"] = device
model_data["n_train"] = 500
model_data["n_test"] = 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["batch_size"] = 50
model_data["learning_rate_adam"] = 5e-3
model_data["num_epoch_adam"] = 500
model_data["gamma"] = 0.5
model_data["patience"] = 500
model_data["min_lr"] = 2e-4
model_data["num_epoch"] = model_data["num_epoch_adam"]
model_data["step_size"] = 25
model_data["E"] = 10
model_data["nu"] = 0.3
model_data["mu"] = model_data["E"] / (2 * (1 + model_data["nu"]))
model_data["lambda"] = model_data["E"] * model_data["nu"] / ((1 + model_data["nu"]) * (1 - 2 * model_data["nu"]))
model_data["W"] = 1     # width
model_data["H"] = 1     # height
model_data["grid_point_num"] = 64
# Reference solution and three errors are computed on the fine grid: bilinear upsample input from 64 to 128 then solve_only; upsample predicted w to 128 then synthesize u
model_data["grid_fine"] = 64
model_data["ref_n_load_steps_f"] = 4
model_data["lambda_loss"] = 1e4
# model_data["lambda_vin"] = 0.5
model_data["fno"] = dict()
model_data["fno"]["modes"] = 16
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["input"] = 7
model_data["fno"]["output"] = 2
model_data["fno"]["use_data"] = False
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["normalized"] = True
model_data["dir"] = os.path.join(_WINO_ROOT, 'data')
model_data["path_train"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_ellip_GF_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_train"]) + '_train.npz')
model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_ellip_GF_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
# Build a distinctive filename from the config
model_data["model_filename"] = "WINO_hyperelasticity_ellip_s{}_n{}_ep{}.pth".format(
    model_data["grid_point_num"],
    model_data["n_data"],
    model_data["num_epoch"]
)
plot_fig = True
plot_idx = 2
model_data["plot_phi_first_n_test"] = 5



os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_PICTURES_DIR, exist_ok=True)
os.makedirs(_RESULT_DIR, exist_ok=True)

if model_data["grid_point_num"] // 2 + 1 < model_data["fno"]["modes"]:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

N_GRID_COARSE = int(model_data["grid_point_num"])
N_GRID_FINE = int(model_data["grid_fine"])
NB_CELL_FINE = N_GRID_FINE - 1


def _upsample_bilinear_1hwc(x_1hwc: torch.Tensor, n_out: int) -> torch.Tensor:
    """Bilinear upsample on a regular grid, (1, H, W, C) -> (1, n_out, n_out, C)."""
    if x_1hwc.shape[1] == n_out and x_1hwc.shape[2] == n_out:
        return x_1hwc
    x = x_1hwc.permute(0, 3, 1, 2).contiguous()
    y = torch.nn.functional.interpolate(
        x,
        size=(n_out, n_out),
        mode="bilinear",
        align_corners=True,
    )
    return y.permute(0, 2, 3, 1).contiguous()


def _sample_grf_field(n: int, mean: float = 4.0, std: float = 1.0, corr_len: float = 0.12, seed: int = 0):
    """Construct a 2D GRF via frequency-domain low-pass filtering."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    filt = np.exp(-0.5 * (corr_len**2) * (KX**2 + KY**2))
    field = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    field = (field - np.mean(field)) / (np.std(field) + 1e-12)
    return mean + std * field



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

# Making dataloaders
t_start = time.time()
train_loader = DataLoader(
    train_dataset,
    batch_size=model_data["batch_size"],
    shuffle=True,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=model_data["batch_size"],
    shuffle=True,
)

model = FNO2d(model_data).to(device)
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
# Original-resolution input (not projected to the fine grid), used for g/f visualization
x_test2_raw = []
# y_test2: Phi-FEM reference displacement u on the fine grid (same resolution as pred)
y_test2 = []
rel_l2_set = []
h1_semi_set = []
energy_norm_set = []
w_pred_l = []
w_true_l = []

with torch.no_grad():
    for x, u_true in test_loader:
        x, u_true = x.to(device), u_true.to(device)
        out = model(x)  # (1, H, W, 1)

        if model_data["normalized"] is True:
            u_true = normalizers[1].decode(u_true)
            out = normalizers[1].decode(out)
            x = normalizers[0].decode(x)
        x_test2_raw.append(x.squeeze(0).detach().cpu())
        # Fine grid: bilinear-interpolate inputs then Phi-FEM reference; upsample predicted w to the fine grid then synthesize u with fine-grid φ,g
        x_f = _upsample_bilinear_1hwc(x, N_GRID_FINE)
        phi_f = x_f[:, :, :, 0:1]
        g_f = x_f[:, :, :, 1:3]
        f_f = x_f[:, :, :, 3:5]
        w_pred_f = _upsample_bilinear_1hwc(out[:, :, :, 0:2], N_GRID_FINE)
        u_pred_f = w_pred_f * phi_f + g_f

        if N_GRID_FINE == N_GRID_COARSE:
            # Same resolution as training data: synthesize u from reference w in the npz, do not call solve_only
            w_ref_batch = u_true[:, :, :, 0:2]
            u_ref_f = w_ref_batch * phi_f + g_f
            w_pred_l.append(w_pred_f)
            w_true_l.append(w_ref_batch)
            ref_desc = "ref=npz w (grid_fine==grid_point_num, skip solve_only)"
        else:
            phi_np = phi_f.detach().cpu().numpy()
            g_np = g_f.detach().cpu().numpy()
            f_np = f_f.detach().cpu().numpy()
            _, sol_ref = solve_only(
                phi_np,
                g_np,
                f_np,
                init_guess=None,
                nb_vert=N_GRID_FINE,
                linear_solver="lu",
                n_load_steps_f=int(model_data["ref_n_load_steps_f"]),
            )
            if sol_ref is None or sol_ref[0] is None:
                print(
                    f"[test {index}] solve_only (ref {N_GRID_FINE}x{N_GRID_FINE}) failed, "
                    f"errors set to nan"
                )
                rel_l2_set.append(float("nan"))
                h1_semi_set.append(float("nan"))
                energy_norm_set.append(float("nan"))
                w_pred_l.append(w_pred_f)
                w_true_l.append(torch.full_like(w_pred_f, float("nan")))
                x_test2.append(x_f.squeeze(0))
                y_test2.append(torch.full_like(u_pred_f, float("nan")))
                pred.append(u_pred_f)
                index = index + 1
                continue

            w_ref_np = sol_ref[0]
            w_ref_t = torch.from_numpy(np.asarray(w_ref_np, dtype=np.float64)).to(
                device=device, dtype=out.dtype
            )
            if w_ref_t.dim() == 4:
                w_ref_t = w_ref_t[0]
            u_ref_f = w_ref_t.unsqueeze(0) * phi_f + g_f
            w_pred_l.append(w_pred_f)
            w_true_l.append(w_ref_t.unsqueeze(0))
            ref_desc = f"ref=solve_only {N_GRID_FINE}x{N_GRID_FINE}"

        x_test2.append(x_f.squeeze(0))

        # 6. Mask at evaluation resolution
        mask_node = get_node_masks(phi_f).squeeze(-1)
        u_true_masked = u_ref_f[mask_node]
        u_pred_masked = u_pred_f[mask_node]

        y_test2.append(u_ref_f)
        pred.append(u_pred_f)
        rel_l2 = myLoss(u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)).item()
        rel_l2_set.append(rel_l2)

        phi_s = np.squeeze(phi_f.detach().cpu().numpy())
        u_true_np = np.squeeze(u_ref_f.detach().cpu().numpy())
        u_pred_np = np.squeeze(u_pred_f.detach().cpu().numpy())
        try:
            h1_semi, energy_norm = compute_errors(
                phi_s,
                u_true_np,
                u_pred_np,
                nb_cell=NB_CELL_FINE,
                deg_v=1,
            )
        except Exception as e:
            print(f"Error computing errors: {e}")
            h1_semi = float("nan")
            energy_norm = float("nan")
        h1_semi_set.append(h1_semi)
        energy_norm_set.append(energy_norm)
        print(
            f"[test {index}] rel_L2(myLoss)={rel_l2:.6e}  "
            f"rel_H1_semi={h1_semi:.6e}  rel_energy={energy_norm:.6e}  "
            f"({ref_desc})"
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
            out = normalizers[1].decode(out)
            x = normalizers[0].decode(x)
        phi_h = x[:, :, :, 0:1]
        g_h = x[:, :, :, 1:3]
        f_h = x[:, :, :, 3:]
        u_h = out[:, :, :, 0:2] * phi_h + g_h
        u_true1 = u_true[:, :, :, 0:2] * phi_h + g_h
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

result_dir = _RESULT_DIR
os.makedirs(result_dir, exist_ok=True)
_rel_l2_npz = os.path.join(
    result_dir,
    model_data["model_filename"].replace(".pth", "_rel_L2.npz"),
)
if N_GRID_FINE != N_GRID_COARSE or N_GRID_FINE == 128:
    np.savez_compressed(
        _rel_l2_npz,
        test_rel_l2=rel_l2_arr,
        test_mean=np.float64(np.nanmean(rel_l2_arr)),
        test_std=np.float64(_nanstd_ddof1(rel_l2_arr)),
        train_mean=np.float64(np.mean(train_rel_l2_arr)),
        train_std=np.float64(_nanstd_ddof1(train_rel_l2_arr)),
    )
    print(f"relative L2 data saved: {_rel_l2_npz}")

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
print("-" * 60)

if plot_fig == True:
    err_labels = [
        r"Train $L^2$ norm",
        r"Test $L^2$ norm",
        r"Test $H^1$ seminorm",
        r"Test energy norm",
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
            "axes.labelsize": _plot_fs,
            "xtick.labelsize": _plot_fs,
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
    ax_err.grid(True, axis="y", linestyle="--", alpha=0.35, which="both")
    fig_err.tight_layout()
    err_bar_path = os.path.join(folder, "ellip_error_bars.png")
    fig_err.savefig(err_bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig_err)
    print("saved error bar figure:", err_bar_path)

    worst_idx = int(np.argmax(rel_l2_set))
    print("worst test L2 index:", worst_idx)

    # Plot 9 figures for 1 test sample only (change plot_idx, e.g. worst_idx)
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
    x_input_raw = x_test2_raw[k].cpu()
    u_exact = y_test2[k].cpu()
    u_pred = pred[k].cpu()
    phi_grid = x_input.unsqueeze(0)[:, :, :, 0:1]
    mask1 = ~(get_node_masks(phi_grid))
    phi_plot_hw = x_input[:, :, 0].detach().cpu().numpy()

    g_hw = x_input_raw[:, :, 1:3]
    f_hw = x_input_raw[:, :, 3:5]
    mask_phi = np.asarray(x_input_raw[:, :, 0].detach().cpu().numpy(), dtype=np.float64) > 0.0

    plot_field_2d(
        g_hw[:, :, 0].numpy(),
        1,
        1,
        r"$g_x$",
        folder=folder,
        file="ellip_g_x",
        mask=mask_phi,
    )
    plot_field_2d(
        g_hw[:, :, 1].numpy(),
        1,
        1,
        r"$g_y$",
        folder=folder,
        file="ellip_g_y",
        mask=mask_phi,
    )
    plot_field_2d(
        f_hw[:, :, 0].numpy(),
        1,
        1,
        r"$f_x$",
        folder=folder,
        file="ellip_f_x",
        mask=mask_phi,
    )
    plot_field_2d(
        f_hw[:, :, 1].numpy(),
        1,
        1,
        r"$f_y$",
        folder=folder,
        file="ellip_f_y",
        mask=mask_phi,
    )

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
        file="ellip_X_exact",
    )
    plot_field_2d_exact(
        ux_p,
        1,
        1,
        "Predict X displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="ellip_X_predict",
    )
    plot_field_2d_exact(
        torch.abs(ux_e - ux_p),
        1,
        1,
        "X displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="ellip_X_error",
        isError=True,
        error_nonnegative_clip=True,
        colorbar_decimals=6,
    )

    plot_field_2d_exact(
        uy_e,
        1,
        1,
        "Exact Y displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="ellip_Y_exact",
    )
    plot_field_2d_exact(
        uy_p,
        1,
        1,
        "Predict Y displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="ellip_Y_predict",
    )
    plot_field_2d_exact(
        torch.abs(uy_e - uy_p),
        1,
        1,
        "Y displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="ellip_Y_error",
        isError=True,
        error_nonnegative_clip=True,
        colorbar_decimals=6,
    )

