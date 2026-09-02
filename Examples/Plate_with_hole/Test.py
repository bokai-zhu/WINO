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

from Phi_FEM.generate_data_Hole import compute_errors
from torch.nn.modules import fold
from torch.utils.data import DataLoader, random_split, Subset
from utils.postprocessing import plot_field_2d_exact, plot_u_matrix_displaced
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
class FNO2d_HyperElasticity(FNO2d):
    def __init__(self, model_data, normalizers=None):
        super(FNO2d_HyperElasticity, self).__init__(model_data)
        if normalizers is not None:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def forward(self, x):
        out = super().forward(x)

        if model_data["normalized"] is True:
            x = self.normalizer_x.decode(x)
            out[..., 0:2] = self.normalizer_y.decode(out[..., 0:2])
        g_h = x[:, :, :, 1:3]
        # multiply by y
        y_coords = torch.linspace(0, 1, out.shape[1]).to(model_data["device"])
        y_grid = y_coords.reshape(1, out.shape[1], 1, 1)
        out[..., 0:2] = out[..., 0:2] * y_grid

        # out[:, -1, :, :2] = u_D
        # out[:, 0, :, :2] = 0

        return out

# Settings dictionary
model_data = dict()
model_data["device"] = device
model_data["n_train"] = 300
model_data["n_test"] = 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["batch_size"] = 50
model_data["learning_rate_adam"] = 5e-3
model_data["num_epoch_adam"] = 2000
model_data["gamma"] = 0.5
model_data["patience"] = 500
# model_data["learning_rate_lbfgs"] = 1.0
# model_data["max_iter_lbfgs"] = 10
# model_data["num_epoch_lbfgs"] = 0
# model_data["num_epoch"] = model_data["num_epoch_adam"] + model_data["num_epoch_lbfgs"]
model_data["num_epoch"] = model_data["num_epoch_adam"]
model_data["step_size"] = 25
# model_data["gamma"] = 0.5
# model_data["patience"] = 50
# model_data["cooldown"] = 5
model_data["min_lr"] = 1e-5
model_data["sigma_D"] = 3
model_data["W"] = 1
model_data["H"] = 1
model_data["E"] = 100
model_data["nu"] = 0.3
model_data["mu"] = model_data["E"] / (2 * (1 + model_data["nu"]))
model_data["lambda"] = model_data["E"] * model_data["nu"] / ((1 + model_data["nu"]) * (1 - 2 * model_data["nu"]))
model_data["grid_point_num"] = 64
# model_data["sensor_point_num"] = model_data["grid_point_num"]
model_data["lambda_loss"] = np.array([1e4, 0.01, 1e-3, 0.01, 1e-14]) * 1
model_data["lambda_vin"] = 0.5
model_data["norm_y"] = model_data["E"]
model_data["norm_p"] = model_data["E"]
model_data["fno"] = dict()
model_data["fno"]["modes"] = 16
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["input"] = 5
model_data["fno"]["output"] = 8
model_data["fno"]["use_data"] = False
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["fno"]["use_grad"] = True
model_data["normalized"] = True
model_data["dir"] = os.path.join(_WINO_ROOT, 'data')
model_data["path_train"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Hole_G_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_train"]) + '_train.npz')
model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Hole_G_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
# Build a distinctive filename from the config
model_data["model_filename"] = "WINO_hyperelasticity_hole_s{}_n{}_ep{}.pth".format(
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

model = FNO2d_HyperElasticity(model_data, normalizers).to(device)
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
            x = normalizers[0].decode(x)
        # y_h = out[:, :, :, 2:6]
        # p_h = out[:, :, :, 6:]
        phi_h = x[:, :, :, 0:1]
        g_h = x[:, :, :, 1:]
        u_h = out[:, :, :, 0:2]
        u_true1 = u_true[:, :, :, 0:2]

        w_pred_l.append(out[..., 0:2])
        w_true_l.append(u_true)


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
        # 8. Relative L2: LpLoss.__call__ -> rel(pred, true) = ||pred-true||_2 / ||true||_2 (masked nodes flattened)
        rel_l2 = myLoss(u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)).item()
        rel_l2_set.append(rel_l2)

        phi_np = np.squeeze(phi_h.detach().cpu().numpy())
        u_true_np = np.squeeze(u_true1.detach().cpu().numpy())
        u_pred_np = np.squeeze(u_h.detach().cpu().numpy())
        try:
            h1_semi, energy_norm = compute_errors(
                phi_np,
                u_true_np,
                u_pred_np,
                nb_cell=model_data["grid_point_num"] - 1,
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
            f"rel_H1_semi={h1_semi:.6e}  rel_energy={energy_norm:.6e}"
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
        phi_h = x[:, :, :, 0:1]
        u_h = out[:, :, :, 0:2]
        u_true1 = u_true[:, :, :, 0:2]
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
    # ax_err.set_title(
    #     r"Relative errors"
    # )
    ax_err.grid(True, axis="y", linestyle="--", alpha=0.35, which="both")
    fig_err.tight_layout()
    err_bar_path = os.path.join(folder, "Hole_error_bars.png")
    fig_err.savefig(err_bar_path, dpi=600, bbox_inches="tight")
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
    phi_plot_hw = x_input[:, :, 0].detach().cpu().numpy()

    g_hw = x_input[:, :, 1:3]
    g_np = g_hw.numpy()
    _, W, _ = g_np.shape
    x_coord = np.linspace(0.0, 1.0, W)

    plt.figure()
    plt.plot(x_coord, g_np[-1, :, 1], label=r"$g_y$")
    plt.xlabel("x")
    plt.ylabel("Traction")
    plt.title("Input", fontsize=20)
    plt.savefig(
        os.path.join(folder, "Hole_Input.png"),
        dpi=600,
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
        file="Hole_X_exact",
    )
    plot_field_2d_exact(
        ux_p,
        1,
        1,
        "Predict X displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Hole_X_predict",
    )
    plot_field_2d_exact(
        torch.abs(ux_e - ux_p),
        1,
        1,
        "X displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Hole_X_error",
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
        file="Hole_Y_exact",
    )
    plot_field_2d_exact(
        uy_p,
        1,
        1,
        "Predict Y displacement",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Hole_Y_predict",
    )
    plot_field_2d_exact(
        torch.abs(uy_e - uy_p),
        1,
        1,
        "Y displacement error",
        phi=phi_plot_hw,
        phi_hole_positive=True,
        folder=folder,
        file="Hole_Y_error",
        isError=True,
        error_nonnegative_clip=True,
    )

    mask_np = np.asarray(mask1.squeeze().cpu().numpy(), dtype=bool)
    phi_np = phi_plot_hw
    u_exact_np = u_exact[0].detach().cpu().numpy()
    u_pred_np = u_pred[0].detach().cpu().numpy()

    plot_u_matrix_displaced(
        u_exact_np,
        mask_np,
        "Exact displaced mesh",
        scale=1.0,
        folder=folder,
        file="Hole_Displaced_exact",
        upsample=10,
        phi=phi_np,
    )
    plot_u_matrix_displaced(
        u_pred_np,
        mask_np,
        "Predict displaced field",
        scale=1.0,
        folder=folder,
        file="Hole_Displaced_predict",
        upsample=10,
        phi=phi_np,
    )

    # Follow phi_fem_VINO_hyper/test.py: fix current-sample phi, uniform top-boundary g_y = constant, sweep loads 15/25/35/45
    g_sweep_values = [15, 25, 35, 45]
    nb_g = len(g_sweep_values)
    x_sweep = (
        x_input.unsqueeze(0)
        .expand(nb_g, -1, -1, -1)
        .contiguous()
        .clone()
        .to(device)
    )
    for ig, gv in enumerate(g_sweep_values):
        x_sweep[ig, :, :, 1] = 0.0
        x_sweep[ig, :, :, 2] = float(gv)
    if model_data["normalized"]:
        x_sweep_in = normalizers[0].encode(x_sweep)
    else:
        x_sweep_in = x_sweep
    with torch.no_grad():
        u_sweep = model(x_sweep_in)[:, :, :, 0:2]
    for ig, gv in enumerate(g_sweep_values):
        plot_u_matrix_displaced(
            u_sweep[ig].detach().cpu().numpy(),
            mask_np,
            scale=1.0,
            folder=folder,
            file=f"Hole_Displaced_predict_g{gv}",
            upsample=10,
            phi=phi_np,
        )
