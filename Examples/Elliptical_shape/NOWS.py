#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Feed predictions to FEMsolver
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter, MaxNLocator

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)
from timeit import default_timer
import time
import os
import sys
from torch.nn.modules import fold
from torch.utils.data import DataLoader, random_split, Subset
_WINO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _WINO_ROOT not in sys.path:
    sys.path.insert(0, _WINO_ROOT)
_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_CASE_DIR, "model")
_PICTURES_DIR = os.path.join(_CASE_DIR, "pictures")
_RESULT_DIR = os.path.join(_CASE_DIR, "result")

from utils.postprocessing import plot_field_2d, plot_field_2d_exact, plot_pred2
from utils.fno_2d import FNO2d
from utils.database_makers import HyperelasticityDataset
from utils.fno_utils import *
from utils.soap import SOAP
from transformers import get_cosine_schedule_with_warmup
from Phi_FEM.generate_data_ellip import solve_only as phi_fem_solve_only
# from utils.scheduler import LR_Scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
# model_data["sensor_point_num"] = model_data["grid_point_num"]
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
model_data["model_filename"] = "RVINO_hyperelasticity_ellip_s{}_n{}_ep{}.pth".format(
    model_data["grid_point_num"],
    model_data["n_data"],
    model_data["num_epoch"]
)

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


# model
model = FNO2d(model_data).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')

optimizer = SOAP(model.parameters(), lr=model_data["learning_rate_adam"],
                 betas=(.95, .99), weight_decay=0, precondition_frequency=5)
scheduler = None
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
save_path = os.path.join(_MODEL_DIR, model_data["model_filename"])
state_dict = torch.load(save_path, map_location=device)
model.load_state_dict(state_dict)

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
test_l2_set = []
time_no_init = []
time_with_init = []
gmres_iters_no_init = []
gmres_iters_with_init = []
newton_iters_no_init = []
newton_iters_with_init = []
newton_residual_hist_no_init = []
newton_residual_hist_with_init = []
phi_fem_with_init_l2_rel = []
phi_fem_with_init_u = []
model_forward_times = []

with torch.no_grad():
    for x, u_true in test_loader:
        x, u_true = x.to(device), u_true.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_model0 = time.perf_counter()
        out = model(x)  # (1, H, W, C)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_forward_times.append(time.perf_counter() - t_model0)

        if model_data["normalized"] is True:
            u_true = normalizers[1].decode(u_true)
            out[..., 0:2] = normalizers[1].decode(out[..., 0:2])
            x = normalizers[0].decode(x)

        phi_h = x[:, :, :, 0:1]
        g_h = x[:, :, :, 1:3]
        f_h = x[:, :, :, 3:5]
        u_h = out[:, :, :, 0:2] * phi_h + g_h
        u_true1 = u_true[:, :, :, 0:2] * phi_h + g_h

        # Phi-FEM initial guess u+y+p: u uses nodal mask; y/p use cut-cell-to-node mask
        mask_node_u = get_node_masks(phi_h).to(out.dtype)
        _, mask_cut_cell = get_cell_masks(phi_h)
        init_guess_masked = out.clone()
        init_guess_masked[:, :, :, 0:2] = init_guess_masked[:, :, :, 0:2] * mask_node_u

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
        # 8. Compute L2 loss *on the masked physical field*
        # myLoss (LpLoss) compares u_pred_masked and u_true_masked
        test_l2 = myLoss(u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)).item()
        test_l2_set.append(test_l2)

        # Run Phi-FEM (no-init / with-init) evaluation in the same loop to avoid a second pass
        phi_i = phi_h.detach().cpu().numpy()
        g_i = g_h.detach().cpu().numpy()
        f_i = f_h.detach().cpu().numpy()
        init_i = init_guess_masked.detach().cpu().numpy()
        t0, _ = phi_fem_solve_only(
            phi_i, g_i, f_i, init_guess=None, nb_vert=model_data["grid_point_num"]
        )
        gmres_it0 = int(
            getattr(
                phi_fem_solve_only,
                "last_total_gmres_iters",
                getattr(phi_fem_solve_only, "last_total_cg_iters", 0),
            )
        )
        nw_it0 = int(getattr(phi_fem_solve_only, "last_total_newton_iters", 0))
        hists_ni = getattr(phi_fem_solve_only, "last_newton_residual_histories", None) or []
        newton_residual_hist_no_init.append(
            list(hists_ni[0]) if len(hists_ni) > 0 else []
        )
        t1, sol1 = phi_fem_solve_only(
            phi_i, g_i, f_i, init_guess=init_i[...], nb_vert=model_data["grid_point_num"], n_load_steps_f=1
        )
        gmres_it1 = int(
            getattr(
                phi_fem_solve_only,
                "last_total_gmres_iters",
                getattr(phi_fem_solve_only, "last_total_cg_iters", 0),
            )
        )
        nw_it1 = int(getattr(phi_fem_solve_only, "last_total_newton_iters", 0))
        hists_wi = getattr(phi_fem_solve_only, "last_newton_residual_histories", None) or []
        newton_residual_hist_with_init.append(
            list(hists_wi[0]) if len(hists_wi) > 0 else []
        )

        U_phi_with_init = None
        if sol1 is not None:
            U_phi_with_init = sol1[0]
        if U_phi_with_init is not None:
            # Solver output is w (in the weak form u = w*φ + g); synthesize physical displacement u in the same convention as the dataset/network
            w_phi = torch.from_numpy(U_phi_with_init[0]).to(
                u_true1.device, dtype=u_true1.dtype
            )
            phi_hw = phi_h.squeeze(0)
            g_hw = g_h.squeeze(0)
            u_phi_i = w_phi * phi_hw + g_hw
            mask_phi = get_node_masks(phi_h).squeeze(-1).squeeze(0)
            u_phi_masked = u_phi_i[mask_phi]
            denom = torch.norm(u_true_masked.reshape(-1), p=2).item()
            numer = torch.norm((u_phi_masked - u_true_masked).reshape(-1), p=2).item()
            rel_l2 = numer / max(denom, 1e-12)
            phi_fem_with_init_l2_rel.append(rel_l2)
            phi_fem_with_init_u.append(u_phi_i.cpu())
        else:
            phi_fem_with_init_l2_rel.append(float("nan"))
            phi_fem_with_init_u.append(None)

        # On non-convergence, solve_only returns duration=nan; do not include in timing stats
        if np.isfinite(t0):
            time_no_init.append(t0)
        if np.isfinite(t1):
            time_with_init.append(t1)
        gmres_iters_no_init.append(gmres_it0)
        gmres_iters_with_init.append(gmres_it1)
        newton_iters_no_init.append(nw_it0)
        newton_iters_with_init.append(nw_it1)
        print(
            f"[{index}] test_l2: {test_l2:.6e}, "
            f"model_forward: {model_forward_times[-1]*1e3:.3f}ms, "
            f"phi-fem time no/with-init: {t0:.3f}s/{t1:.3f}s, "
            f"gmres no/with: {gmres_it0}/{gmres_it1}, "
            f"newton no/with: {nw_it0}/{nw_it1}, "
            f"phi-fem(with-init) relL2: {phi_fem_with_init_l2_rel[-1]:.6e}"
        )
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())
print("Index of maximum error is", torch.argmax(test_l2_set).item())
model_forward_times = np.asarray(model_forward_times, dtype=np.float64)
print(
    "--- model forward (out) timing ---\n"
    f"  n_samples: {len(model_forward_times)}, "
    f"mean: {float(model_forward_times.mean()):.6e}s ({float(model_forward_times.mean()*1e3):.3f}ms), "
    f"std: {float(model_forward_times.std(ddof=0)):.6e}s"
)

time_no_init = np.asarray(time_no_init, dtype=np.float64)
time_with_init = np.asarray(time_with_init, dtype=np.float64)
n_no = int(time_no_init.size)
n_wi = int(time_with_init.size)
mean_no = float(time_no_init.mean()) if n_no > 0 else float("nan")
std_no = float(time_no_init.std(ddof=0)) if n_no > 0 else float("nan")
mean_wi = float(time_with_init.mean()) if n_wi > 0 else float("nan")
std_wi = float(time_with_init.std(ddof=0)) if n_wi > 0 else float("nan")
print("--- Phi-FEM load-step loop timing stats (s) ---")
print(f"  finite timings only; no_init valid: {n_no}, with_init valid: {n_wi}")
print(f"  no_init    mean: {mean_no:.4f}  std: {std_no:.4f}")
print(f"  with_init  mean: {mean_wi:.4f}  std: {std_wi:.4f}")
if np.isfinite(mean_no) and np.isfinite(mean_wi) and mean_wi > 0:
    print(f"  mean ratio (no_init / with_init): {mean_no / mean_wi:.4f}")
gmres_iters_no_init = np.asarray(gmres_iters_no_init, dtype=np.float64)
gmres_iters_with_init = np.asarray(gmres_iters_with_init, dtype=np.float64)
mean_gmres_no = float(gmres_iters_no_init.mean())
mean_gmres_wi = float(gmres_iters_with_init.mean())
var_gmres_no = float(gmres_iters_no_init.var(ddof=0))
var_gmres_wi = float(gmres_iters_with_init.var(ddof=0))
print("--- Phi-FEM GMRES iteration stats ---")
print(f"  no_init    mean: {mean_gmres_no:.4f}  variance: {var_gmres_no:.4f}")
print(f"  with_init  mean: {mean_gmres_wi:.4f}  variance: {var_gmres_wi:.4f}")
newton_iters_no_init = np.asarray(newton_iters_no_init, dtype=np.float64)
newton_iters_with_init = np.asarray(newton_iters_with_init, dtype=np.float64)
mean_nw_no = float(newton_iters_no_init.mean())
mean_nw_wi = float(newton_iters_with_init.mean())
var_nw_no = float(newton_iters_no_init.var(ddof=0))
var_nw_wi = float(newton_iters_with_init.var(ddof=0))
print("--- Phi-FEM Newton iteration stats ---")
print(f"  no_init    mean: {mean_nw_no:.4f}  variance: {var_nw_no:.4f}")
print(f"  with_init  mean: {mean_nw_wi:.4f}  variance: {var_nw_wi:.4f}")
phi_fem_with_init_l2_rel = np.asarray(phi_fem_with_init_l2_rel, dtype=np.float64)
valid_phi = np.isfinite(phi_fem_with_init_l2_rel)
if np.any(valid_phi):
    mean_phi_l2 = float(np.mean(phi_fem_with_init_l2_rel[valid_phi]))
    var_phi_l2 = float(np.var(phi_fem_with_init_l2_rel[valid_phi], ddof=0))
    print("--- Phi-FEM(with-init) relative L2 error stats ---")
    print(f"  valid samples: {int(np.sum(valid_phi))}/{len(phi_fem_with_init_l2_rel)}")
    print(f"  mean: {mean_phi_l2:.6e}  variance: {var_phi_l2:.6e}")
else:
    print("--- Phi-FEM(with-init) relative L2 error stats ---")
    print("  no valid samples (Phi-FEM may have failed to converge on all)")

# Newton residual history: colored by no-init / with-network-init; x is cumulative Newton iters, y is SNES ||F||
plot_height = 5.0
fig_nw, ax_nw = plt.subplots(figsize=(9.0, plot_height))
color_no_init = "C0"
color_with_init = "C1"
alpha_lines = 0.24
lw = 0.85
n_plotted_no = 0
n_plotted_with = 0
n_empty_no = 0
n_empty_with = 0
for hist in newton_residual_hist_no_init:
    if not hist:
        n_empty_no += 1
        continue
    xs = np.arange(1, len(hist) + 1, dtype=np.float64)
    y = np.clip(np.asarray(hist, dtype=np.float64), 1e-30, None)
    ax_nw.semilogy(xs, y, color=color_no_init, alpha=alpha_lines, linewidth=lw)
    n_plotted_no += 1
for hist in newton_residual_hist_with_init:
    if not hist:
        n_empty_with += 1
        continue
    xs = np.arange(1, len(hist) + 1, dtype=np.float64)
    y = np.clip(np.asarray(hist, dtype=np.float64), 1e-30, None)
    ax_nw.semilogy(xs, y, color=color_with_init, alpha=alpha_lines, linewidth=lw)
    n_plotted_with += 1
ax_nw.set_xlabel("Newton iteration")
ax_nw.set_ylabel("Residual")
ax_nw.set_title("All test samples of Newton residual history")
ax_nw.grid(True, which="both", linestyle=":", alpha=0.5)
legend_elems = [
    Line2D(
        [0],
        [0],
        color=color_no_init,
        linewidth=2,
        alpha=0.85,
        label=r"$\varphi$-FEM",
    ),
    Line2D(
        [0],
        [0],
        color=color_with_init,
        linewidth=2,
        alpha=0.85,
        label=f"NOWS",
    ),
]
ax_nw.legend(handles=legend_elems, loc="upper right")
note_parts = []
if n_empty_no:
    note_parts.append(f"no-init no records {n_empty_no}")
if n_empty_with:
    note_parts.append(f"with-init no records {n_empty_with}")
if note_parts:
    ax_nw.text(
        0.02,
        0.02,
        "; ".join(note_parts) + " samples",
        transform=ax_nw.transAxes,
        verticalalignment="bottom",
    )
fig_nw.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.95)
os.makedirs(folder, exist_ok=True)
nw_png = "ellip_NewtonResidual.png"
fig_nw.savefig(os.path.join(folder, nw_png), dpi=200)
plt.close(fig_nw)
print(
    f"--- Newton residual curve saved: {folder}/{nw_png} "
    f"(no-init lines={n_plotted_no}, with-init lines={n_plotted_with}; "
    f"no records: no-init {n_empty_no}, with-init {n_empty_with})"
)

# GMRES total iteration boxplot (two columns: no init / with network init)
fig_box, ax_box = plt.subplots(figsize=(6, plot_height))
box_data = [
    np.asarray(gmres_iters_no_init, dtype=np.float64),
    np.asarray(gmres_iters_with_init, dtype=np.float64),
]
bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.65)
ax_box.set_xticklabels(["No init", "With init"])
for patch, c in zip(bp["boxes"], [color_no_init, color_with_init]):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
    patch.set_edgecolor("0.25")
for whisker in bp["whiskers"]:
    whisker.set(color="0.35", linewidth=1.0)
for cap in bp["caps"]:
    cap.set(color="0.35", linewidth=1.0)
for median in bp["medians"]:
    median.set(color="0.15", linewidth=1.4)
ax_box.set_ylabel("GMRES iterations")
ax_box.set_title("GMRES iteration count distribution")
ax_box.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax_box.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
ax_box.grid(True, axis="y", linestyle=":", alpha=0.55)
fig_box.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.95)
box_png = "ellip_GMRESIters_boxplot.png"
fig_box.savefig(os.path.join(folder, box_png), dpi=200)
plt.close(fig_box)
print(f"--- GMRES iteration boxplot saved: {folder}/{box_png}")




