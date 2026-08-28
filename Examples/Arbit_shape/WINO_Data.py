#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
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

from utils.postprocessing import plot_pred2, plot_field_2d
from utils.fno_2d import FNO2d
from utils.database_makers import HyperelasticityDataset
from utils.fno_utils import *
from utils.soap import SOAP
from transformers import get_cosine_schedule_with_warmup
# from utils.scheduler import LR_Scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
model_data["patience"] = 250
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
model_data["lambda_loss"] = 1e5
# model_data["lambda_vin"] = 0.5
model_data["fno"] = dict()
model_data["fno"]["modes"] = 16
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["input"] = 7
model_data["fno"]["output"] = 2
model_data["fno"]["use_data"] = True
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["normalized"] = True
model_data["dir"] = os.path.join(_WINO_ROOT, 'data')
model_data["path_train"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Arbit_GF_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_train"]) + '_train.npz')
model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Arbit_GF_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
# Build a distinctive filename from the config
model_data["model_filename"] = "RVINO_Data_hyperelasticity_arbit_s{}_n{}_ep{}.pth".format(
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
single_train = Subset(train_dataset, [0])


X_train = []
Y_train = []
X_test = []
Y_test = []
for x, y in train_dataset:
    X_train.append(x)
    Y_train.append(y)
for x, y in test_dataset:
    X_test.append(x)
    Y_test.append(y)

X_train = torch.stack(X_train)
Y_train = torch.stack(Y_train)
X_test  = torch.stack(X_test)
Y_test  = torch.stack(Y_test)

# model
model = FNO2d(model_data).to(device)
# model = FNO2d(model_data).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
print(f"\nmodel_data:{model_data}")

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)
################################################################
# training
################################################################
optimizer = SOAP(model.parameters(), lr=model_data["learning_rate_adam"],
                 betas=(.95, .99), weight_decay=0, precondition_frequency=5)
scheduler = None
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_l2_log, test_l2_log = train_rvino_hyperelasticity_DBC_Guass(
    model, model_data, train_loader, test_loader, myLoss, optimizer, scheduler, normalizers
)
print("Training time: ", default_timer() - t1)

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
        # y_h = out[:, :, :, 2:6]
        # p_h = out[:, :, :, 6:]
        phi_h = x[:, :, :, 0:1]
        g_h = x[:, :, :, 1:3]
        u_h = out[:, :, :, 0:2] * phi_h + g_h
        u_true1 = u_true[:, :, :, 0:2] * phi_h + g_h

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
        # 8. Compute L2 loss *on the masked physical field*
        # myLoss (LpLoss) compares u_pred_masked and u_true_masked
        test_l2 = myLoss(u_pred_masked.reshape(1, -1), u_true_masked.reshape(1, -1)).item()
        test_l2_set.append(test_l2)

        print(index, test_l2)
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())
print("Index of maximum error is", torch.argmax(test_l2_set).item())
################################################################
# evaluation
################################################################

# Plotting a random function from the test data generated by GRF
index = torch.argmax(test_l2_set).item()
print(index)
# index = 0
# x_test_plot = np.linspace(0., L, grid_point_num).astype('float32')
# y_test_plot = np.linspace(0., L, grid_point_num).astype('float32')
# x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)
#
x_input = x_test2[index].cpu()

# fig_font = "DejaVu Serif"
# plt.rcParams["font.family"] = fig_font
# plt.figure()
# plt.contourf(x_test_plot, y_test_plot, x_input, levels=500, cmap='hsv')
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('Input Function')
# plt.show()

u_exact = y_test2[index].cpu()
u_pred = pred[index].cpu()
w_pred = w_pred_l[index].cpu()
w_exact = w_true_l[index].cpu()
phi_grid = x_input.unsqueeze(0)[:, :, :, 0:1]
mask1 = ~(get_node_masks(phi_grid))
# mask2 = (abs(u_exact[0, :, :, 1]) <= 1e-4).unsqueeze(0).unsqueeze(3)
# mask1 = u_exact[..., 0] == 0
# mask1[:, 0] = False
# print(mask1.shape, u_exact.shape)
# plot_pred2(x_plot_grid, y_plot_grid, u_exact, u_pred, 'Test with GRF', 'Ex1')

################################################################
# save model
################################################################
save_path = os.path.join(_MODEL_DIR, model_data["model_filename"])

# Save model parameters (prefer saving only state_dict to save space and keep flexibility)
torch.save(model.state_dict(), save_path)
print(f"Model saved successfully to: {save_path}")
