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
################################################################
#  configurations
################################################################
class FNO2d_HyperElasticity_Trapezoid(FNO2d):
    def __init__(self, model_data, normalizers=None):
        super(FNO2d_HyperElasticity_Trapezoid, self).__init__(model_data)
        if normalizers is not None:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def forward(self, x):
        out = super().forward(x)

        if model_data["normalized"] is True:
            x = self.normalizer_x.decode(x)
            out[..., 0:2] = self.normalizer_y.decode(out[..., 0:2])
        g_h = x[:, :, :, 1:3]
        # multiply by x
        x_coords = torch.linspace(0, 4, out.shape[2]).to(model_data["device"])
        x_grid = x_coords.reshape(1, 1, out.shape[2], 1)
        out[..., 0:2] = out[..., 0:2] * x_grid
        return out

# Settings dictionary
model_data = dict()
model_data["device"] = device
model_data["n_train"] = 1000
model_data["n_test"] = 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["batch_size"] = 50
model_data["learning_rate_adam"] = 1e-2
model_data["num_epoch"] = 5000
model_data["gamma"] = 0.8
model_data["patience"] = 250
model_data["min_lr"] = 2e-4
model_data["step_size"] = 1
# model_data["gamma"] = 0.5
# model_data["patience"] = 50
# model_data["cooldown"] = 5
model_data["sigma_D"] = 3
model_data["E"] = 500
model_data["nu"] = 0.3
model_data["mu"] = model_data["E"] / (2 * (1 + model_data["nu"]))
model_data["lambda"] = model_data["E"] * model_data["nu"] / ((1 + model_data["nu"]) * (1 - 2 * model_data["nu"]))
model_data["W"] = 4     # width
model_data["H"] = 2     # height
model_data["grid_point_num"] = 51
model_data["grid_point_x"] = model_data["grid_point_num"]
model_data["grid_point_y"] = model_data["grid_point_num"]
model_data["lambda_loss"] = np.array([1e3, 1e-4, 1e1, 1e3, 0]) * 1e0
model_data["lambda_vcut"] = 0.5
model_data["norm_y"] = 1
model_data["norm_p"] = 1
model_data["fno"] = dict()
model_data["fno"]["modes"] = 16
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["input"] = 5
model_data["fno"]["output"] = 8
model_data["fno"]["use_data"] = True
model_data["fno"]["use_grad"] = True
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["normalized"] = True
model_data["dir"] = os.path.join(_WINO_ROOT, 'data')
model_data["path_train"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Trapezoid_G_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_train"]) + '_train.npz')
model_data["path_test"] = os.path.join(model_data["dir"],
                                  'Hyperelasticity_Trapezoid_G_u_s' + str(model_data["grid_point_num"])
                                  + '_n' + str(model_data["n_test"]) + '_test.npz')
model_data["model_filename"] = "FNO_Grad_hyperelasticityTrapezoid_y{}x{}_n{}_ep{}.pth".format(
    model_data["grid_point_y"],
    model_data["grid_point_x"],
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
# train_loader = DataLoader(
#     single_train,
#     batch_size=model_data["batch_size"],
#     shuffle=True,
#     drop_last=True
# )
# test_loader = DataLoader(
#     single_train,
#     batch_size=model_data["batch_size"],
#     shuffle=True,
# )
# model_data["n_train"] = 1
# model_data["n_test"] = 1


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
model = FNO2d_HyperElasticity_Trapezoid(model_data, normalizers).to(device)
# model = FNO2d(model_data).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
print(f"\nmodel_data:{model_data}")

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)


optimizer = SOAP(model.parameters(), lr=model_data["learning_rate_adam"],
                 betas=(.95, .99), weight_decay=0, precondition_frequency=5)
# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=10,  # give SOAP 20 steps to build a robust preconditioner
#     num_training_steps=model_data["num_epoch"]
# )
scheduler = None
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_l2_log, test_l2_log = train_fno_hyperelasticity_trapezoid_Guass(
    model, model_data, train_loader, test_loader, myLoss, optimizer, scheduler, normalizers
)
# save_path = os.path.join(_MODEL_DIR, model_data["model_filename"])
# state_dict = torch.load(save_path, map_location=device)
# model.load_state_dict(state_dict)
print("Training time: ", default_timer() - t1)

folder = _PICTURES_DIR


# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test,
#                                                                          U_test), batch_size=1, shuffle=False)
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
# save model
################################################################
save_path = os.path.join(_MODEL_DIR, model_data["model_filename"])

# Save model parameters (prefer saving only state_dict to save space and keep flexibility)
torch.save(model.state_dict(), save_path)
print(f"Model saved successfully to: {save_path}")


