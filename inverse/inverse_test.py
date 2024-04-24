"""Backend supported: pytorch, paddle, jax

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""

import deepxde as dde
import numpy as np
import time
import os

n_iter = 10000
log_every = 25
available_time = False# 2*5 #minutes
log_output_fields = {0: "Ux", 1: "Uy", 2: "Sxx", 3: "Syy", 4: "Sxy"}
net_type = ["spinn", "pfnn"][0]
optimizers = ["adam", "LBFGS"][0]
DIC_measure = True
loss_weights = [1,1,1,1,1,1,1]

if not DIC_measure:
    loss_weights = loss_weights[:5]

if net_type == "spinn":
    dde.config.set_default_autodiff("forward")

lmbd = 1.0
mu = 0.5
Q = 4.0


sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack
pi = dde.backend.as_tensor(np.pi)

if dde.backend.backend_name == "jax":
    import jax.numpy as jnp

geom = dde.geometry.Rectangle([0, 0], [1, 1])


# Exact solutions
def func(x):
    if net_type == "spinn":
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
        x = stack(x_mesh, axis=-1)

    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu

    return np.hstack((ux, uy, Sxx, Syy, Sxy))


def HardBC(x, f):
    if net_type == "spinn":
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
        x = stack(x_mesh, axis=-1)

    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(pi * x[:, 0])
    Sxy = f[:, 4]
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def fx(x):
    return (
        -lmbd
        * (
            4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - mu
        * (
            np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - 8 * mu * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    )


def fy(x):
    return (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
            - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
        )
        - mu
        * (
            2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
    )


def jacobian(f, x, i, j):
    if dde.backend.backend_name == "jax":
        return dde.grad.jacobian(f, x, i=i, j=j)[
            0
        ]  # second element is the function used by jax to compute the gradients
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)


def pde(x, f):
    # x_mesh = jnp.meshgrid(x[:,0].ravel(), x[:,0].ravel(), indexing='ij')
    if net_type == "spinn":
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[:, 0], x[:, 1], indexing="ij")]
        x = stack(x_mesh, axis=1)

    E_xx = jacobian(f, x, i=0, j=0)
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y - fx(x)
    momentum_y = Sxy_x + Syy_y - fy(x)

    if dde.backend.backend_name == "jax":
        f = f[0]  # f[1] is the function used by jax to compute the gradients

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]



bcs = []
num_boundary = 0

n_DIC = 20
# X_DIC = geom.uniform_points(1000, boundary=False)
X_DIC_input = np.stack([np.linspace(0, 1, n_DIC)] * 2, axis=1)
X_DIC_mesh = [x_.ravel() for x_ in np.meshgrid(X_DIC_input[:,0],X_DIC_input[:,1],indexing="ij")]
X_DIC_plot = stack(X_DIC_mesh, axis=1)
if net_type != "spinn":
    X_DIC_input = X_DIC_plot
U_DIC = func(X_DIC_input)


measure_Ux = dde.PointSetBC(X_DIC_input, U_DIC[:, 0:1], component=0)
measure_Uy = dde.PointSetBC(X_DIC_input, U_DIC[:, 1:2], component=1)

if DIC_measure:
    bcs += [measure_Ux, measure_Uy]



def get_num_params(net, input_shape=None):
    if dde.backend.backend_name == "pytorch":
        return sum(p.numel() for p in net.parameters())
    elif dde.backend.backend_name == "paddle":
        return sum(p.numpy().size for p in net.parameters())
    elif dde.backend.backend_name == "jax":
        if input_shape is None:
            raise ValueError("input_shape must be provided for jax backend")
        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        return sum(
            p.size for p in jax.tree_leaves(net.init(rng, jnp.ones(input_shape)))
        )


activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
if net_type == "spinn":
    layers = [32, 32, 32, 32, 5]
    net = dde.nn.SPINN(layers, activation, initializer)
    num_point = 64
    total_points = num_point**2 + num_boundary**2
    num_params = get_num_params(net, input_shape=layers[0])
    X_plot = np.stack([np.linspace(0, 1, 100)] * 2, axis=1)

else:
    layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
    net = dde.nn.PFNN(layers, activation, initializer)
    num_point = 500
    total_points = num_point + num_boundary
    num_params = get_num_params(net, input_shape=layers[0])
    X_mesh = np.meshgrid(
        np.linspace(0, 1, 100, dtype=np.float32),
        np.linspace(0, 1, 100, dtype=np.float32),
        indexing="ij",
    )
    X_plot = np.stack((X_mesh[0].ravel(), X_mesh[1].ravel()), axis=1)


data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=num_point,
    num_boundary=num_boundary,
    solution=func,
    num_test=num_point,
    is_SPINN=net_type == "spinn",
)

net.apply_output_transform(HardBC)


folder_name = f"{net_type}_{available_time if available_time else n_iter}{'min' if available_time else 'iter'}"
dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(dir_path, "results")

# Check if any folders with the same name exist
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]

# If there are existing folders, find the highest number suffix
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    if suffixes:
        max_suffix = max(suffixes)
        folder_name = f"{folder_name}-{max_suffix + 1}"
    else:
        folder_name = f"{folder_name}-1"

# Create the new folder
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

callbacks = [dde.callbacks.Timer(available_time)] if available_time else []
for i, field in log_output_fields.items():
    callbacks.append(dde.callbacks.OperatorPredictor(X_plot, lambda x, output, i=i: output[0][:, i], period=log_every, filename=os.path.join(new_folder_path, f"{field}_history.dat")))

model = dde.Model(data, net)
model.compile(optimizer, lr=0.01, metrics=["l2 relative error"], loss_weights=loss_weights)

start_time = time.time()
losshistory, train_state = model.train(
    iterations=n_iter, callbacks=callbacks, display_every=log_every
)
elapsed = time.time() - start_time


def log_config(fname):
    import json
    import platform
    import psutil

    system_info = {
        "OS": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU count": psutil.cpu_count(),
        "RAM": psutil.virtual_memory().total / (1024**3),
    }

    if dde.backend.backend_name == "pytorch":
        import torch

        gpu_info = {
            "GPU available": torch.cuda.is_available(),
            "GPU device name": torch.cuda.get_device_name(0),
            "GPU device count": torch.cuda.device_count(),
            "CUDA version": torch.version.cuda,
            "Torch version": torch.__version__,
            "CUDNN version": torch.backends.cudnn.version(),
        }
    else:
        gpu_info = {}

    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
        "batch_size": total_points,
        "num_params": num_params,
        "activation": activation,
        "initializer": initializer,
        "optimizer": optimizer,
        "net_type": net_type,
        "layers": layers,
        "logged_fields": log_output_fields,
        "loss_weights": loss_weights,
        "DIC_measure": DIC_measure,
    }

    info = {**system_info, **gpu_info, **execution_info}
    info_json = json.dumps(info, indent=4)

    with open(fname, "w") as f:
        f.write(info_json)


log_config(os.path.join(new_folder_path, "config.json"))
dde.utils.save_loss_history(
    losshistory, os.path.join(new_folder_path, "loss_history.dat")
)
