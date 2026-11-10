import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
HIDDEN = 64
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
SEASONS_PER_SOIL = 1
VAL_SOILS = 500
N_ACTIONS = 5

PARAM_NAMES = ["theta_wp", "theta_fc", "drain_rate"]
REPORT_DAYS = [0, 10, 30, 60, 124]

MODEL_PATH = "checkpoints/soil_estimator.pt"
PLOT_PATH = "results/estimator_error_vs_day.png"
CSV_PATH = "results/estimator_rmse_per_day.csv"

START = time.time()


def log(msg):
    minutes = (time.time() - START) / 60
    print(f"[{minutes:.1f} min] {msg}", flush=True)


def load(path):
    data = np.load(path)
    obs = data["obs"]
    actions = data["actions"][..., None] / (N_ACTIONS - 1)
    x = np.concatenate([obs, actions], axis=-1).astype(np.float32)
    y = data["soil"].astype(np.float32)
    return x, y


class SoilEstimator(nn.Module):
    def __init__(self, n_inputs=6, hidden=HIDDEN, n_outputs=3):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h)


def loss_fn(pred, y):
    target = y[:, None, :].expand_as(pred)
    return ((pred - target) ** 2).mean()


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        return loss_fn(model(x), y).item()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_all, y_all = load("data/soil_dataset_train.npz")
    x_test, y_test = load("data/soil_dataset_test.npz")

    n_val = VAL_SOILS * SEASONS_PER_SOIL
    x_train = x_all[:-n_val]
    y_train = y_all[:-n_val]
    x_val = x_all[-n_val:]
    y_val = y_all[-n_val:]
    log(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")

    mean = y_train.mean(axis=0)
    std = y_train.std(axis=0)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy((y_train - mean) / std)
    xv = torch.from_numpy(x_val)
    yv = torch.from_numpy((y_val - mean) / std)

    loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    model = SoilEstimator()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(xt)
        val_loss = evaluate_loss(model, xv, yv)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " (best)"
        log(f"epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}{marker}")

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "hidden": HIDDEN}, MODEL_PATH)
    log(f"saved best model (val {best_val:.4f}) to {MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test)).numpy() * std + mean
    err = pred - y_test[:, None, :]
    rmse = np.sqrt((err ** 2).mean(axis=0))
    baseline = np.sqrt(((y_test - mean) ** 2).mean(axis=0))

    print()
    print("Test RMSE per day, compared to always guessing the training mean")
    print("day | " + " | ".join(PARAM_NAMES))
    for d in REPORT_DAYS:
        values = " | ".join(f"{rmse[d, i]:.4f}" for i in range(3))
        print(f"{d} | {values}")
    values = " | ".join(f"{baseline[i]:.4f}" for i in range(3))
    print(f"baseline | {values}")

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    days = np.arange(len(rmse))
    np.savetxt(CSV_PATH, np.column_stack([days, rmse]), delimiter=",",
               header="day," + ",".join(PARAM_NAMES), comments="")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.plot(days, rmse[:, i], label="LSTM estimator")
        ax.axhline(baseline[i], linestyle="--", color="gray", label="guess training mean")
        ax.set_title(PARAM_NAMES[i])
        ax.set_xlabel("day of season")
        ax.set_ylabel("RMSE")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Soil estimation error on unseen test soils")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    log(f"saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()