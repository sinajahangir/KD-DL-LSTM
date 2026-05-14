#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knowledge distillation for multi-basin streamflow with two input modalities.

Overview
--------
**Teacher** — an LSTM pretrained on a *single* high-quality input sequence (e.g.
many catchment attributes over ``seq_length`` days). It is frozen during KD.

**Student** — an LSTM that sees a *hybrid* window: the same high-resolution
features for the first ``seq_length - 1`` days, and features from a second
(coarser or alternate) source on the *last* day only. That mimics replacing part
of the forcing with an operational product (reanalysis, remote sensing, etc.).

**Loss** — ``MSE(discharge) + beta * MSE(last_hidden_student, last_hidden_teacher)``.

The teacher is frozen; only the student is optimized.

Data layout (generic CSVs)
----------------------------
Expected columns include ``basin_id``,
target discharge ``q``, and feature columns. Train/test CSVs must be row-aligned
by basin and time order between the *primary* and *secondary* files.

Default filenames below are placeholders; rename or symlink your real files.

  - ``data_primary_train.csv`` / ``data_primary_test.csv`` — primary features + ``q``
  - ``data_secondary_train.csv`` / ``data_secondary_test.csv`` — secondary features
  - ``basin_subset_seed_{seed}.csv`` — one ``basin_id`` per row (training subset)

Basins where primary, secondary, teacher-feature, or target rows disagree in
length, or where a 90/10 time split leaves fewer than ``seq_length`` steps in
train or validation, are skipped so ``ConcatDataset`` stays well-defined.

References
----------
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Paths and hyperparameters — replace defaults with your data and checkpoints."""

    # Tabular time series: one row per (basin, day); must align across primary/secondary.
    csv_primary_train: Path = Path("data_primary_train.csv")
    csv_primary_test: Path = Path("data_primary_test.csv")
    csv_secondary_train: Path = Path("data_secondary_train.csv")
    csv_secondary_test: Path = Path("data_secondary_test.csv")
    # One basin id per row; ``{seed}`` is filled from ``seed`` below.
    basin_subset_pattern: str = "basin_subset_seed_{seed}.csv"

    # Frozen teacher LSTM (input size must match teacher feature count).
    teacher_checkpoint: Path = Path("teacher_pretrained.pt")

    # Training
    seed: int = 7
    seq_length: int = 365
    batch_size: int = 256
    beta: float = 20.0  # weight on hidden-state alignment loss
    max_epochs: int = 40
    early_stop_patience: int = 4
    lr: float = 1e-3
    scheduler_patience: int = 2
    scheduler_factor: float = 0.1
    val_every_n_epochs: int = 2

    # Model
    teacher_input_size: int = 32
    student_input_size: int = 30  # after concatenating seq_student channels
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.4

    # Evaluation: number of basin ids to score (0 .. n_basins_eval - 1).
    n_basins_eval: int = 421

    # Output artifacts
    student_save_path: str = "student_lstm_seed_{seed}_beta_{beta}.pt"
    metrics_train_subset_csv: str = "metrics_eval_train_subset_seed_{seed}_beta_{beta}.csv"
    metrics_other_basins_csv: str = "metrics_eval_other_basins_seed_{seed}_beta_{beta}.csv"


CONFIG = Config()  # Edit paths here, or symlink placeholder CSV names to your data.


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def nash_sutcliffe_error(q_obs: np.ndarray, q_sim: np.ndarray) -> float | None:
    if len(q_sim) != len(q_obs):
        return None
    num = float(np.sum(np.square(q_sim - q_obs)))
    den = float(np.sum(np.square(q_obs - np.mean(q_obs))))
    return 1.0 - (num / den)


def pearson_r(pred: np.ndarray, obs: np.ndarray) -> float:
    pred = np.reshape(pred, (-1, 1))
    obs = np.reshape(obs, (-1, 1))
    return float(stats.pearsonr(pred.flatten(), obs.flatten())[0])


def kge(prediction: np.ndarray, observation: np.ndarray) -> float:
    nas = np.logical_or(np.isnan(prediction), np.isnan(observation))
    pred = np.copy(np.reshape(prediction, (-1, 1)))
    obs = np.copy(np.reshape(observation, (-1, 1)))
    r = pearson_r(pred[~nas], obs[~nas])
    beta = float(np.nanmean(pred) / np.nanmean(obs))
    gamma = (float(np.nanstd(pred) / np.nanstd(obs)) / beta) if beta != 0 else np.nan
    return float(1.0 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5)


# ---------------------------------------------------------------------------
# Sequence windows (used in evaluation loop)
# ---------------------------------------------------------------------------


def split_sequence_multi_train(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
    n_steps_in: int,
    n_steps_out: int,
    mode: str = "seq",
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: X shape (n_samples, n_steps_in, n_features), y (n_samples, ...)."""
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    k = 0
    sequence_x = np.copy(np.asarray(sequence_x))
    sequence_y = np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence_x):
            break
        seq_x = sequence_x[k:end_ix]
        if n_steps_out == 0:
            seq_y = sequence_y[end_ix - 1 : out_end_ix]
        elif mode == "single":
            seq_y = sequence_y[out_end_ix - 1]
        else:
            seq_y = sequence_y[end_ix:out_end_ix]
        x_list.append(seq_x)
        y_list.append(seq_y.flatten())
        k += 1
    xx, yy = np.asarray(x_list), np.asarray(y_list)
    if n_steps_out == 0 or n_steps_out == 1:
        yy = yy.reshape((len(xx), 1))
    return xx, yy


# ---------------------------------------------------------------------------
# PyTorch dataset: teacher vs student inputs
# ---------------------------------------------------------------------------
# data1: student — primary features for first seq_length-1 days
# data2: student — secondary features on the last day only (one row expanded)
# data3: teacher — full primary feature sequence (seq_length days)
# targets: discharge aligned to end of window


class TimeSeriesDataset(Dataset):
    """One sample = (teacher_seq, student_seq, target_q_at_window_end)."""

    def __init__(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        data3: np.ndarray,
        targets: np.ndarray,
        seq_length: int,
    ) -> None:
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.data3 = torch.tensor(data3, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.data1) - self.seq_length + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence1 = self.data1[idx : idx + self.seq_length - 1, :]
        seq_teacher = self.data3[idx : idx + self.seq_length, :]
        sequence2 = self.data2[idx + self.seq_length - 1, :].unsqueeze(0)
        seq_student = torch.cat((sequence1, sequence2), dim=0)
        target = self.targets[idx + self.seq_length - 1]
        return seq_teacher, seq_student, target


class LSTMModel(nn.Module):
    """Single-layer LSTM + dropout + linear head; returns (prediction, last_hidden)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout_prob: float = 0.4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        h_last = self.dropout(lstm_out[:, -1, :])
        return self.linear(h_last), lstm_out[:, -1, :]


class KnowledgeDistillation(nn.Module):
    """Wraps frozen teacher and trainable student; forward returns logits + hiddens."""

    def __init__(self, student: nn.Module, teacher: nn.Module, beta: float) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.beta = beta

    def forward(
        self, x_teacher: torch.Tensor, x_student: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_out, student_h = self.student(x_student)
        with torch.no_grad():
            _, teacher_h = self.teacher(x_teacher)
        return student_out, student_h, teacher_h


class DistillationLoss(nn.Module):
    """L = MSE(y_hat, y) + beta * MSE(h_student, h_teacher)."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_out: torch.Tensor,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        true_values: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        return self.mse(student_out, true_values) + beta * self.mse(
            student_hidden, teacher_hidden
        )


# ---------------------------------------------------------------------------
# Data loading and column selection
# ---------------------------------------------------------------------------


def load_and_prepare_data(cfg: Config):
    df = pd.read_csv(cfg.csv_primary_train).dropna()
    df_test = pd.read_csv(cfg.csv_primary_test)
    df_secondary = pd.read_csv(cfg.csv_secondary_train)
    df_test_secondary = pd.read_csv(cfg.csv_secondary_test)

    mean_ = np.asarray(df.iloc[:, 1:].mean())
    std_ = np.asarray(df.iloc[:, 1:].std())
    mean_secondary = np.asarray(df_secondary.iloc[:, 0:].mean())
    std_secondary = np.asarray(df_secondary.iloc[:, 0:].std())

    df_test_tr = df_test.iloc[:, 1:].copy()
    df_test_tr = (df_test_tr - mean_) / std_
    df_test_tr = df_test_tr.drop(columns=["basin_id"])
    df_test_tr["basin_id"] = df_test["basin_id"]

    df_test_tr_secondary = df_test_secondary.iloc[:, 0:].copy()
    df_test_tr_secondary = (df_test_tr_secondary - mean_secondary) / std_secondary
    df_test_tr_secondary = df_test_tr_secondary.drop(columns=["basin_id"])
    df_test_tr_secondary["basin_id"] = df_test_secondary["basin_id"]

    mean_q = float(df["q"].mean())
    std_q = float(df["q"].std())

    df_tr = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / (x.std()), axis=0)
    df_tr = df_tr.drop(columns=["basin_id"])
    df_tr["basin_id"] = df["basin_id"]

    df_tr_secondary = df_secondary.iloc[:, 0:].apply(
        lambda x: (x - x.mean()) / (x.std()), axis=0
    )
    df_tr_secondary = df_tr_secondary.drop(columns=["basin_id"])
    df_tr_secondary["basin_id"] = df_secondary["basin_id"]

    columns = df_tr.columns.to_list()
    for c in ("q", "basin_id", "srad", "vp"):
        columns.remove(c)

    columns_t = df_tr.columns.to_list()
    for c in ("q", "basin_id"):
        columns_t.remove(c)

    # Secondary feature columns: edit to match your CSV (example uses a fixed head + tail slice).
    columns_secondary = [
        "total_precipitation_sum",
        "temperature_2m_max",
        "temperature_2m_min",
    ] + df_test_tr_secondary.columns.to_list()[-28:-1]

    return {
        "df": df,
        "df_tr": df_tr,
        "df_tr_secondary": df_tr_secondary,
        "df_test_tr": df_test_tr,
        "df_test_tr_secondary": df_test_tr_secondary,
        "mean_q": mean_q,
        "std_q": std_q,
        "columns": columns,
        "columns_t": columns_t,
        "columns_secondary": columns_secondary,
    }


def build_concat_datasets(
    data: dict,
    random_numbers: list,
    seq_length: int,
) -> tuple[ConcatDataset, ConcatDataset, list]:
    """Per-basin train/val TimeSeriesDataset objects concatenated; returns skipped ids."""
    df = data["df"]
    df_tr = data["df_tr"]
    df_tr_secondary = data["df_tr_secondary"]
    columns = data["columns"]
    columns_t = data["columns_t"]
    columns_secondary = data["columns_secondary"]

    datasets_train: list[TimeSeriesDataset] = []
    datasets_val: list[TimeSeriesDataset] = []
    skipped: list = []

    for ii in random_numbers:
        data_all = np.asarray(df_tr[df["basin_id"] == ii].loc[:, columns])
        data_all_t = np.asarray(df_tr[df["basin_id"] == ii].loc[:, columns_t])
        data_all_secondary = np.asarray(
            df_tr_secondary[df_tr_secondary["basin_id"] == ii].loc[:, columns_secondary]
        )
        targets = np.asarray(df_tr[df["basin_id"] == ii]["q"]).reshape((-1, 1))

        train_size = int(0.9 * len(data_all))
        train_secondary_size = int(0.9 * len(data_all_secondary))
        val_size = len(data_all) - train_size

        bad = (
            train_size < seq_length
            or train_secondary_size < seq_length
            or val_size < seq_length
            or len(data_all_t) != len(data_all)
            or len(targets) != len(data_all)
        )
        if bad:
            skipped.append(ii)
            continue

        secondary_split = int(0.9 * len(data_all_secondary))
        ds_tr = TimeSeriesDataset(
            data_all[:train_size, :],
            data_all_secondary[:secondary_split, :],
            data_all_t[:train_size, :],
            targets[:train_size, :].reshape((-1, 1)),
            seq_length,
        )
        ds_va = TimeSeriesDataset(
            data_all[train_size:, :],
            data_all_secondary[secondary_split:, :],
            data_all_t[train_size:, :],
            targets[train_size:, :].reshape((-1, 1)),
            seq_length,
        )
        datasets_train.append(ds_tr)
        datasets_val.append(ds_va)

    if not datasets_train:
        raise RuntimeError(
            "No basins passed validation (aligned primary / secondary / teacher columns "
            "and long enough train/val splits). Check CSV integrity and basin_subset_pattern."
        )
    return ConcatDataset(datasets_train), ConcatDataset(datasets_val), skipped


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_main(cfg: Config) -> None:
    set_seed(cfg.seed)
    data = load_and_prepare_data(cfg)

    basin_list_path = Path(cfg.basin_subset_pattern.format(seed=cfg.seed))
    random_numbers = list(pd.read_csv(basin_list_path).iloc[:, 0])
    print(f"Basins listed for training subset: {len(random_numbers)}")

    primary_basins = set(data["df"]["basin_id"].unique())
    secondary_basins = set(data["df_tr_secondary"]["basin_id"].unique())
    rn_set = set(random_numbers)
    print("Basin id coverage in training subset vs CSVs (empty lists = OK):")
    print(f"  In primary only : {sorted((primary_basins - secondary_basins) & rn_set)}")
    print(f"  In secondary only: {sorted((secondary_basins - primary_basins) & rn_set)}")
    print(f"  In neither CSV   : {sorted(rn_set - primary_basins - secondary_basins)}")

    dataset_train, dataset_val, skipped = build_concat_datasets(
        data, random_numbers, cfg.seq_length
    )
    print(f"Training on {len(dataset_train.datasets)} basins; skipped {len(skipped)} basins.")
    if skipped:
        print(f"Skipped basin ids: {sorted(skipped)}")

    loader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = LSTMModel(
        cfg.teacher_input_size,
        cfg.hidden_size,
        cfg.num_layers,
        1,
        cfg.dropout,
    )
    teacher.load_state_dict(torch.load(cfg.teacher_checkpoint, map_location=device, weights_only=True))
    teacher.to(device)

    student = LSTMModel(
        cfg.student_input_size,
        cfg.hidden_size,
        cfg.num_layers,
        1,
        cfg.dropout,
    ).to(device)

    model = KnowledgeDistillation(student, teacher, beta=cfg.beta).to(device)
    criterion = DistillationLoss()
    optimizer = optim.Adam(model.student.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_factor,
        min_lr=1e-6,
    )
    val_loss_fn = nn.MSELoss()

    best_val = float("inf")
    stall = 0
    best_state: dict | None = None

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_t, x_s, y in loader_train:
            x_t = x_t.to(device)
            x_s = x_s.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out, h_s, h_t = model(x_t, x_s)
            loss = criterion(out, h_s, h_t, y, model.beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        if epoch % cfg.val_every_n_epochs != 0:
            continue

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_t, x_s, y in loader_val:
                x_t = x_t.to(device)
                x_s = x_s.to(device)
                y = y.to(device)
                out, _, _ = model(x_t, x_s)
                val_loss += val_loss_fn(out, y).item()
                n_val += 1

        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            stall = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        print(f"Epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f} lr={scheduler.get_last_lr()}")
        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best weights from validation.")

    save_path = cfg.student_save_path.format(seed=cfg.seed, beta=int(cfg.beta))
    torch.save(model.student.state_dict(), save_path)
    print(f"Saved student to {save_path}")


def evaluate_per_basin(cfg: Config, data: dict, random_numbers: list) -> None:
    """NSE/KGE on test period for basin ids 0..420; writes two CSVs (in vs not in training list)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = LSTMModel(
        cfg.student_input_size,
        cfg.hidden_size,
        cfg.num_layers,
        1,
        cfg.dropout,
    ).to(device)
    path = cfg.student_save_path.format(seed=cfg.seed, beta=int(cfg.beta))
    student.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    student.eval()

    columns = data["columns"]
    columns_secondary = data["columns_secondary"]
    std_q = data["std_q"]
    mean_q = data["mean_q"]
    df_test_tr = data["df_test_tr"]
    df_test_tr_secondary = data["df_test_tr_secondary"]

    rn_set = set(random_numbers)
    rows_in_list: list[dict] = []
    rows_not_in_list: list[dict] = []

    for ii in range(0, cfg.n_basins_eval):
        temp_xx = np.asarray(df_test_tr[df_test_tr["basin_id"] == ii].loc[:, columns])
        temp_xx_secondary = np.asarray(
            df_test_tr_secondary[df_test_tr_secondary["basin_id"] == ii].loc[:, columns_secondary]
        )
        temp_yy = np.asarray(df_test_tr[df_test_tr["basin_id"] == ii]["q"]).reshape((-1, 1))
        xx, yy = split_sequence_multi_train(temp_xx, temp_yy, 365, 0, mode="seq")
        xx_secondary, _ = split_sequence_multi_train(temp_xx_secondary, temp_yy, 365, 0, mode="seq")
        if len(xx) == 0:
            continue
        aa = xx[:, :-1, :]
        bb = xx_secondary[:, -1:, :]
        xx_hybrid = np.concatenate((aa, bb), axis=1)
        x_test = torch.tensor(xx_hybrid, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred, _ = student(x_test)
        y_pred = y_pred.cpu().numpy()
        y_obs = yy * std_q + mean_q
        y_sim = y_pred * std_q + mean_q
        nse = nash_sutcliffe_error(y_obs, y_sim)
        kge_v = kge(y_sim, y_obs)
        rec = {"basin_id": ii, "nse": nse, "kge": kge_v}
        if ii in rn_set:
            rows_in_list.append(rec)
        else:
            rows_not_in_list.append(rec)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beta_i = int(cfg.beta)
    path_in = cfg.metrics_train_subset_csv.format(seed=cfg.seed, beta=beta_i)
    path_out = cfg.metrics_other_basins_csv.format(seed=cfg.seed, beta=beta_i)
    pd.DataFrame(rows_in_list).to_csv(path_in, index=False)
    pd.DataFrame(rows_not_in_list).to_csv(path_out, index=False)
    print(f"Wrote metrics: {path_in}, {path_out}")


if __name__ == "__main__":
    cfg = CONFIG
    train_main(cfg)
    data_eval = load_and_prepare_data(cfg)
    basin_list_path = Path(cfg.basin_subset_pattern.format(seed=cfg.seed))
    rnd = list(pd.read_csv(basin_list_path).iloc[:, 0])
    evaluate_per_basin(cfg, data_eval, rnd)
