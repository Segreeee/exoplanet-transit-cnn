"""CNN a due torri (vista globale + vista locale) con feature scalari, in PyTorch.

Ispirata all'idea di AstroNet (Shallue & Vanderburg, 2018), ma scritta da zero e piu'
piccola: ~0.3 M parametri con i default, adatta a una GPU da 4-8 GB (e anche alla CPU).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from exotransit.metrics import evaluate_scores
from exotransit.preprocess import GLOBAL_BINS, LOCAL_BINS, N_EXTRA


def _tower(base: int, stages: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 1
    for s in range(stages):
        ch = base * 2**s
        layers += [
            nn.Conv1d(in_ch, ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        ]
        in_ch = ch
    # pooling adattivo a lunghezza fissa: mantiene la posizione (fase) e rende il modello
    # indipendente dal numero di bin
    layers += [nn.AdaptiveAvgPool1d(8), nn.Flatten()]
    return nn.Sequential(*layers)


class TwoTowerCNN(nn.Module):
    """Input: vista globale (B, G), vista locale (B, L), feature scalari (B, n_extra).
    Output: logit (B,) della classe "pianeta"."""

    def __init__(
        self,
        n_extra: int = N_EXTRA,
        base: int = 16,
        global_stages: int = 4,
        local_stages: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.global_tower = _tower(base, global_stages)
        self.local_tower = _tower(base, local_stages)
        self.extra_norm = nn.BatchNorm1d(n_extra) if n_extra > 0 else None
        feat = 8 * base * 2 ** (global_stages - 1) + 8 * base * 2 ** (local_stages - 1) + n_extra
        self.head = nn.Sequential(
            nn.Linear(feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, global_view, local_view, extra=None):
        parts = [
            self.global_tower(global_view.unsqueeze(1)),
            self.local_tower(local_view.unsqueeze(1)),
        ]
        if self.extra_norm is not None:
            parts.append(self.extra_norm(extra))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, global_view, local_view, extra=None, batch_size: int = 256):
        self.eval()
        device = next(self.parameters()).device
        out = []
        for i in range(0, len(global_view), batch_size):
            sl = slice(i, i + batch_size)
            g = torch.as_tensor(global_view[sl], dtype=torch.float32, device=device)
            l_ = torch.as_tensor(local_view[sl], dtype=torch.float32, device=device)
            e = (
                None
                if extra is None
                else torch.as_tensor(extra[sl], dtype=torch.float32, device=device)
            )
            out.append(torch.sigmoid(self(g, l_, e)).cpu().numpy())
        return np.concatenate(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FitResult:
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_pr_auc: float = float("nan")


def _to_loader(g, l_, e, y, batch_size, shuffle):
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (g, l_, e, y)]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def fit(
    model: TwoTowerCNN,
    train: tuple,
    val: tuple,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 8,
    device: torch.device | None = None,
    seed: int = 0,
) -> FitResult:
    """Addestra con BCE (pos_weight per lo sbilanciamento) ed early stopping sulla PR-AUC di
    validazione. `train` e `val` sono tuple (global, local, extra, y). Ripristina i pesi
    migliori alla fine."""
    torch.manual_seed(seed)
    device = device or pick_device()
    model.to(device)

    y_train = np.asarray(train[3], dtype=np.float32)
    n_pos = max(float(y_train.sum()), 1.0)
    pos_weight = torch.tensor((len(y_train) - n_pos) / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _to_loader(*train, batch_size, shuffle=True)

    result = FitResult()
    best_state, stale = None, 0
    best_score, best_loss = -np.inf, np.inf
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for g, l_, e, y in loader:
            g, l_, e, y = (t.to(device) for t in (g, l_, e, y))
            opt.zero_grad()
            loss = loss_fn(model(g, l_, e), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        scores = model.predict_proba(val[0], val[1], val[2])
        m = evaluate_scores(np.asarray(val[3]), scores)
        val_loss = float(log_loss(val[3], np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1]))
        result.history.append(
            {"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss, **m}
        )
        score = m["pr_auc"] if not np.isnan(m["pr_auc"]) else -1.0
        # a parita' di PR-AUC (es. problemi facili che arrivano a 1.0) vince la loss di
        # validazione piu' bassa: evita di restituire il modello dell'epoca 0
        improved = score > best_score + 1e-4 or (
            abs(score - best_score) <= 1e-4 and val_loss < best_loss
        )
        if improved:
            best_score, best_loss = score, val_loss
            result.best_val_pr_auc, result.best_epoch = m["pr_auc"], epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return result


__all__ = [
    "GLOBAL_BINS",
    "LOCAL_BINS",
    "FitResult",
    "TwoTowerCNN",
    "count_parameters",
    "fit",
    "pick_device",
]
