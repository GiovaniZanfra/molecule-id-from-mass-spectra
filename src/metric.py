"""Métricas suportadas. Para adicionar uma, crie a função e registre em METRICS.

Convenção de previsões:
- binary: vetor (n,) com probabilidade da classe positiva
- regression: vetor (n,)
- multiclass: matriz (n, k) de probabilidades, colunas na ordem de class_names()
- retrieval: y = chave da métrica (chem.metric_key) da resposta, uma por consulta;
  p = uma lista de chaves por consulta, melhor palpite primeiro
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import rankdata


def auc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, np.float64) - np.asarray(p, np.float64)) ** 2)))


def rmsle(y, p) -> float:
    p = np.clip(np.asarray(p, np.float64), 0, None)
    return float(np.sqrt(np.mean((np.log1p(np.asarray(y, np.float64)) - np.log1p(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, np.float64) - np.asarray(p, np.float64))))


def accuracy(y, p) -> float:
    p = np.asarray(p)
    pred = p.argmax(1) if p.ndim == 2 else (p >= 0.5).astype(int)
    return float(np.mean(pred == np.asarray(y)))


def mlogloss(y, p) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum(1, keepdims=True)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=np.int64)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def map3(y, p) -> float:
    p = np.asarray(p)
    top = np.argsort(-p, axis=1)[:, :3]
    y = np.asarray(y)[:, None]
    hit = top == y
    return float(np.mean((hit / np.arange(1, 4)).sum(1)))


MAX_RANK = 25


def reciprocal_ranks(y, p, max_rank: int = MAX_RANK) -> np.ndarray:
    """1/posição do primeiro acerto por consulta (0 se nenhum), como o scorer oficial v13.

    Entradas vazias ("") são descartadas antes de contar posições; chave None (SMILES que não
    parseia) ocupa posição. Lista com mais de max_rank palpites é erro, como no Kaggle.
    """
    out = np.zeros(len(y), dtype=np.float64)
    for i, (truth, guesses) in enumerate(zip(y, p)):
        guesses = [g for g in guesses if g != ""]
        if len(guesses) > max_rank:
            raise ValueError(f"consulta {i}: {len(guesses)} palpites (máximo {max_rank})")
        for rank, g in enumerate(guesses, 1):
            if g is not None and g == truth:
                out[i] = 1.0 / rank
                break
    return out


def mrr25(y, p) -> float:
    return float(reciprocal_ranks(y, p).mean()) if len(y) else float("nan")


@dataclass(frozen=True)
class Metric:
    fn: Callable
    higher_is_better: bool
    tasks: tuple


METRICS: dict[str, Metric] = {
    "auc": Metric(auc, True, ("binary",)),
    "logloss": Metric(logloss, False, ("binary",)),
    "accuracy": Metric(accuracy, True, ("binary", "multiclass")),
    "mlogloss": Metric(mlogloss, False, ("multiclass",)),
    "map3": Metric(map3, True, ("multiclass",)),
    "rmse": Metric(rmse, False, ("regression",)),
    "rmsle": Metric(rmsle, False, ("regression",)),
    "mae": Metric(mae, False, ("regression",)),
    "mrr25": Metric(mrr25, True, ("retrieval",)),
}
