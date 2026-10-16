"""Espectros em arrays planos: limpeza padrão, cache do train e similaridade cosseno.

O train tem ~400M picos; como listas no pandas não cabe em 11 GB. Aqui cada conjunto de espectros
é um `Spectra`: mz e intensidade concatenados (float32) + offsets, picos do espectro i em
mz[off[i]:off[i+1]], ordenados por m/z, intensidade relativa ao maior pico (= 1).

Limpeza padrão (CLEAN), igual para train e teste: remove picos acima de precursor + 2 Da (o teste
já vem assim), renormaliza pelo maior pico restante, remove picos < min_rel e guarda os top_k mais
intensos. Limpeza mais forte (top 60, raiz da intensidade...) é escolha do experimento.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLEAN = dict(max_above_precursor=2.0, min_rel=0.001, top_k=128)
CACHE_VERSION = 1


@dataclass
class Spectra:
    mz: np.ndarray      # float32, todos os picos concatenados
    inten: np.ndarray   # float32, mesmo tamanho
    off: np.ndarray     # int64, len = n_espectros + 1

    def __len__(self) -> int:
        return len(self.off) - 1

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.off[i], self.off[i + 1]
        return np.asarray(self.mz[a:b]), np.asarray(self.inten[a:b])

    def n_peaks(self) -> np.ndarray:
        return np.diff(self.off)

    def take(self, idx) -> "Spectra":
        """Subconjunto compacto (cópia), na ordem de idx."""
        idx = np.asarray(idx, dtype=np.int64)
        a, b = self.off[idx], self.off[idx + 1]
        n = b - a
        flat = _ranges(a, b)
        off = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(n, out=off[1:])
        return Spectra(np.asarray(self.mz[flat]), np.asarray(self.inten[flat]), off)


def _ranges(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Concatena range(lo[i], hi[i]) para todo i, vetorizado."""
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    return starts + np.arange(total, dtype=np.int64)


def clean_flat(mz: np.ndarray, inten: np.ndarray, lengths: np.ndarray, precursor: np.ndarray,
               max_above_precursor: float = CLEAN["max_above_precursor"],
               min_rel: float = CLEAN["min_rel"], top_k: int = CLEAN["top_k"]) -> Spectra:
    """Limpeza vetorizada de muitos espectros dados como arrays planos + número de picos por espectro."""
    n = len(lengths)
    seg = np.repeat(np.arange(n, dtype=np.int64), lengths)
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    prec = np.asarray(precursor, dtype=np.float64)
    limit = np.where(np.isfinite(prec), prec + max_above_precursor, np.inf)
    keep = np.isfinite(mz) & np.isfinite(inten) & (inten > 0) & (mz <= limit[seg])
    mz, inten, seg = mz[keep], inten[keep], seg[keep]
    order = np.lexsort((-inten, seg))             # por espectro, mais intenso primeiro
    mz, inten, seg = mz[order], inten[order], seg[order]
    counts = np.bincount(seg, minlength=n)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(seg)) - first[seg]
    base = np.zeros(n)
    has = counts > 0
    base[has] = inten[first[has]]
    rel = inten / base[seg]
    keep = (rank < top_k) & (rel >= min_rel)
    mz, rel, seg = mz[keep], rel[keep], seg[keep]
    order = np.lexsort((mz, seg))
    mz, rel, seg = mz[order], rel[order], seg[order]
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(seg, minlength=n), out=off[1:])
    return Spectra(mz.astype(np.float32), rel.astype(np.float32), off)


def clean_frame(df) -> Spectra:
    """Limpa os espectros de um DataFrame pequeno (ex.: teste) com colunas de picos e precursor_mz."""
    mzs = [np.asarray(x if x is not None else [], dtype=np.float64) for x in df["ms2_mzs"]]
    ints = [np.asarray(x if x is not None else [], dtype=np.float64) for x in df["ms2_normalized_intensities"]]
    lengths = np.array([len(x) for x in mzs], dtype=np.int64)
    flat_mz = np.concatenate(mzs) if len(mzs) else np.zeros(0)
    flat_int = np.concatenate(ints) if len(ints) else np.zeros(0)
    return clean_flat(flat_mz, flat_int, lengths, df["precursor_mz"].to_numpy(dtype=np.float64))


def build_cache(parquet_path: Path, out_dir: Path) -> Path:
    """Lê o train por row group, limpa e grava mz/inten/off em .npy (abre depois com mmap)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    t0 = time.time()
    parts_mz, parts_int, lens = [], [], []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
        lm = pc.fill_null(pc.list_value_length(t["ms2_mzs"]), 0).to_numpy()
        li = pc.fill_null(pc.list_value_length(t["ms2_normalized_intensities"]), 0).to_numpy()
        if not np.array_equal(lm, li):
            raise ValueError(f"row group {g}: ms2_mzs e intensidades com tamanhos diferentes")
        s = clean_flat(pc.list_flatten(t["ms2_mzs"]).to_numpy(),
                       pc.list_flatten(t["ms2_normalized_intensities"]).to_numpy(),
                       lm.astype(np.int64), t["precursor_mz"].to_numpy(zero_copy_only=False))
        parts_mz.append(s.mz)
        parts_int.append(s.inten)
        lens.append(np.diff(s.off))
        print(f"  row group {g + 1}/{pf.num_row_groups}: {len(s):,} espectros, {len(s.mz):,} picos "
              f"({time.time() - t0:.0f}s)", flush=True)
    off = np.zeros(sum(len(x) for x in lens) + 1, dtype=np.int64)
    np.cumsum(np.concatenate(lens), out=off[1:])
    np.save(out_dir / "mz.npy", np.concatenate(parts_mz))
    np.save(out_dir / "inten.npy", np.concatenate(parts_int))
    np.save(out_dir / "off.npy", off)
    (out_dir / "params.txt").write_text(f"version={CACHE_VERSION} {CLEAN}\n")
    return out_dir


def load_cache(cache_dir: Path, mmap: bool = True) -> Spectra:
    mode = "r" if mmap else None
    return Spectra(np.load(cache_dir / "mz.npy", mmap_mode=mode), np.load(cache_dir / "inten.npy", mmap_mode=mode),
                   np.load(cache_dir / "off.npy"))


# --------------------------------------------------------------------------- similaridade

def weigh(mz: np.ndarray, inten: np.ndarray, top: int = 60, power: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Top-N picos, intensidade^power, norma L2 = 1, ordenado por m/z."""
    if len(mz) > top:
        keep = np.sort(np.argpartition(-inten, top - 1)[:top])
        mz, inten = mz[keep], inten[keep]
    w = np.asarray(inten, dtype=np.float64) ** power
    norm = np.sqrt((w * w).sum())
    return np.asarray(mz, dtype=np.float64), (w / norm if norm > 0 else w)


def weigh_many(s: Spectra, top: int = 60, power: float = 0.5) -> Spectra:
    """weigh() para todos os espectros de um Spectra (pequeno)."""
    mzs, ws = [], []
    for i in range(len(s)):
        m, w = weigh(*s.get(i), top=top, power=power)
        mzs.append(m)
        ws.append(w)
    off = np.zeros(len(s) + 1, dtype=np.int64)
    np.cumsum([len(m) for m in mzs], out=off[1:])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    return Spectra(cat(mzs), cat(ws), off)


def cosine_one_to_many(q_mz: np.ndarray, q_w: np.ndarray, lib: Spectra, tol: float = 0.01,
                       rounds: int = 10) -> np.ndarray:
    """Cosseno de um espectro (já pesado) contra todos os de `lib` (já pesados), vetorizado.

    Casamento um-para-um por melhores mútuos, em rodadas: o par (i, j) conta se j é o melhor pico de i
    naquele espectro e i é o melhor de j; os picos casados saem e a rodada se repete. Nunca passa do
    cosseno guloso e converge para ele.
    """
    n = len(lib)
    out = np.zeros(n)
    if n == 0 or len(q_mz) == 0 or len(lib.mz) == 0:
        return out
    lib_seg = np.repeat(np.arange(n), np.diff(lib.off))
    order = np.argsort(lib.mz, kind="stable")
    L_mz, L_w, L_seg = np.asarray(lib.mz)[order], np.asarray(lib.inten)[order], lib_seg[order]
    lo = np.searchsorted(L_mz, q_mz - tol, "left")
    hi = np.searchsorted(L_mz, q_mz + tol, "right")
    j = _ranges(lo, hi)
    if len(j) == 0:
        return out
    i = np.repeat(np.arange(len(q_mz)), hi - lo)
    prod = q_w[i] * L_w[j]
    seg = L_seg[j]
    qkey = seg * len(q_mz) + i
    for _ in range(rounds):
        mutual = np.intersect1d(_group_argmax(qkey, prod), _group_argmax(j, prod), assume_unique=True)
        out += np.bincount(seg[mutual], weights=prod[mutual], minlength=n)
        rest = ~np.isin(qkey, qkey[mutual]) & ~np.isin(j, j[mutual])
        if not rest.any():
            break
        qkey, j, seg, prod = qkey[rest], j[rest], seg[rest], prod[rest]
    return out


def _group_argmax(keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Índice (em keys/vals) do maior valor de cada grupo."""
    order = np.lexsort((-vals, keys))
    k = keys[order]
    first = np.concatenate([[True], k[1:] != k[:-1]])
    return order[first]
