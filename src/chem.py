"""Química do time: chave da métrica (idêntica ao scorer oficial) e propriedades de estruturas.

A chave é a do notebook `metric/casmi-mean-reciprocal-rank` v13: RDKit 2026.03.3,
RemoveStereochemistry → TautomerEnumerator().Canonicalize → InChIKey[:14].
Outra versão do RDKit pode canonicalizar tautômeros diferente — por isso a versão é fixa.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import warnings
from pathlib import Path

RDKIT_VERSION = "2026.03.3"          # como rdkit.__version__ reporta
RDKIT_REQUIREMENT = "rdkit==2026.3.3"  # como o PyPI escreve a mesma versão

# os 10 adutos do teste oculto (página de dados)
TEST_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")

# m/z do íon = massa neutra + ADDUCT_SHIFT[aduto] (carga 1; massas monoisotópicas, elétron incluído)
_H, _E, _O, _N, _C = 1.00782503207, 0.00054857990946, 15.99491461956, 14.0030740048, 12.0
_PROTON = _H - _E
_H2O = 2 * _H + _O
ADDUCT_SHIFT = {
    "[M+H]+": _PROTON,
    "[M+NH4]+": _N + 4 * _H - _E,
    "[M-H2O+H]+": _PROTON - _H2O,
    "[M-2H2O+H]+": _PROTON - 2 * _H2O,
    "[M+Na]+": 22.9897692809 - _E,
    "[M+K]+": 38.96370668 - _E,
    "[M-H]-": -_PROTON,
    "[M-H2O-H]-": -_PROTON - _H2O,
    "[M+CH2O2-H]-": (_C + 2 * _H + 2 * _O) - _PROTON,
    "[M+Cl]-": 34.968852682 + _E,
}
# o teste oculto foi corrigido; íons positivos do enveda-180 no train estão ~0,55 mDa pesados
ELECTRON_MASS = _E


def neutral_mass(precursor_mz, adduct) -> float:
    """Massa neutra a partir do precursor; NaN se o aduto não está na tabela."""
    shift = ADDUCT_SHIFT.get(adduct)
    return float("nan") if shift is None else float(precursor_mz) - shift


_Chem = None
_TAUT = None
_Desc = None
_RMD = None


def _install_wheel():
    """No Kaggle o RDKit não vem na imagem: instala o wheel do dataset metric/rdkit-2026-3-3-wheel."""
    root = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
    wheels = sorted(root.rglob("rdkit-2026.3.3*.whl")) if root.exists() else []
    if not wheels:
        raise ImportError(f"RDKit não instalado e nenhum wheel rdkit-2026.3.3 em {root}. "
                          f"Local: pip install {RDKIT_REQUIREMENT}. Kaggle: anexe metric/rdkit-2026-3-3-wheel.")
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
           f"--find-links={wheels[0].parent}", RDKIT_REQUIREMENT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ImportError(f"falhou: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    importlib.invalidate_caches()


def ensure_rdkit():
    global _Chem, _TAUT, _Desc, _RMD
    if _Chem is not None:
        return
    try:
        import rdkit
    except ImportError:
        _install_wheel()
        import rdkit
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    if rdkit.__version__ != RDKIT_VERSION:
        warnings.warn(f"RDKit {rdkit.__version__} ≠ {RDKIT_VERSION} do scorer: a chave pode divergir.")
    _Chem, _Desc, _RMD = Chem, Descriptors, rdMolDescriptors
    _TAUT = rdMolStandardize.TautomerEnumerator()


def metric_key(smiles) -> str | None:
    """InChIKey14 da métrica oficial, ou None se o SMILES não parseia (None ocupa posição no ranking)."""
    ensure_rdkit()
    if not isinstance(smiles, str):
        return None
    try:
        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        _Chem.RemoveStereochemistry(mol)
        return _Chem.MolToInchiKey(_TAUT.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def structure_info(smiles) -> tuple:
    """(chave da métrica, massa monoisotópica, fórmula) num único parse."""
    ensure_rdkit()
    if not isinstance(smiles, str):
        return None, float("nan"), None
    try:
        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, float("nan"), None
        mass = float(_Desc.ExactMolWt(mol))
        formula = _RMD.CalcMolFormula(mol)
        _Chem.RemoveStereochemistry(mol)
        key = _Chem.MolToInchiKey(_TAUT.Canonicalize(mol))[:14]
        return key, mass, formula
    except (ValueError, RuntimeError, TypeError):
        return None, float("nan"), None


def _pool_map(fn, items, n_jobs, chunksize=256):
    items = list(items)
    n_jobs = n_jobs or os.cpu_count() or 1
    if n_jobs == 1 or len(items) < 2 * chunksize:
        return [fn(x) for x in items]
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context()
    with ctx.Pool(n_jobs) as pool:
        return pool.map(fn, items, chunksize=chunksize)


def metric_keys(smiles, n_jobs: int | None = None) -> list:
    """metric_key em paralelo, na ordem de entrada."""
    return _pool_map(metric_key, smiles, n_jobs)


def structure_infos(smiles, n_jobs: int | None = None) -> list:
    return _pool_map(structure_info, smiles, n_jobs)
