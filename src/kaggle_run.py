"""Roda experimentos na GPU do Kaggle e traz os artefatos de volta para o registry.

    python src/kaggle_run.py setup                         # 1x: sobe folds.npy (+ extras) como dataset privado
    python src/kaggle_run.py run experiments/x.py          # empacota e envia (não espera)
    python src/kaggle_run.py run experiments/x.py --wait   # envia e espera terminar
    python src/kaggle_run.py wait [nome ...]               # espera pendentes, baixa e registra
    python src/kaggle_run.py status                        # lista pendentes
    python src/kaggle_run.py pull <nome> [--from-dir D]    # importa saídas de um kernel terminado
    python src/kaggle_run.py build experiments/x.py        # só monta o pacote (teste, sem enviar)

Precisa do CLI oficial (pip install kaggle) com credencial configurada e das
regras da competição aceitas no site. O CV é sempre recalculado localmente ao importar.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CFG, EXTRA_DIR, FOLDS_PATH, KAGGLE_DIR, KIT_ROOT, NAME_RE, OOF_DIR,  # noqa: E402
                    PRED_DIR, REPORT_DIR, SCRIPTS_DIR, read_registry, register)

SRC = Path(__file__).resolve().parent
TOOLS = {"kaggle_run.py", "audit.py", "hill_climb.py", "make_folds.py", "new_exp.py", "new_note.py",
         "top.py", "make_packet.py"}
BUILD = KAGGLE_DIR / "build"
PENDING = KAGGLE_DIR / "pending.json"
FAILED = KAGGLE_DIR / "failed.json"
LOG_DIR = REPORT_DIR / "kaggle_logs"


# --------------------------------------------------------------------------- utilidades

def kaggle_user() -> str:
    user = CFG.get("kaggle_username") or os.environ.get("KAGGLE_USERNAME", "")
    if not user:
        cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        cred = cfg_dir / "kaggle.json"
        if cred.exists():
            user = json.loads(cred.read_text()).get("username", "")
    if not user:
        sys.exit("não sei seu usuário do Kaggle: preencha kaggle_username no comp.json")
    return user


def prefix() -> str:
    if CFG.get("kaggle_kernel_prefix"):
        return CFG["kaggle_kernel_prefix"]
    m = re.search(r"s\d+e\d+", CFG["competition"])
    return m.group(0) if m else re.sub(r"[^a-z0-9]+", "-", CFG["competition"].lower())[:20].strip("-")


def kernel_slug(name: str) -> str:
    slug = f"{prefix()}-{name}".lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    if len(slug) > 50:
        sys.exit(f"slug do kernel '{slug}' passa de 50 caracteres; encurte kaggle_kernel_prefix")
    return slug


def folds_dataset() -> str:
    return f"{kaggle_user()}/{CFG.get('kaggle_folds_dataset') or prefix() + '-kit-folds'}"


def cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        out = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("CLI do Kaggle não encontrado: pip install kaggle (e configure a credencial)")
    if check and out.returncode != 0:
        print(out.stdout, out.stderr, sep="\n")
        sys.exit(f"falhou: kaggle {' '.join(args)}")
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- setup

def cmd_setup(_args):
    if not FOLDS_PATH.exists():
        sys.exit("folds.npy não existe — rode python src/make_folds.py antes")
    ds_id = folds_dataset()
    slug = ds_id.split("/", 1)[1]
    d = KAGGLE_DIR / "folds_dataset"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(FOLDS_PATH, d / "folds.npy")
    meta = FOLDS_PATH.with_suffix(".meta.json")
    if meta.exists():
        shutil.copy(meta, d / "folds.meta.json")
    extras = CFG.get("extra_files") or []
    for name in extras:
        src = EXTRA_DIR / name
        if not src.exists():
            sys.exit(f"extra_files lista {name}, mas {src} não existe")
        shutil.copy(src, d / name)
    title = slug if len(slug) >= 6 else f"{slug}-folds"
    save_json(d / "dataset-metadata.json", {"title": title, "id": ds_id, "licenses": [{"name": "CC0-1.0"}]})
    out = cli("datasets", "create", "-p", str(d), check=False)
    text = (out.stdout + out.stderr).lower()
    if out.returncode == 0 and "error" not in text:
        print(f"[ok] dataset privado criado: {ds_id} (folds + {len(extras)} extras)")
        print("     pode levar alguns minutos até ficar disponível para os kernels")
        return
    if "already" in text or "exists" in text:
        cli("datasets", "version", "-p", str(d), "-m", f"update {dt.datetime.now():%Y-%m-%d %H:%M}")
        print(f"[ok] nova versão do dataset {ds_id}")
        return
    print(out.stdout, out.stderr, sep="\n")
    sys.exit("não consegui criar o dataset (veja a mensagem acima)")


# --------------------------------------------------------------------------- build / run

HEADER = '''# === bootstrap do kit (gerado por src/kaggle_run.py — não edite) ===
import base64 as _b64, os as _os, sys as _sys
_KIT = _os.path.join(_os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"), "_kit")
_FILES = {files}
for _p, _c in _FILES.items():
    _full = _os.path.join(_KIT, _p)
    _os.makedirs(_os.path.dirname(_full), exist_ok=True)
    with open(_full, "wb") as _f:
        _f.write(_b64.b64decode(_c))
_os.environ.update({{"KIT_ON_KAGGLE": "1", "KIT_ROOT": _KIT, "KIT_EXP_NAME": "{name}"}})
_sys.path.insert(0, _os.path.join(_KIT, "src"))
# === fim do bootstrap ===
'''


def build(exp: Path, gpu: bool = True, internet: bool = False) -> tuple[str, Path]:
    exp = Path(exp)
    name = exp.stem
    if not NAME_RE.match(name):
        sys.exit(f"{exp.name}: nome fora do padrão <familia>_<autor>_<NNN> (use src/new_exp.py)")
    code = exp.read_text(encoding="utf-8")
    if "level=2" in code or "load_oof(" in code:
        sys.exit("este experimento lê OOFs locais (stacking) — rode local: python " + str(exp))
    files = {f"src/{p.name}": base64.b64encode(p.read_bytes()).decode()
             for p in sorted(SRC.glob("*.py")) if p.name not in TOOLS}
    files["comp.json"] = base64.b64encode((KIT_ROOT / "comp.json").read_bytes()).decode()
    d = BUILD / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / f"{name}.py").write_text(HEADER.format(files=json.dumps(files), name=name) + code, encoding="utf-8")
    user = kaggle_user()
    slug = kernel_slug(name)
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": f"{name}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu), "enable_internet": bool(internet),
        "dataset_sources": [folds_dataset()], "competition_sources": [CFG["competition"]], "kernel_sources": [],
    }
    meta.update(CFG.get("kaggle_kernel_extra") or {})
    save_json(d / "kernel-metadata.json", meta)
    return name, d


def cmd_build(args):
    name, d = build(Path(args.exp), gpu=not args.no_gpu, internet=args.internet)
    print(f"[ok] pacote de {name} em {d}")


def cmd_run(args):
    exp = Path(args.exp)
    name, d = build(exp, gpu=not args.no_gpu, internet=args.internet)
    if name in set(read_registry()["name"]):
        sys.exit(f"{name} já está no registry — crie outro experimento com src/new_exp.py --from {name}")
    out = cli("kernels", "push", "-p", str(d), check=False)
    text = out.stdout + out.stderr
    if out.returncode != 0 or "successfully pushed" not in text.lower():
        print(text)
        sys.exit("push falhou. Se for limite de GPU/sessões do Kaggle, espere os pendentes (kaggle_run.py wait).")
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exp, SCRIPTS_DIR / f"{name}.py")
    pend = load_json(PENDING)
    pend[name] = {"kernel": json.loads((d / "kernel-metadata.json").read_text())["id"],
                  "pushed_at": dt.datetime.now().isoformat(timespec="seconds"), "exp": str(exp)}
    save_json(PENDING, pend)
    print(f"[ok] {name} enviado → https://www.kaggle.com/code/{pend[name]['kernel']}")
    if args.wait:
        wait([name], args.timeout, args.interval)
    else:
        print("     quando quiser importar: python src/kaggle_run.py wait")


# --------------------------------------------------------------------------- status / wait / pull

def kernel_status(kernel: str) -> str:
    out = cli("kernels", "status", kernel, check=False)
    text = (out.stdout + out.stderr).lower()
    for key in ("complete", "error", "cancel", "running", "queued"):
        if key in text:
            return "error" if key == "cancel" else key
    return "unknown"


def cmd_status(_args):
    pend = load_json(PENDING)
    if not pend:
        print("nenhum kernel pendente")
        return
    for name, info in pend.items():
        print(f"{name:30s} {kernel_status(info['kernel']):10s} enviado {info['pushed_at']}")


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return "(sem log)"
    text = path.read_text(encoding="utf-8", errors="replace")
    try:  # o log do Kaggle costuma ser uma lista JSON de {stream_name, data}
        text = "".join(e.get("data", "") for e in json.loads(text))
    except Exception:
        pass
    return "\n".join(text.splitlines()[-n:])


def pull(name: str, from_dir: Path | None = None, force: bool = False) -> bool:
    pend = load_json(PENDING)
    if from_dir is None:
        if name not in pend:
            sys.exit(f"{name} não está nos pendentes; use --from-dir com a pasta de saída baixada")
        from_dir = BUILD / name / "output"
        if from_dir.exists():
            shutil.rmtree(from_dir)
        from_dir.mkdir(parents=True)
        cli("kernels", "output", pend[name]["kernel"], "-p", str(from_dir))
    from_dir = Path(from_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted(from_dir.rglob("*.log"))
    if logs:
        shutil.copyfile(logs[0], LOG_DIR / f"{name}.log")
    bundles = sorted(from_dir.rglob(f"{name}_artifacts.npz"))
    if not bundles:
        print(f"[erro] {name}: kernel terminou sem {name}_artifacts.npz (o finish() rodou?). Final do log:")
        print(_tail(LOG_DIR / f"{name}.log"))
        return False
    z = np.load(bundles[0], allow_pickle=False)
    dests = {"oof": OOF_DIR / f"{name}.npy", "preds": PRED_DIR / f"{name}.npy", "meta": OOF_DIR / f"{name}.meta.json"}
    for dest in dests.values():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            sys.exit(f"{dest} já existe (artefatos não são sobrescritos)")
    np.save(dests["oof"], z["oof"])
    np.save(dests["preds"], z["preds"])
    dests["meta"].write_text(str(z["meta"]), encoding="utf-8")
    register(name, where="kaggle")
    if name in pend:
        pend.pop(name)
        save_json(PENDING, pend)
    return True


def wait(names: list[str] | None, timeout_min: float, interval: float):
    pend = load_json(PENDING)
    targets = [n for n in (names or list(pend)) if n in pend]
    if not targets:
        print("nada pendente")
        return
    t0 = time.time()
    while targets:
        for name in list(targets):
            st = kernel_status(pend[name]["kernel"])
            if st == "complete":
                pull(name)
                targets.remove(name)
            elif st == "error":
                out_dir = BUILD / name / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                cli("kernels", "output", pend[name]["kernel"], "-p", str(out_dir), check=False)
                logs = sorted(out_dir.rglob("*.log"))
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                if logs:
                    shutil.copyfile(logs[0], LOG_DIR / f"{name}.log")
                print(f"[erro] {name} falhou no Kaggle. Final do log:\n{_tail(LOG_DIR / f'{name}.log')}")
                failed = load_json(FAILED)
                current = load_json(PENDING)
                failed[name] = current.pop(name, pend[name])
                save_json(FAILED, failed)
                save_json(PENDING, current)
                targets.remove(name)
        if not targets:
            break
        if time.time() - t0 > timeout_min * 60:
            print(f"timeout: ainda rodando {targets}. Rode wait de novo depois.")
            return
        print(f"[{dt.datetime.now():%H:%M}] aguardando {len(targets)}: {', '.join(targets)}", flush=True)
        time.sleep(interval)


def cmd_wait(args):
    wait(args.names or None, args.timeout, args.interval)


def cmd_pull(args):
    pull(args.name, Path(args.from_dir) if args.from_dir else None, force=args.force)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    for cmd, fn in (("build", cmd_build), ("run", cmd_run)):
        p = sub.add_parser(cmd)
        p.add_argument("exp")
        p.add_argument("--no-gpu", action="store_true")
        p.add_argument("--internet", action="store_true")
        if cmd == "run":
            p.add_argument("--wait", action="store_true")
            p.add_argument("--timeout", type=float, default=240, help="minutos")
            p.add_argument("--interval", type=float, default=60, help="segundos entre consultas")
        p.set_defaults(fn=fn)
    p = sub.add_parser("wait")
    p.add_argument("names", nargs="*")
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--interval", type=float, default=60)
    p.set_defaults(fn=cmd_wait)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("pull")
    p.add_argument("name")
    p.add_argument("--from-dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_pull)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
