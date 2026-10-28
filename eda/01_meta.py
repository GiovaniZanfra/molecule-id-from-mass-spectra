"""EDA 1 — metadados do train e do teste: bibliotecas × instrumento × aduto × energia, picos, precursor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from chem import ADDUCT_SHIFT, ELECTRON_MASS, TEST_ADDUCTS  # noqa: E402
from common import load_spectra, load_structures, load_test_full, load_train, train_struct  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

m = load_train()
t = load_test_full()
sp = load_spectra()
st = load_structures()
ts = train_struct()
m["clean_peaks"] = sp.n_peaks()
m["test_adduct"] = m["adduct"].isin(TEST_ADDUCTS)
m["n_ce"] = m["collision_energy_ev"].map(lambda x: 0 if x is None else len(x))
m["ce0"] = m["collision_energy_ev"].map(lambda x: np.nan if x is None or len(x) == 0 else float(x[0]))

print("=== por biblioteca")
g = m.groupby("ingest_lib")
tab = pd.DataFrame({
    "spectra": g.size(),
    "structs": g["inchikey14"].nunique(),
    "pos%": g["ionization_mode"].agg(lambda s: 100 * (s == "positive").mean()).round(0),
    "test_add%": g["test_adduct"].mean().mul(100).round(0),
    "MH%": g["adduct"].agg(lambda s: 100 * (s == "[M+H]+").mean()).round(0),
    "raw_pk_med": g["num_peaks"].median(),
    "clean_pk_med": g["clean_peaks"].median(),
    "ce_missing%": g["n_ce"].agg(lambda s: 100 * (s == 0).mean()).round(0),
    "ce_multi%": g["n_ce"].agg(lambda s: 100 * (s > 1).mean()).round(0),
}).sort_values("spectra", ascending=False)
print(tab)

print("\n=== instrumento por biblioteca (top 3)")
for lib, s in g["instrument_type"]:
    vc = s.fillna("None").value_counts(normalize=True).head(3).round(2)
    print(f"{lib:20s}", dict(vc))

print("\n=== adutos (train inteiro, top 20) e no teste")
print(m["adduct"].value_counts().head(20))
print(t["adduct"].value_counts())

print("\n=== energia de colisão: enveda-180 vs teste")
e = m[m["ingest_lib"] == "enveda-180"]
print("e180 n_ce:", e["n_ce"].value_counts().sort_index().to_dict())
print("e180 ce0:", e["ce0"].value_counts().head(12).to_dict())
print("e180 units:", e["collision_energy_orig_units"].value_counts().to_dict())
tce = t["collision_energy_ev"].map(lambda x: 0 if x is None else len(x))
print("test n_ce:", tce.value_counts().sort_index().to_dict())
print("test ce0:", t["collision_energy_ev"].map(lambda x: None if x is None or len(x) == 0 else float(x[0])).value_counts().head(12).to_dict())
print("test instrumento:", t["instrument_type"].value_counts().to_dict())
print("e180 instrumento:", e["instrument_type"].value_counts().to_dict())
npx = m[m["ingest_lib"] == "enveda-np-examples"]
print("np-examples instrumento:", npx["instrument_type"].value_counts().to_dict())
print("np-examples ce0:", npx["ce0"].value_counts().head(8).to_dict(), "n_ce:", npx["n_ce"].value_counts().to_dict())

print("\n=== teste visível")
spm = t.groupby("molecule_id").size()
print("espectros/molécula:", spm.describe().round(2).to_dict())
print("adutos por molécula (n distintos):", t.groupby("molecule_id")["adduct"].nunique().value_counts().to_dict())
print("modos por molécula:", t.groupby("molecule_id")["ionization_mode"].nunique().value_counts().to_dict())
raw = t["ms2_mzs"].map(len)
print("picos crus teste: mediana", raw.median(), " train e180 mediana", e["num_peaks"].median())

print("\n=== espectros por estrutura em e180 e np (linhas por estrutura dentro da biblioteca)")
for lib in ["enveda-180", "enveda-np-examples"]:
    x = m[m["ingest_lib"] == lib].groupby("inchikey14").size()
    print(lib, x.describe().round(1).to_dict())
    xa = m[m["ingest_lib"] == lib].groupby("inchikey14")["adduct"].nunique()
    print("   adutos distintos por estrutura:", xa.value_counts().sort_index().head(8).to_dict())

print("\n=== erro do precursor recalculado (ppm, com sinal) contra a estrutura rotulada")
mass = np.where(ts >= 0, st["mass"].to_numpy()[np.maximum(ts, 0)], np.nan)
shift = m["adduct"].map(ADDUCT_SHIFT).to_numpy(dtype=float)
ion = mass + shift
ppm = (m["precursor_mz"].to_numpy() - ion) / ion * 1e6
m["ppm"] = ppm
mda = (m["precursor_mz"].to_numpy() - ion) * 1e3
m["mda"] = mda
ok = np.isfinite(ppm) & (np.abs(ppm) < 20)
print("fração com |ppm|<20 por biblioteca:")
print(pd.Series(ok).groupby(m["ingest_lib"]).mean().round(3).to_dict())
mm = m[ok]
print(mm.groupby(["ingest_lib", "ionization_mode"])["mda"].agg(["median", "count"]).round(3))
print(mm.groupby(["ingest_lib", "ionization_mode"])["ppm"].agg(["median", lambda s: s.abs().quantile(0.95)]).round(2))
print(f"(massa do elétron = {ELECTRON_MASS * 1e3:.3f} mDa)")

print("\n=== massa das estruturas")
print("todas:", st["mass"].describe().round(0).to_dict())
print("e180:", st.loc[st["in_e180"], "mass"].describe().round(0).to_dict())
print("np:", st.loc[st["in_np"], "mass"].describe().round(0).to_dict())
tm = t["precursor_mz"]
print("teste precursor_mz:", tm.describe().round(0).to_dict())
