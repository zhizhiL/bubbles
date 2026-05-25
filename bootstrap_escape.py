"""
Nonparametric bootstrap for bubble-escape statistics.

For each shot, resamples rows of (X, Y, R) with replacement B times and
reports the 95% CI on each summary statistic. No model assumption beyond
"observed events are an iid sample from the shot's empirical distribution".

Within-shot CIs only -- shot-to-shot variability is not captured (would
need a cluster bootstrap with >>2 repeats per condition).

test_04 excluded (suspected ill-run); test_06 used for (200, b=1).
"""

import numpy as np
import pandas as pd
from scipy import stats

# ---- config -----------------------------------------------------------------
B      = 2000           # bootstrap iterations
CI     = 95             # CI level (%)
SEED   = 0
D_RING = 46.0           # mm, ring diameter for normalisation
DATA   = "/mnt/user-data/uploads"

SHOTS = [
    # label,      file,             Up,  b, V_nom (uL)
    ("120, b=1",  "bubbles_03.csv", 120, 1, 20),
    ("120, b=3",  "bubbles_01.csv", 120, 3, 40),
    ("200, b=1",  "bubbles_06.csv", 200, 1, 20),
    ("200, b=3",  "bubbles_05.csv", 200, 3, 40),
]

# ---- statistics: each takes (R, Y, X) in mm and returns a scalar ------------
def s_Vtot(R, Y, X):                                    # total escaped vol, uL
    return (4/3 * np.pi * R**3).sum()

def s_medD(R, Y, X):                                    # median bubble diam, um
    return np.median(2 * R * 1000)

def s_cntYD(R, Y, X):                                   # count-median Y/D
    return np.median(Y) / D_RING

def s_volYD(R, Y, X):                                   # volume-weighted med Y/D
    V = R**3                                            # ∝ bubble volume
    order = np.argsort(Y)
    cumF = np.cumsum(V[order]) / V.sum()
    return Y[order][np.searchsorted(cumF, 0.5)] / D_RING

def s_Yspan(R, Y, X):                                   # 5-95 pctile range, /D
    return (np.percentile(Y, 95) - np.percentile(Y, 5)) / D_RING

def s_sigmaX(R, Y, X):                                  # lateral spread, mm
    return X.std(ddof=1)

def s_rho(R, Y, X):                                     # Spearman ρ(D, Y)
    return stats.spearmanr(R, Y).statistic

STATS = {
    "V_count (uL)": s_Vtot,
    "med D (um)":   s_medD,
    "cnt-Y/D":      s_cntYD,
    "vol-Y/D":      s_volYD,
    "Yspan/D":      s_Yspan,
    "sigma_X (mm)": s_sigmaX,
    "rho(D, Y)":    s_rho,
}

# ---- bootstrap --------------------------------------------------------------
def bootstrap_shot(R, Y, X, stat_fns, B, rng):
    """B resamples of the rows; returns {stat_name: array of B values}."""
    n   = len(R)
    out = {name: np.empty(B) for name in stat_fns}
    for b in range(B):
        idx = rng.integers(0, n, n)
        Rb, Yb, Xb = R[idx], Y[idx], X[idx]
        for name, fn in stat_fns.items():
            out[name][b] = fn(Rb, Yb, Xb)
    return out

def ci_from_boots(arr, ci):
    a = (100 - ci) / 2
    return np.percentile(arr, a), np.percentile(arr, 100 - a)

# ---- main -------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    print(f"Nonparametric bootstrap: B = {B}, CI = {CI}%, seed = {SEED}\n")

    results = {}
    for lab, fn, Up, b, Vnom in SHOTS:
        df    = pd.read_csv(f"{DATA}/{fn}")
        R, Y, X = df['R_world'].values, df['Y_world'].values, df['X_world'].values
        point = {name: f(R, Y, X) for name, f in STATS.items()}
        boots = bootstrap_shot(R, Y, X, STATS, B, rng)
        cis   = {name: ci_from_boots(arr, CI) for name, arr in boots.items()}
        results[lab] = dict(point=point, ci=cis, Vnom=Vnom, N=len(R), boots=boots)

    # table: stats as rows, shots as columns
    cell_w = 24
    hdr = f"{'stat':<14}" + "".join(f"{lab:>{cell_w}}" for lab, *_ in SHOTS)
    print(hdr)
    print("-" * len(hdr))

    def fmt(p, lo, hi):
        return f"  {p:7.2f} [{lo:6.2f}, {hi:6.2f}]"

    for name in STATS:
        row = f"{name:<14}"
        for lab, *_ in SHOTS:
            p     = results[lab]['point'][name]
            lo, hi = results[lab]['ci'][name]
            row  += fmt(p, lo, hi)
        print(row)

    # derived: V_count / V_nom with CI propagated from V_count bootstrap
    row = f"{'V/V_nom':<14}"
    for lab, *_ in SHOTS:
        d  = results[lab]
        Vn = d['Vnom']
        p  = d['point']['V_count (uL)'] / Vn
        lo, hi = (x / Vn for x in d['ci']['V_count (uL)'])
        row += fmt(p, lo, hi)
    print(row)

    print(f"\nN per shot: " + ", ".join(f"{lab}={results[lab]['N']}" for lab, *_ in SHOTS))

    return results

if __name__ == "__main__":
    results = main()
