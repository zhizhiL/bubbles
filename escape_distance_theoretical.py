"""
Comparison of the Stokes-rise prediction L_e ~ U_p * a / V_buoy(D)
(chapter Eq. 1.5) against the observed (D_b, x_escape) data.

Left panel : L_e(D) curve at each U_p with observed events overlaid.
             Each point is one escape event; the gap between the data cloud
             and the curve is the discrepancy quoted in section 1.2.6.
Right panel: inversion. Treat each observed event as if its x_escape *were*
             the Stokes prediction and solve back for D_implied. Compare to
             the actual D_b -- this makes the per-event mismatch explicit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- constants --------------------------------------------------------------
nu   = 1.0e-6    # m^2/s
g    = 9.81      # m/s^2
a_mm = 10       # core radius
D_RING = 40.0    # ring diameter for normalisation

# ---- data -------------------------------------------------------------------
DATA = Path('/home/zl483/bubbles/out')   # change if your data lives elsewhere
SHOTS = [
    ('120, b=1', 'bubbles_03.csv', 120, '#1f77b4'),
    ('120, b=3', 'bubbles_01.csv', 120, '#d62728'),
    ('200, b=1', 'bubbles_06.csv', 200, '#2ca02c'),
    ('200, b=3', 'bubbles_05.csv', 200, '#9467bd'),
]

# ---- Stokes-rise predictions ------------------------------------------------
def Vbuoy_stokes_mms(D_um):
    """Stokes terminal rise velocity [mm/s] for diameter D_um [um]."""
    return g * (D_um * 1e-6)**2 / (18 * nu) * 1000

def Le_stokes_D(Up_mms, D_um):
    """Predicted L_e [mm] from chapter Eq. 1.5."""
    return Up_mms * a_mm / Vbuoy_stokes_mms(D_um)

def D_from_Le(Up_mms, Le_mm):
    """Inversion: D [um] such that Stokes-Le equals Le_mm at given Up."""
    Vb_required = Up_mms * a_mm / Le_mm                       # mm/s
    D_m = np.sqrt(Vb_required / 1000 * 18 * nu / g)
    return D_m * 1e6

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

# left: L_e(D) curves vs observed scatter
D_grid = np.logspace(np.log10(40), np.log10(2500), 300)

for lab, fn, Up, col in SHOTS:
    df = pd.read_csv(DATA / fn)
    Dum = 2 * df['R_world'].values * 1000
    xD  = df['Y_world'].values / D_RING
    ax[0].scatter(Dum, xD, s=12, alpha=0.45, color=col, label=lab,
                  edgecolors='none')

# overlay prediction curves (one per Up, since L_e depends on Up explicitly)
for Up, ls in [(120, '-'), (200, '--')]:
    Le_D = Le_stokes_D(Up, D_grid) / D_RING
    ax[0].plot(D_grid, Le_D, ls, color='k', lw=1.6,
               label=fr'Stokes Eq. 1.5,  $U_p={Up}$ mm/s')

ax[0].axhspan(4, 18, color='grey', alpha=0.08, lw=0,
              label='imaging window')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel(r'bubble diameter $D_b$ (µm)')
ax[0].set_ylabel(r'escape distance  $x/D$')
ax[0].set_title(r'Stokes prediction vs observed $(D_b, x)$')
ax[0].set_xlim(40, 2500); ax[0].set_ylim(0.05, 60)
ax[0].grid(True, which='both', alpha=0.3)
ax[0].legend(fontsize=8, loc='lower left')

# right: per-event implied vs actual D
for lab, fn, Up, col in SHOTS:
    df = pd.read_csv(DATA / fn)
    Dum_obs = 2 * df['R_world'].values * 1000
    x_mm    = df['Y_world'].values
    D_impl  = D_from_Le(Up, x_mm)
    ax[1].scatter(Dum_obs, D_impl, s=12, alpha=0.45, color=col, label=lab,
                  edgecolors='none')

# unity line
lims = (40, 2500)
ax[1].plot(lims, lims, 'k-', lw=1.2, label=r'$D_{\rm implied} = D_b$')
# decade reference lines
for k in [10, 100]:
    ax[1].plot(lims, [v/k for v in lims], 'k:', lw=0.8,
               label=fr'$D_{{\rm implied}} = D_b / {k}$' if k==10 else None)

ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlim(*lims); ax[1].set_ylim(20, 2500)
ax[1].set_xlabel(r'observed diameter $D_b$ (µm)')
ax[1].set_ylabel(r'$D$ implied by Stokes Eq. 1.5 at observed $x$ (µm)')
ax[1].set_title(r'$L_e$ implied vs actual diameter $D$')
ax[1].grid(True, which='both', alpha=0.3)
ax[1].legend(fontsize=8, loc='upper left')

fig.tight_layout()

# save as pdf
fig.savefig('/home/zl483/bubbles/out/stokes_Le_comparison.pdf',
            dpi=140, bbox_inches='tight')

# ---- summary stats ----------------------------------------------------------
print("Summary: ratio of observed x / Stokes-predicted L_e per condition")
print(f"{'condition':<12}{'med D':>8}{'Le_pred(medD)':>16}{'med x':>10}{'ratio x/Le':>14}")
for lab, fn, Up, col in SHOTS:
    df = pd.read_csv(DATA / fn)
    Dum = 2 * df['R_world'].values * 1000
    x_mm = df['Y_world'].values
    medD = np.median(Dum)
    Le_at_medD = Le_stokes_D(Up, medD)
    ratio = np.median(x_mm) / Le_at_medD
    print(f"{lab:<12}{medD:>8.0f}{Le_at_medD:>14.1f} mm"
          f"{np.median(x_mm):>9.1f} mm{ratio:>12.0f}x")

print(f"\nFigure saved: /mnt/user-data/outputs/stokes_Le_comparison.png")