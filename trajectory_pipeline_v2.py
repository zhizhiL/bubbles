"""
Bubble-laden vortex ring trajectory analysis pipeline.

Implements the data-processing steps described in §3 of the companion
LaTeX document: per-shot cleaning, Model 1/2/3 fits with t_launch=0
enforced, F-test model selection, physical readout (alpha, C_eff,
A=w_inf/tau), and cross-shot ratio tests / universal-curve diagnostics.

Expected input per shot: CSV with three rows (time, x, z) and no header,
each row being a comma-separated 1D array of length n. See e.g.
new_trajectory.csv from the conversation.

Usage:
    results = analyse_shot('shot.csv', V_b_uL=40, U_p=120,
                            R_mm=23, a_mm=7, label='U120_V40_rep1')
    fig = plot_shot_diagnostics(results)

For cross-shot analysis, accumulate `results` dicts and pass to
`compare_shots(results_list)`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import stats


# ----------------------------- physical constants -------------------------- #
G = 9810.0  # mm/s^2


def V_ring(R_mm, a_mm):
    """Torus volume, mm^3."""
    return 2 * np.pi**2 * R_mm * a_mm**2


# -------------------------- model functions -------------------------------- #
def model1_linear(t, z0, w0):
    """Model 1: no bubble effect, z = z0 + w0 * t."""
    return z0 + w0 * t


def model3_quadratic(t, z_launch, A):
    """Model 3: buoyancy only, z = z_launch + A * t^2 / 2  (t=0 at launch)."""
    return z_launch + 0.5 * A * t**2


def model2_lncosh(t, z_launch, w_inf, tau):
    """Model 2: buoyancy + drag, t=0 fixed at piston launch.

    z(t) = z_launch + w_inf * tau * ln cosh(t / tau)
    """
    return z_launch + w_inf * tau * np.log(np.cosh(t / tau))


def model2_velocity(t, w_inf, tau):
    """dz/dt = w_inf * tanh(t / tau)."""
    return w_inf * np.tanh(t / tau)


# -------------------------- IO and cleaning -------------------------------- #
def load_csv(path):
    """Load a CSV with three rows: t, x, z. Returns 3 ndarrays."""
    raw = pd.read_csv(path, header=None).to_numpy()
    if raw.shape[0] != 3:
        raise ValueError(f'Expected 3 rows (t, x, z); got shape {raw.shape}')
    return raw[0], raw[1], raw[2]


def clean_tracker_artefacts(t, x, z, z_jump_thresh_mm=20, dt_thresh_s=0.1,
                            early_t_cut_s=None, early_z_cut_mm=None):
    """Drop obvious tracker glitches.

    - Late-time isolated drops in z: a point is flagged if |z[i] - median(z
      in a window of width 1 s centred on t[i])| > z_jump_thresh_mm.
    - Optional early-time hard cut: drop points with t < early_t_cut_s
      AND z < early_z_cut_mm. Use this when the tracker is locked on the
      piston tip before the ring is detected.
    """
    keep = np.ones(len(t), dtype=bool)

    # late-time outlier detection via rolling median
    for i in range(len(t)):
        window = np.abs(t - t[i]) < 0.5  # +/- 0.5 s
        if window.sum() < 5:
            continue
        local_med = np.median(z[window])
        if abs(z[i] - local_med) > z_jump_thresh_mm:
            # only drop if it's an isolated point in a tight cluster
            close = window & (np.abs(t - t[i]) < dt_thresh_s)
            if close.sum() <= 3:
                keep[i] = False

    # early-time hard cut
    if early_t_cut_s is not None and early_z_cut_mm is not None:
        early_artefact = (t < early_t_cut_s) & (z < early_z_cut_mm)
        keep &= ~early_artefact

    return keep


# -------------------------- fitting --------------------------------------- #
def fit_model1(t, z):
    """Linear regression. Returns (params, std_errs, residuals, rms)."""
    p, cov = np.polyfit(t, z, 1, cov=True)
    z_fit = np.polyval(p, t)
    rms = np.sqrt(np.mean((z - z_fit)**2))
    # polyfit returns [slope, intercept]; convert to [z0, w0]
    params = np.array([p[1], p[0]])
    se = np.sqrt(np.diag(cov))[::-1]
    return params, se, z - z_fit, rms


def fit_model3(t, z):
    """Pure quadratic fit with t=0 enforced as launch time.

    z = z_launch + 0.5 * A * t^2
    """
    # t = t[t < early_t_cut]
    # z = z[:len(t)]
    p, cov = curve_fit(model3_quadratic, t, z, p0=[z.min(), 5.0])
    z_fit = model3_quadratic(t, *p)
    rms = np.sqrt(np.mean((z - z_fit)**2))
    return p, np.sqrt(np.diag(cov)), z - z_fit, rms


def fit_model2(t, z, bounds=None):
    """Model 2 with t_launch=0 fixed.

    Returns (params, std_errs, residuals, rms) where params = (z_launch,
    w_inf, tau). Initial guess uses the asymptotic slope.
    """
    if bounds is None:
        bounds = ([-30, 1.0, 0.05], [60, 30.0, 15.0])

    # use the late-time slope as w_inf guess
    late = t > 0.5 * t.max()
    w_inf_guess = max(1.0, np.polyfit(t[late], z[late], 1)[0])
    p0 = [z.min(), w_inf_guess, 1.0]

    p, cov = curve_fit(model2_lncosh, t, z, p0=p0, bounds=bounds)
    z_fit = model2_lncosh(t, *p)
    rms = np.sqrt(np.mean((z - z_fit)**2))
    return p, np.sqrt(np.diag(cov)), z - z_fit, rms


def f_test_nested(rss_reduced, rss_full, p_reduced, p_full, n):
    """One-tailed nested F-test. Returns (F, p_value)."""
    df1 = p_full - p_reduced
    df2 = n - p_full
    if df1 <= 0 or df2 <= 0 or rss_full <= 0:
        return np.nan, np.nan
    F = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    p = 1 - stats.f.cdf(F, df1, df2)
    return F, p


# -------------------------- physical readout ------------------------------ #
def physical_readout(w_inf, tau, V_b_uL_nominal, R_mm, a_mm,
                     C_d_eff=3.76, C_AM=0.69):
    """Map Model 3 (lncosh) parameters (w_inf, tau) to per-shot V_b given
    universal coefficients (C_d_eff, C_AM) from the joint fit.

    Inversion of eq 1.21. Two independent estimates of V_b are available:

      (i) From the buoyancy acceleration A = w_inf / tau,
              A = g V_b / ((1 + C_AM) V_r)
          -> V_b = A (1 + C_AM) V_r / g
          This uses C_AM only (not C_d_eff) and is the primary estimate.

      (ii) From the plateau velocity alone,
              w_inf^2 = g V_b / (C_d_eff V_r^{2/3})
          -> V_b = w_inf^2 C_d_eff V_r^{2/3} / g
          This uses C_d_eff only and is a redundant cross-check; the
          ratio V_b_fit_uL_cross / V_b_fit_uL tests internal consistency
          of the universal-coefficient assumption on this shot. A value
          near 1 indicates the shot belongs to the same physical regime
          as the shots used to estimate (C_d_eff, C_AM).

    The defaults C_d_eff = 3.76, C_AM = 0.69 are the joint-fit estimates
    from the four-shot U_p = 120 mm/s set. Override these kwargs when
    applying alternative coefficient choices.
    """
    V_b_nom = float(V_b_uL_nominal)
    V_r = V_ring(R_mm, a_mm)
    A = w_inf / tau

    V_b_fit_from_A = A * (1.0 + C_AM) * V_r / G
    V_b_fit_from_w = w_inf**2 * C_d_eff * V_r**(2.0 / 3.0) / G

    retention = V_b_fit_from_A / V_b_nom if V_b_nom > 0 else np.nan
    consistency = (V_b_fit_from_w / V_b_fit_from_A
                   if V_b_fit_from_A > 0 else np.nan)

    return {
        'A_mm_s2': A,
        'V_ring_mm3': V_r,
        'V_b_fit_uL': V_b_fit_from_A,
        'V_b_fit_uL_cross': V_b_fit_from_w,
        'V_b_fit_consistency': consistency,
        'retention': retention,
        'C_d_eff_used': C_d_eff,
        'C_AM_used': C_AM,
    }


# -------------------------- per-shot pipeline ----------------------------- #
def analyse_shot(path, V_b_uL, U_p, R_mm=23.0, a_mm=7.0, time_offset=True,
                 label=None, early_t_cut_s=0.6, early_z_cut_mm=6.0,
                 early_t_cut_quadratic=4,
                 C_d_eff=3.612, C_AM=0.11):
    """Full per-shot pipeline.

    Returns a dict containing cleaned data, fits, F-tests, and physical
    readouts. Intended to be passed in a list to `compare_shots`.

    The universal coefficients (C_d_eff, C_AM) are inputs from the joint
    fit (sec 1.3.5). V_b_uL is the *nominal* loading and is not used by
    the Model 2 fit itself; it is retained for comparison against the
    fit-derived V_b_fit_uL via the retention ratio.
    """
    t, x, z = load_csv(path)
    if time_offset:
        t = t - t[t > 0].min()

    keep = clean_tracker_artefacts(t, x, z, z_jump_thresh_mm=30,
                                   early_t_cut_s=early_t_cut_s,
                                   early_z_cut_mm=early_z_cut_mm)
    tc, xc, zc = t[keep], x[keep], z[keep]

    # all three fits
    p1, se1, r1, rms1 = fit_model1(tc, zc)
    p2, se2, r2, rms2 = fit_model2(tc, zc)

    # quadratic fit on the early window only (t < 1.5 * tau from M2)
    quad_cutoff = 1.5 * p2[2]
    p3, se3, r3, rms3 = fit_model3(tc[tc < quad_cutoff], zc[tc < quad_cutoff])

    n = len(tc)
    F_21, pv_21 = f_test_nested(np.sum(r1**2), np.sum(r2**2), 2, 3, n)
    F_23, pv_23 = f_test_nested(np.sum(r3**2), np.sum(r2**2), 2, 3, n)

    phys = physical_readout(p2[1], p2[2], V_b_uL, R_mm, a_mm,
                            C_d_eff=C_d_eff, C_AM=C_AM)

    # x(t) deceleration check (independent of vertical model)
    px = np.polyfit(tc, xc, 2)
    U_ring_start = 2 * px[0] * tc.min() + px[1]
    U_ring_end = 2 * px[0] * tc.max() + px[1]

    return {
        'label': label or path,
        'V_b_uL': V_b_uL,                    # nominal loading
        'U_p': U_p,
        'R_mm': R_mm,
        'a_mm': a_mm,
        # data
        't_raw': t, 'x_raw': x, 'z_raw': z,
        'keep_mask': keep,
        't': tc, 'x': xc, 'z': zc,
        # fits
        'model1': {'params': p1, 'se': se1, 'rms': rms1, 'resid': r1},
        'model3': {'params': p3, 'se': se3, 'rms': rms3, 'resid': r3},
        'model2': {'params': p2, 'se': se2, 'rms': rms2, 'resid': r2,
                   'z_launch': p2[0], 'w_inf': p2[1], 'tau': p2[2],
                   'z_launch_se': se2[0], 'w_inf_se': se2[1],
                   'tau_se': se2[2]},
        # model selection
        'F_M1_vs_M2': F_21, 'pval_M1_vs_M2': pv_21,
        'F_M3_vs_M2': F_23, 'pval_M3_vs_M2': pv_23,
        # physical readout (V_b derived from universal C_d, C_AM)
        'A_mm_s2': phys['A_mm_s2'],
        'V_ring_mm3': phys['V_ring_mm3'],
        'V_b_fit_uL': phys['V_b_fit_uL'],
        'V_b_fit_uL_cross': phys['V_b_fit_uL_cross'],
        'V_b_fit_consistency': phys['V_b_fit_consistency'],
        'retention': phys['retention'],
        'C_d_eff_used': phys['C_d_eff_used'],
        'C_AM_used': phys['C_AM_used'],
        # horizontal kinematics
        'U_ring_start': U_ring_start,
        'U_ring_end': U_ring_end,
        'x_decel_mm_s2': -2 * px[0],
    }



# -------------------------- visualisation --------------------------------- #
def plot_shot_diagnostics(res, savepath=None):
    """Four-panel diagnostic plot for a single shot."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    t, x, z = res['t'], res['x'], res['z']

    # A: z(t) with all three fits
    ax = axes[0, 0]
    excluded_mask = ~res['keep_mask']
    if excluded_mask.any():
        ax.scatter(res['t_raw'][excluded_mask], res['z_raw'][excluded_mask],
                   s=12, color='red', marker='x', label='excluded')
    ax.scatter(t, z, s=8, alpha=0.5, color='C0', label='data')
    tt = np.linspace(0, t.max() * 1.02, 500)
    ax.plot(tt, model1_linear(tt, *res['model1']['params']), 'C1-', lw=1.2,
            label=f"M1 lin: w0={res['model1']['params'][1]:.2f} "
                  f"(RMS={res['model1']['rms']:.2f})")
    ax.plot(tt, model3_quadratic(tt, *res['model3']['params']), 'C2--',
            lw=1.2,
            label=f"M2 quad: A={res['model3']['params'][1]:.2f} "
                  f"(RMS={res['model3']['rms']:.2f})")
    ax.plot(tt, model2_lncosh(tt, *res['model2']['params']), 'k-', lw=2,
            label=f"M3: w_inf={res['model2']['w_inf']:.2f}, "
                  f"tau={res['model2']['tau']:.2f} "
                  f"(RMS={res['model2']['rms']:.2f})")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (mm)')
    ax.set_ylim(0.8*z.min(), 1.2*z.max())

    ax.set_title(f"A. z(t)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # B: early-time zoom
    ax = axes[0, 1]
    early = t < max(4.0, 3 * res['model2']['tau'])
    ax.scatter(t[early]/res['model2']['tau'], z[early], s=22, alpha=0.6, color='C0')
    tte = np.linspace(0, max(4.0, 3 * res['model2']['tau']), 200)
    ax.plot(tte/res['model2']['tau'], model2_lncosh(tte, *res['model2']['params']), 'k-', lw=2,
            label='M3')
    ax.plot(tte/res['model2']['tau'], model3_quadratic(tte, *res['model3']['params']), 'C2--',
            lw=1.5, label='M2 (quadratic only)')
    ax.set_xlabel('t / tau')
    ax.set_ylabel('z (mm)')
    ax.set_xlim(0, 3)
    ax.set_ylim(bottom=0, top=1.2 * z[early].max())
    ax.set_title('B. Early time')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # C: vertical velocity
    ax = axes[1, 0]
    if len(t) > 31:
        w_data = savgol_filter(z, 31, 3, deriv=1, delta=t[1] - t[0])
        ax.scatter(t, w_data, s=8, alpha=0.4, color='C0', label='dz/dt')
    ax.plot(tt, model2_velocity(tt, res['model2']['w_inf'],
                                 res['model2']['tau']),
            'k-', lw=2, label=f"w_inf={res['model2']['w_inf']:.2f}")
    ax.axhline(res['model2']['w_inf'], color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('dz/dt (mm/s)')
    ax.set_title('C. Vertical velocity')
    ax.set_ylim(-5, max(25, 1.5 * res['model2']['w_inf']))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # D: residuals
    ax = axes[1, 1]
    ax.scatter(t, res['model2']['resid'], s=10, color='k', alpha=0.6,
               label='M3 residuals')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('residual (mm)')
    ax.set_title(f"D. model 2 residuals (RMS={res['model2']['rms']:.2f} mm)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches='tight')
    return fig


def compare_shots(results_list, savepath=None):
    """Cross-shot comparison: overlay z(t), check w_inf and A ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    colors = plt.cm.tab10.colors
    tt = np.linspace(0, max(r['t'].max() for r in results_list), 500)

    # A: z(t) overlay
    ax = axes[0, 0]
    for i, res in enumerate(results_list):
        c = colors[i % 10]
        ax.scatter(res['t'], res['z'], s=6, alpha=0.3, color=c)
        ax.plot(tt, model2_lncosh(tt, *res['model2']['params']),
                color=c, lw=1.5,
                label=f"{res['label']}: w_inf={res['model2']['w_inf']:.2f}, "
                      f"tau={res['model2']['tau']:.2f}")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (mm)')
    ax.set_title('A. All shots, M3 fits')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # B: universal-curve check
    ax = axes[0, 1]
    for i, res in enumerate(results_list):
        c = colors[i % 10]
        zl = res['model2']['z_launch']
        w = res['model2']['w_inf']
        tau = res['model2']['tau']
        u = res['t'] / tau
        Z = (res['z'] - zl) / (w * tau)
        ax.scatter(u, Z, s=8, alpha=0.4, color=c, label=res['label'])
    uu = np.linspace(0, 4, 500)
    ax.plot(uu, np.log(np.cosh(uu)), 'k--', lw=1.5, label='ln cosh(u)')
    ax.set_xlabel('u = t / tau')
    ax.set_ylabel('Z = (z - z_launch) / (w_inf * tau)')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.2 * (res['z'].max()))
    ax.set_title('B. Universal-curve collapse')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # C: w_inf vs V_b
    ax = axes[1, 0]
    Vbs = np.array([r['V_b_uL'] for r in results_list])
    ws = np.array([r['model2']['w_inf'] for r in results_list])
    w_se = np.array([r['model2']['w_inf_se'] for r in results_list])
    Ups = np.array([r['U_p'] for r in results_list])
    for Up in np.unique(Ups):
        m = Ups == Up
        ax.errorbar(Vbs[m], ws[m], yerr=w_se[m], marker='o', ls='',
                    label=f'U_p={Up}', capsize=3)
    # reference sqrt(V_b) line anchored at the median
    if len(Vbs) >= 2:
        Vref = np.median(Vbs)
        wref = np.median(ws)
        Vgrid = np.linspace(Vbs.min() * 0.8, Vbs.max() * 1.2, 100)
        ax.plot(Vgrid, wref * np.sqrt(Vgrid / Vref), 'k--', alpha=0.5,
                label='~ sqrt(V_b)')
    ax.set_xlabel('V_b (uL)')
    ax.set_ylabel('w_inf (mm/s)')
    ax.set_title('C. w_inf vs V_b (M3 predicts sqrt(V_b))')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # D: A vs V_b
    ax = axes[1, 1]
    As = np.array([r['A_mm_s2'] for r in results_list])
    for Up in np.unique(Ups):
        m = Ups == Up
        ax.plot(Vbs[m], As[m], 'o', label=f'U_p={Up}')
    if len(Vbs) >= 2:
        Vref = np.median(Vbs)
        Aref = np.median(As)
        Vgrid = np.linspace(Vbs.min() * 0.8, Vbs.max() * 1.2, 100)
        ax.plot(Vgrid, Aref * (Vgrid / Vref), 'k--', alpha=0.5,
                label='~ V_b (linear)')
    ax.set_xlabel('V_b (uL)')
    ax.set_ylabel('A = w_inf/tau (mm/s^2)')
    ax.set_title('D. A vs V_b (M3 predicts linear in V_b)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches='tight')
    return fig


def summary_table(results_list):
    """Build a pandas DataFrame summarising all shots."""
    rows = []
    for r in results_list:
        rows.append({
            'label': r['label'],
            'U_p': r['U_p'],
            'V_b_nom_uL': r['V_b_uL'],
            'V_b_fit_uL': r['V_b_fit_uL'],
            'V_b_fit_uL_cross': r['V_b_fit_uL_cross'],
            'retention': r['retention'],
            'V_b_consistency': r['V_b_fit_consistency'],
            'z_launch_mm': r['model2']['z_launch'],
            'w_inf_mm_s': r['model2']['w_inf'],
            'w_inf_se': r['model2']['w_inf_se'],
            'tau_s': r['model2']['tau'],
            'tau_se': r['model2']['tau_se'],
            'A_mm_s2': r['A_mm_s2'],
            'C_d_eff_used': r['C_d_eff_used'],
            'C_AM_used': r['C_AM_used'],
            'rms_M2': r['model2']['rms'],
            'rms_M1': r['model1']['rms'],
            'rms_M3': r['model3']['rms'],
            'pval_M1_vs_M2': r['pval_M1_vs_M2'],
            'pval_M3_vs_M2': r['pval_M3_vs_M2'],
            'U_ring_start': r['U_ring_start'],
            'U_ring_end': r['U_ring_end'],
        })
    return pd.DataFrame(rows)


# -------------------------- example usage --------------------------------- #
if __name__ == '__main__':
    # example: replace with your actual file paths and metadata
    shot_specs = [
        # (path, V_b in uL, U_p in mm/s, label)
        # ('shots/U120_V40_rep1.csv', 40, 120, 'U120_V40_rep1'),
        # ('shots/U120_V20_rep1.csv', 20, 120, 'U120_V20_rep1'),
        # ('shots/U160_V40_rep1.csv', 40, 160, 'U160_V40_rep1'),
        # ...
    ]

    results = []
    for path, V_b, U_p, label in shot_specs:
        res = analyse_shot(path, V_b_uL=V_b, U_p=U_p, label=label)
        results.append(res)
        plot_shot_diagnostics(res, savepath=f'{label}_diagnostics.png')
        plt.close()

    if results:
        compare_shots(results, savepath='cross_shot_comparison.png')
        df = summary_table(results)
        df.to_csv('shot_summary.csv', index=False)
        print(df.to_string())
