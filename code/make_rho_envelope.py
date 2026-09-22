# -*- coding: utf-8 -*-
"""
Illustrations for the "two prototypes express K classes" theory (two figures).

The reduction at the heart of the theory: the decision problem depends on the
sample only through the scalar distance ratio rho = ||z-p2|| / ||z-p1||, and the
score of each class is an affine function of rho, f_k(rho) = y_1k*rho + y_2k.

  fig1 (upper_envelope):   the upper envelope of the affine line family is cut
                           into K segments -> the expressiveness theorem
  fig2 (rho_distribution): the empirical distribution of rho narrows and
                           concentrates toward 1 as d grows -> concentration
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------- Theoretical construction ----------------

def symmetric_soft_labels(K):
    """Symmetric soft-label construction (appendix, proof of the K-class theorem):
    y_1k = sum_{l=k}^{K-1} l / sum_{l=1}^{K-1} l^2,   y_2k = y_1,K+1-k."""
    S = sum(l * l for l in range(1, K))
    y1 = np.array([sum(range(k, K)) / S for k in range(1, K + 1)])
    y2 = y1[::-1]
    return y1, y2

def decision_boundaries(K):
    """Decision boundaries between adjacent classes, t_k = (K-k)/k, k = 1..K-1."""
    return np.array([(K - k) / k for k in range(1, K)])

def sample_rho(d, n=200000, seed=0):
    """z ~ Uniform([-3,3]^d), p1 = -e1, p2 = e1; returns rho = ||z-p2||/||z-p1||."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-3.0, 3.0, size=(n, d))
    z1 = z[:, 0]
    R2 = np.sum(z[:, 1:] ** 2, axis=1)   # squared sum of the d-1 coords orthogonal to the axis
    rho = np.sqrt(((z1 - 1.0) ** 2 + R2) / ((z1 + 1.0) ** 2 + R2))
    return rho

# ---------------- Parameters ----------------
K = 10
y1, y2 = symmetric_soft_labels(K)
bounds = decision_boundaries(K)            # [9, 4, 7/3, 1.5, 1, 2/3, 3/7, 1/4, 1/9]
sorted_bounds = np.sort(bounds)

RHO_LO, RHO_HI = 7e-2, 1.4e1               # rho display range (log axis), wide at both ends
rho_grid = np.logspace(np.log10(RHO_LO), np.log10(RHO_HI), 4000)

# Okabe-Ito colour-blind-safe palette (8 colours) plus 2 supplements, 10 in total
class_colors = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
    '#56B4E9', '#F0E442', '#000000', '#7A7A7A', '#8B4513',
]
dims = [3, 5, 10, 20, 50]
dcolors = plt.cm.Blues(np.linspace(0.45, 0.95, len(dims)))   # single-hue ramp, light to dark

plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
})

base = os.path.dirname(os.path.abspath(__file__))
figdir = os.path.join(base, "figures")
os.makedirs(figdir, exist_ok=True)


# ================= Figure 1: the affine upper envelope =================
fig1, ax1 = plt.subplots(figsize=(14.0, 7.0))

edges = np.concatenate([[RHO_LO], sorted_bounds, [RHO_HI]])
for i in range(K):                                   # shaded background = winning interval
    k_win = K - 1 - i
    ax1.axvspan(edges[i], edges[i + 1], color=class_colors[k_win], alpha=0.13, zorder=0)

for k in range(K):                                   # the K affine lines (pale grey)
    ax1.plot(rho_grid, y1[k] * rho_grid + y2[k], color='0.70', lw=1.1, alpha=0.85, zorder=1)

F = np.vstack([y1[k] * rho_grid + y2[k] for k in range(K)])
ax1.plot(rho_grid, F.max(axis=0), color='black', lw=2.6, zorder=3)   # upper envelope

for t in sorted_bounds:                              # decision boundaries (dashed)
    ax1.axvline(t, color='0.20', ls='--', lw=1.0, alpha=0.55, zorder=2)
ax1.axvline(1.0, color='red', lw=1.6, alpha=0.85, zorder=2)          # rho = 1

for i in range(K):                                   # label the winning class of each segment
    lo, hi = edges[i], edges[i + 1]
    rho_mid = np.sqrt(lo * hi)
    k_win = K - 1 - i
    f_mid = y1[k_win] * rho_mid + y2[k_win]
    ax1.text(rho_mid, f_mid + 0.05, f'$C_{{{k_win+1}}}$', ha='center', va='bottom',
             fontsize=26, color=class_colors[k_win], fontweight='bold', zorder=4)

ax1.set_xscale('log')
ax1.set_xlim(RHO_LO, RHO_HI)
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Distance ratio  $\\rho$', fontsize=30)
ax1.set_ylabel('Affine class score', fontsize=30)
ax1.set_title('Affine upper envelope for ten classes', fontsize=30, pad=12)

fig1.tight_layout()
fig1.savefig(os.path.join(figdir, 'upper_envelope.png'), dpi=600, bbox_inches='tight')
fig1.savefig(os.path.join(figdir, 'upper_envelope.pdf'), bbox_inches='tight')
plt.close(fig1)


# ================= Figure 2: rho distribution vs dimension (KDE-smoothed) =================
fig2, ax2 = plt.subplots(figsize=(14.0, 7.0))

x_log = np.linspace(np.log10(RHO_LO), np.log10(RHO_HI), 800)
for d, c in zip(dims, dcolors):
    rho = sample_rho(d)
    logrho = np.log10(rho)
    # Subsample for the KDE (smoother and faster); estimate the density in log space.
    idx = np.random.default_rng(d).choice(len(logrho), size=min(30000, len(logrho)),
                                         replace=False)
    kde = gaussian_kde(logrho[idx])
    dens = kde(x_log)
    dens = np.clip(dens, dens.max() * 1e-4, None)     # flatten the tails; keeps the log axis connected
    ax2.plot(10 ** x_log, dens, color=c, lw=2.2, label=f'$d={d}$')

for t in sorted_bounds:
    ax2.axvline(t, color='0.20', ls='--', lw=1.0, alpha=0.40, zorder=1)
ax2.axvline(1.0, color='red', lw=1.6, alpha=0.85, zorder=2)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(RHO_LO, RHO_HI)
ax2.set_xlabel('Distance ratio  $\\rho$', fontsize=30)
ax2.set_ylabel('Density', fontsize=30)
ax2.legend(loc='upper right', fontsize=26, framealpha=0.9)
ax2.set_title('Distance-ratio concentration', fontsize=30, pad=12)

fig2.tight_layout()
fig2.savefig(os.path.join(figdir, 'rho_distribution.png'), dpi=200, bbox_inches='tight')
fig2.savefig(os.path.join(figdir, 'rho_distribution.pdf'), bbox_inches='tight')
plt.close(fig2)

print('saved fig1:', os.path.join(figdir, 'upper_envelope.png'))
print('saved fig2:', os.path.join(figdir, 'rho_distribution.png'))
