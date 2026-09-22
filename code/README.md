# Prototype Expressiveness

Numerical verification of two-prototype soft-label expressiveness, and of the
distance-ratio concentration that limits its high-dimensional visibility.

Simulation code for the Experiments section of *Less-Than-One-Shot Learning in
High-Dimensional Spaces*. Two self-contained scripts, one per experiment:

| Script | Experiment | Figure |
|---|---|---|
| [`make_rho_envelope.py`](make_rho_envelope.py) | Illustrating the expressiveness of two prototypes | `upper_envelope.png` |
| [`k_extension_experiment.py`](k_extension_experiment.py) | Dimension-dependent uniform coverage | `k_extension_dimension_heatmap_2.png` |

## Scope: what these experiments do and do not show

> **They verify a representational claim, not clustering accuracy.** Neither
> script evaluates clustering on a real dataset, and neither should be cited as
> evidence that the construction is useful in practice.

The theory reduces the decision rule to a single scalar. Multiplying the score
`g_k(z) = y_1k/||z−p_1|| + y_2k/||z−p_2||` by the positive quantity
`||z−p_1||` leaves the argmax unchanged, so

```
argmax_k g_k(z)  =  argmax_k ( y_1k·ρ + y_2k ),    ρ = ||z−p_2|| / ||z−p_1||.
```

Each class becomes an affine function of the one scalar `ρ`, and the decision
problem is an upper-envelope problem in one dimension regardless of the ambient
dimension `d`.

Two consequences follow, and **they must be read together**:

1. **Existence is dimension-free.** Two prototypes with probability-valued soft
   labels can give each of `K` classes a non-empty decision region, for any
   finite `K`.
2. **Visibility is not.** Under `z ~ Uniform([−3,3]^d)` the ratio concentrates:
   `E|ρ−1| = Θ(1/d)`. The regions keep positive measure, but their *uniform*
   measure collapses toward the class owning the interval containing `ρ = 1`.

The construction therefore does **not** become more separable in higher
dimensions — the theory positively predicts the opposite, and the dimension
sweep below confirms it. Do not present these results as evidence of
high-dimensional separability.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires Python 3.11+. Developed and tested on CPython 3.13.9 with NumPy 2.3.5,
SciPy 1.16.3 and Matplotlib 3.10.6.

## Running

### 1. Affine upper envelope

```bash
python make_rho_envelope.py
```

Writes `figures/upper_envelope.{png,pdf}` and `figures/rho_distribution.{png,pdf}`.

The first figure is the one used in the paper. It shows ten affine class scores
whose slopes and intercepts are set by the soft labels, with the strictly ordered
intersections of adjacent pairs defining the decision boundaries and the upper
envelope (black) partitioned into ten consecutive winning intervals. Every class
dominates on some non-empty interval of `ρ`, so the number of representable
classes is bounded by the number of ordered intersections the soft labels can
encode, not by the number of prototypes.

> **Note on the second figure.** `rho_distribution.png` is not used in the paper.
> Its KDE is known to show two artefacts at high `d`: the peak at `ρ = 1` splits
> into a notch, and the tails flatten into plateaus. The cause is that `ρ`
> concentrates so tightly at `d = 50` that the KDE bandwidth is comparable to the
> evaluation grid step, so the curve is under-resolved; the plateaus come from
> the `np.clip` floor on the density. The qualitative conclusion — the
> distribution narrows as `d` grows — is unaffected, but do not read the shape of
> the peak or the level of the tails off this figure.

### 2. Dimension sweep

```bash
python k_extension_experiment.py
```

Sweeps `d ∈ {3,5,10,20,50}` × `K ∈ {2,3,4,5,6,8,10,20}` — 40 configurations,
400,000 uniform samples each — and writes `results/k_extension_results.csv` plus
four figures. `d = 1, 2` are deliberately never scanned: the concentration bound
requires the singularity at `p_1` to be integrable, which holds only for `d > 2`.

Shorter runs (use `--output-dir` so the full run's artefacts are not overwritten):

```bash
python k_extension_experiment.py --dimensions 3 --k-values 3,10 --samples 50000 --output-dir results/check
```

## Results

At each configuration, every sample is classified twice — once through the
distance form `g_k(z)` and once through the affine form `y_1k·ρ + y_2k`. The two
agree on **every** sampled point in all 40 configurations, which is a regression
check on the algebraic reduction rather than an empirical finding.

Effective classes `R` recovered by uniform sampling, as `R/K`:

| `d` | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 | K=10 | K=20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2/2 | 3/3 | 4/4 | 5/5 | 6/6 | 8/8 | 10/10 | 20/20 |
| 5 | 2/2 | 3/3 | 4/4 | 5/5 | 6/6 | 8/8 | 9/10 | 10/20 |
| 10 | 2/2 | 2/3 | 2/4 | 1/5 | 1/6 | 1/8 | 1/10 | 2/20 |
| 20 | 1/2 | 1/3 | 1/4 | 1/5 | 1/6 | 1/8 | 1/10 | 1/20 |
| 50 | 1/2 | 1/3 | 1/4 | 1/5 | 1/6 | 1/8 | 1/10 | 1/20 |

Targeted sampling — drawing points from inside every theoretically non-empty
interval of `ρ` — recovers all `K` classes in all 40 configurations, including
`d = 50`.

Concentration of the distance ratio (independent of `K`):

| `d` | `E|ρ−1|` | `d·E|ρ−1|` |
|---:|---:|---:|
| 3 | 3.885e-1 | 1.166 |
| 5 | 2.100e-1 | 1.050 |
| 10 | 1.009e-1 | 1.009 |
| 20 | 5.010e-2 | 1.002 |
| 50 | 2.002e-2 | 1.001 |

`d·E|ρ−1|` converges monotonically toward 1, matching both the rate and the
constant of the concentration theorem.

**Existence is dimension-free; measure is not.** Targeted sampling never loses a
class; uniform sampling goes from a perfect score at `d = 3` to recovering a
single class at `d ≥ 20`.

## Caveats

- **Targeted sampling is a measure-zero existence check.** Every point it
  generates lies on the axis joining the prototypes, a one-dimensional subset of
  `R^d` of `d`-dimensional Lebesgue measure zero. It can show a region is
  non-empty; it says nothing about the region's volume, and its coverage cannot
  be added to or used to corroborate the uniform-sampling coverage.
- **The construction depends on a schedule.** The soft labels use the ordered
  intersection schedule `t_i = 0.25 · 1.10^(i−1)`. A steeper schedule (1.8) drives
  the smallest region below the Monte Carlo resolution at `K ≥ 8` and uniform
  sampling then misses classes. Any strictly increasing positive schedule
  realises all `K` classes; the schedule only controls how much probability mass
  they receive.
- **The two-prototype reduction does not extend to more prototypes.**
  Multi-prototype methods have no single scalar `ρ`, so nothing here predicts
  their behaviour.
- Each dimension reuses the same seed, so the `d = 3` row is bit-reproducible and
  matches the numbers reported in the paper.

## Files

```
make_rho_envelope.py       affine upper envelope + rho distribution figures
k_extension_experiment.py  dimension sweep, CSV and four figures
requirements.txt           runtime dependencies
```

Generated output (`figures/`, `results/`) is gitignored — run the scripts to
produce it.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use this code.
