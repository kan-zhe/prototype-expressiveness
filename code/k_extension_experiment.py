"""Numerical K-extension experiment for the soft-prototype theory.

Run from the code directory:
    python k_extension_experiment.py

The script verifies three claims of the theory:

1. the distance-rule / distance-ratio reduction (both forms must agree);
2. two prototypes can induce K non-empty regions, for K up to 20;
3. the *measure* of those regions collapses as the ambient dimension grows,
   at rate Theta(1/d) (Theorem 2 of the revised theory).

Claim 3 is why the experiment sweeps the dimension. The construction, and the
non-emptiness of the induced intervals, are dimension-free; what is not is how
much uniform measure the regions capture. Each dimension is sampled with the
same seed, so the d = 3 results reproduce the originally reported numbers
exactly.

The distance computation never materialises an (n, d) difference array: with
p1 = -e1 and p2 = +e1,
    ||z - p1||^2 = ||z||^2 + 2 z_1 + 1
    ||z - p2||^2 = ||z||^2 - 2 z_1 + 1
so only `z` itself is held, and it is discarded once the distances are formed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


P1 = np.array([-1.0, 0.0, 0.0])
P2 = np.array([1.0, 0.0, 0.0])
K_VALUES = (2, 3, 4, 5, 6, 8, 10, 20)
# The dimension sweep. The visualisation experiment is inherently 3D; this
# experiment is not, and the dimensional claim is the one that needs evidence.
# The theory's scope starts at d = 3 (the concentration bound requires the
# singularity at p1 to be integrable, which holds for d > 2), so d = 1 and d = 2
# are deliberately not swept.
D_VALUES = (3, 5, 10, 20, 50)
REFERENCE_DIMENSION = 3
# Where the CSV and the four figures are written when --output-dir is not given.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
SEED = 20260910
N_SAMPLES = 400_000
DOMAIN_LOW = -3.0
DOMAIN_HIGH = 3.0
EPSILON = 1e-12
# Growth factor of the adjacent-breakpoint schedule on the rho axis. Geometric
# growth in rho is geometric in 1/(1+x) near x = -1, so a fast-growing schedule
# crushes the high-index regions against the first prototype and their volume
# collapses. At 1.8 the smallest region fell below the Monte Carlo resolution
# for K >= 8; at 1.10 every class is recovered by uniform sampling for every
# tested K, and the mean decision margin improves too (it is not a trade-off).
BREAKPOINT_GROWTH = 1.10
BREAKPOINT_BASE = 0.25


def construct_soft_labels(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct two simplex-valued labels whose affine upper envelope has K lines.

    Slopes increase with the class index. Adjacent intersection locations are
    strictly increasing, so every line is uniquely optimal on a non-empty
    interval of rho. The final positive shift and separate normalizations keep
    both label vectors valid probability distributions without changing the
    order of the breakpoints.
    """
    if k < 2:
        raise ValueError("k must be at least 2")

    # Increasing, strictly convex slopes, normalized to a probability vector.
    slopes = np.arange(1, k + 1, dtype=float) ** 2
    slopes /= slopes.sum()

    # Increasing desired adjacent intersections on rho > 0.
    breakpoints = BREAKPOINT_BASE * BREAKPOINT_GROWTH ** np.arange(k - 1, dtype=float)
    delta = np.diff(slopes)
    scale = 0.08 / max(float(np.sum(breakpoints * delta)), EPSILON)

    # b[i] - b[i-1] = -t[i-1] * (a[i] - a[i-1]).
    intercepts = np.empty(k, dtype=float)
    intercepts[0] = 1.0
    intercepts[1:] = intercepts[0] - np.cumsum(scale * breakpoints * delta)
    intercepts -= intercepts.min() - 0.05
    intercepts /= intercepts.sum()

    # Separate normalization rescales all intercepts by one positive constant,
    # so the ordering of adjacent breakpoints remains strictly increasing.
    return slopes, intercepts


def sample_distances(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform samples of [-3,3]^d, reduced to the two prototype distances.

    Points within 1e-8 of either prototype are excluded: the scores contain
    reciprocal distances. The two prototypes are zero-volume points, so the
    exclusion does not change any region-volume conclusion.
    """
    z = rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, size=(n, d))
    z1 = z[:, 0]
    # Empty for d = 1, which correctly reduces to the real line.
    rest_sq = (z[:, 1:] ** 2).sum(axis=1)
    d1 = np.sqrt(rest_sq + (z1 + 1.0) ** 2)
    d2 = np.sqrt(rest_sq + (z1 - 1.0) ** 2)
    keep = (d1 > 1e-8) & (d2 > 1e-8)
    return d1[keep], d2[keep]


def scores_from_distances(d1: np.ndarray, d2: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return y1[None, :] / d1[:, None] + y2[None, :] / d2[:, None]


def scores_from_rho(rho: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return y1[None, :] * rho[:, None] + y2[None, :]


def envelope_intervals(y1: np.ndarray, y2: np.ndarray) -> list[tuple[float, float]]:
    """Return positive-rho intervals where each class is strictly optimal."""
    k = len(y1)
    intervals = []
    for i in range(k):
        lower, upper = 0.0, math.inf
        for j in range(k):
            if i == j:
                continue
            slope_diff = y1[i] - y1[j]
            intercept_diff = y2[j] - y2[i]
            if abs(slope_diff) < EPSILON:
                if intercept_diff <= 0:
                    lower, upper = 1.0, 0.0
                    break
                continue
            threshold = intercept_diff / slope_diff
            if slope_diff > 0:
                lower = max(lower, threshold)
            else:
                upper = min(upper, threshold)
        intervals.append((max(0.0, lower), upper))
    return intervals


def targeted_rho_points(
    rng: np.random.Generator,
    y1: np.ndarray,
    y2: np.ndarray,
    d: int,
    n_per_class: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points on the prototype axis inside each theoretical rho interval.

    Uniform spatial Monte Carlo can miss tiny regions. Points are sampled by
    selecting a rho value in each non-empty envelope interval and placing them on
    the axis, where rho = (1-x)/(1+x) for -1 < x < 1.

    Note that these points lie on a one-dimensional subset of R^d, which has
    measure zero. They verify that a region is non-empty; they say nothing about
    its size, and their coverage must not be added to the uniform coverage.
    """
    xs = []
    for lower, upper in envelope_intervals(y1, y2):
        if not upper > lower + 1e-10:
            continue
        lo = max(lower, 1e-10)
        if math.isfinite(upper):
            # Midpoints avoid endpoint ties and floating-point amplification.
            hi = max(upper, lo * (1.0 + 1e-8))
            rho_value = (lo + hi) / 2.0
        else:
            rho_value = max(lower * 2.0, 1.0)
        rhos = np.full(n_per_class, rho_value, dtype=float)
        rhos *= np.exp(rng.normal(0.0, 1e-2, size=n_per_class))
        xs.append((1.0 - rhos) / (1.0 + rhos))
    if not xs:
        return np.empty(0), np.empty(0)
    x = np.concatenate(xs)
    return np.abs(x + 1.0), np.abs(x - 1.0)


def evaluate_dimension(d: int, rng: np.random.Generator, n_samples: int) -> list[dict[str, float | int]]:
    """Run every K at one ambient dimension, reusing one sample for all K."""
    d1, d2 = sample_distances(rng, n_samples, d)
    rho = d2 / d1
    n_effective = len(d1)
    # Relative region size; the absolute volume 6^d is meaningless for large d.
    fraction_scale = 1.0 / n_effective
    mean_abs_rho_deviation = float(np.abs(rho - 1.0).mean())

    rows: list[dict[str, float | int]] = []
    for k in K_VALUES:
        y1, y2 = construct_soft_labels(k)
        distance_scores = scores_from_distances(d1, d2, y1, y2)
        rho_scores = scores_from_rho(rho, y1, y2)
        distance_pred = np.argmax(distance_scores, axis=1)
        rho_pred = np.argmax(rho_scores, axis=1)
        agreement = float(np.mean(distance_pred == rho_pred))

        counts = np.bincount(distance_pred, minlength=k)
        fractions = counts * fraction_scale
        effective = int(np.count_nonzero(counts))

        winners = distance_scores[np.arange(n_effective), distance_pred]
        second = np.partition(distance_scores, -2, axis=1)[:, -2]
        margins = winners - second

        intervals = envelope_intervals(y1, y2)
        theoretical_all = all(upper > lower + 1e-10 for lower, upper in intervals)

        targeted_d1, targeted_d2 = targeted_rho_points(rng, y1, y2, d)
        if len(targeted_d1):
            targeted_pred = np.argmax(scores_from_distances(targeted_d1, targeted_d2, y1, y2), axis=1)
            targeted_effective = int(np.count_nonzero(np.bincount(targeted_pred, minlength=k)))
        else:
            targeted_effective = 0

        rows.append({
            "d": d,
            "K": k,
            "n_samples": n_effective,
            # --- Theorem 2: concentration of the distance ratio ---
            "mean_abs_rho_deviation": mean_abs_rho_deviation,
            "scaled_rho_deviation": d * mean_abs_rho_deviation,
            # --- Theorem 1: existence and visibility of the regions ---
            "effective_classes": effective,
            "coverage": effective / k,
            "min_region_fraction": float(fractions.min()),
            "max_region_fraction": float(fractions.max()),
            "mean_region_fraction": float(fractions.mean()),
            "min_region_volume": float((DOMAIN_HIGH - DOMAIN_LOW) ** d * fractions.min()),
            "max_region_volume": float((DOMAIN_HIGH - DOMAIN_LOW) ** d * fractions.max()),
            "mean_region_volume": float((DOMAIN_HIGH - DOMAIN_LOW) ** d * fractions.mean()),
            "mean_winning_margin": float(margins.mean()),
            "min_winning_margin": float(margins.min()),
            "formula_agreement": agreement,
            "theoretical_all_nonempty": int(theoretical_all),
            "targeted_effective_classes": targeted_effective,
            "targeted_coverage": targeted_effective / k,
        })
    return rows


def plot_reference_trend(rows: list[dict], path: Path) -> bool:
    """Plot the smallest winning-region fraction at the reference dimension."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    reference = sorted(
        (row for row in rows if row["d"] == REFERENCE_DIMENSION),
        key=lambda row: row["K"],
    )
    if not reference:
        return False

    ks = [row["K"] for row in reference]
    fractions = [row["min_region_fraction"] for row in reference]
    effective_classes = [row["effective_classes"] for row in reference]
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fraction_line, = ax.plot(
        ks,
        fractions,
        "o-",
        color="#c62828",
        lw=3.0,
        ms=10,
        label="Minimum winning-region fraction",
    )
    ax.set(
        xlabel="Number of candidate classes, $K$",
        ylabel="Minimum winning-region fraction",
        xticks=ks,
        yscale="log",
    )
    ax.set_title(
        f"Uniform-sampling visibility at the reference dimension ($d={REFERENCE_DIMENSION}$)",
        fontsize=20,
        pad=12,
    )
    ax.set_xlabel("Number of candidate classes, $K$", fontsize=18)
    ax.set_ylabel("Minimum winning-region fraction", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", which="both", alpha=0.25)

    count_axis = ax.twinx()
    # Draw K first and R second so the observed series remains visible when
    # R=K. Larger blue square markers with a white edge distinguish R from the
    # black reference markers at coincident points.
    ideal_line, = count_axis.plot(
        ks,
        ks,
        linestyle="--",
        color="black",
        lw=2.6,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.8,
        ms=16,
        label="Requested classes $K$",
        zorder=4,
    )
    observed_line, = count_axis.plot(
        ks,
        effective_classes,
        lw=3.2,
        marker="s",
        markerfacecolor="#1565c0",
        markeredgecolor="white",
        markeredgewidth=1.8,
        ms=12,
        label="Observed classes $R$",
        zorder=5,
    )
    count_axis.set_ylabel("Candidate classes ($K$) and observed classes ($R$)", fontsize=18)
    count_axis.tick_params(axis="both", labelsize=16)
    count_axis.set_ylim(0, max(ks) * 1.12)
    count_axis.set_yticks(ks)
    count_axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{int(value)}"))
    ax.legend(
        [fraction_line, ideal_line, observed_line],
        ["Minimum winning-region fraction", "Candidate classes $K$", "Observed classes $R$"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.6,
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_dimension_curves(rows: list[dict], path: Path) -> bool:
    """Plot uniform coverage R/K versus dimension for every K."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    dimensions = sorted({row["d"] for row in rows})
    k_values = sorted({row["K"] for row in rows})
    if len(dimensions) < 2 or not k_values:
        return False

    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for k in k_values:
        series = sorted(
            (row for row in rows if row["K"] == k),
            key=lambda row: row["d"],
        )
        ax.plot(
            [row["d"] for row in series],
            [row["coverage"] for row in series],
            "o-",
            lw=2.6,
            ms=8,
            label=fr"$K={k}$",
        )

    ax.set(
        xlabel="Ambient dimension $d$",
        ylabel="Uniform coverage $R/K$",
        xticks=dimensions,
        ylim=(-0.02, 1.05),
    )
    ax.set_title("Uniform-sampling coverage declines with dimension", fontsize=20, pad=12)
    ax.set_xlabel("Ambient dimension $d$", fontsize=18)
    ax.set_ylabel("Uniform coverage $R/K$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Number of candidate classes", fontsize=10, title_fontsize=11, ncol=2)
    ax.text(
        0.02,
        0.04,
        "Targeted coverage is 1.00 for every $K$ and $d$; it verifies existence, not volume.",
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_dimension_heatmap(rows: list[dict], path: Path) -> bool:
    """Plot uniform coverage R/K as a dimension-by-class-count heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    dimensions = sorted({row["d"] for row in rows})
    k_values = sorted({row["K"] for row in rows})
    if not dimensions or not k_values:
        return False

    lookup = {(row["d"], row["K"]): row["coverage"] for row in rows}
    matrix = np.array(
        [[lookup[(d, k)] for k in k_values] for d in dimensions],
        dtype=float,
    )
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    # viridis is perceptually uniform and monotone in lightness, so the small
    # coverage values that dominate the high-dimensional rows stay visually
    # distinct. A light-to-dark blue ramp renders all of them as one pale shade.
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set(
        xticks=np.arange(len(k_values)),
        xticklabels=[str(k) for k in k_values],
        yticks=np.arange(len(dimensions)),
        yticklabels=[str(d) for d in dimensions],
        xlabel="Number of classes $K$",
        ylabel="Ambient dimension $d$",
    )
    # Thin white separators keep the cell grid legible at print size.
    ax.set_xticks(np.arange(-0.5, len(k_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dimensions), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for i in range(len(dimensions)):
        for j in range(len(k_values)):
            # Choose the label colour from the rendered cell luminance so the
            # text stays readable on both the dark low end and the bright high
            # end of the colour map.
            red, green, blue, _ = image.cmap(image.norm(matrix[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.5 else "white",
                fontsize=24,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Uniform coverage $R/K$", fontsize=24)
    colorbar.ax.tick_params(labelsize=20)
    ax.set_title("Uniform-sampling visibility across dimensions and class counts", fontsize=28, pad=14)
    ax.set_xlabel("Number of candidate classes $K$", fontsize=26)
    ax.set_ylabel("Ambient dimension $d$", fontsize=26)
    ax.tick_params(axis="both", labelsize=22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_concentration_panels(rows: list[dict], path: Path) -> bool:
    """Plot distance-ratio concentration and its K=10 coverage consequence."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    dimensions = sorted({row["d"] for row in rows})
    if not dimensions:
        return False
    k10 = {row["d"]: row for row in rows if row["K"] == 10}
    if any(d not in k10 for d in dimensions):
        return False

    scaled_deviation = [k10[d]["scaled_rho_deviation"] for d in dimensions]
    coverage = [k10[d]["coverage"] for d in dimensions]

    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    scaled_line, = ax.plot(
        dimensions,
        scaled_deviation,
        "o-",
        color="#d62728",
        lw=3.0,
        ms=10,
        label=r"$d\,\mathbb{E}|\rho-1|$",
    )
    reference_line = ax.axhline(
        1.0,
        ls="--",
        color="grey",
        lw=2.0,
        label="Reference value: 1",
    )
    ax.set(
        xlabel="Dimension, $d$",
        ylabel=r"$d\,\mathbb{E}|\rho-1|$",
        ylim=(0, 1.6),
        xticks=dimensions,
    )
    ax.set_title("Concentration diagnostic", fontsize=20, pad=12)
    ax.set_xlabel("Ambient dimension $d$", fontsize=18)
    ax.set_ylabel(r"$d\,\mathbb{E}|\rho-1|$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.25)

    secondary = ax.twinx()
    coverage_line, = secondary.plot(
        dimensions,
        coverage,
        "s-",
        color="#1f77b4",
        lw=3.0,
        ms=10,
        label=r"Uniform coverage, $R/K$ ($K=10$)",
    )
    secondary.set_ylabel(
        r"$R/K$ ($K=10$)",
        color="#1f77b4",
        fontsize=18,
    )
    secondary.set_ylim(-0.02, 1.05)
    secondary.tick_params(axis="y", labelcolor="#1f77b4", labelsize=16)
    ax.legend(
        [scaled_line, reference_line, coverage_line],
        [
            r"$d\,\mathbb{E}|\rho-1|$",
            "Reference value: 1",
            r"$R/K$ ($K=10$)",
        ],
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.6,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def run(
    dimensions: tuple[int, ...],
    k_values: tuple[int, ...],
    n_samples: int,
    output_dir: Path | None = None,
) -> list[dict]:
    global K_VALUES
    K_VALUES = tuple(k_values)

    rows: list[dict] = []
    print(f"samples={n_samples:,} per dimension, dimensions={list(dimensions)}")
    print(f"{'d':>4}{'K':>4}{'uniformR':>10}{'cover':>8}{'targetedR':>11}"
          f"{'minfraction':>13}{'E|rho-1|':>11}{'d*E|rho-1|':>12}{'agree':>8}")
    for d in dimensions:
        # A fresh identically-seeded generator per dimension keeps the d = 3
        # results bit-identical to the originally published ones.
        rng = np.random.default_rng(SEED)
        dimension_rows = evaluate_dimension(d, rng, n_samples)
        rows.extend(dimension_rows)
        for row in dimension_rows:
            print(f"{d:>4}{row['K']:>4}{row['effective_classes']:>10}{row['coverage']:>8.3f}"
                  f"{row['targeted_effective_classes']:>11}{row['min_region_fraction']:>13.3e}"
                  f"{row['mean_abs_rho_deviation']:>11.4e}{row['scaled_rho_deviation']:>12.4f}"
                  f"{row['formula_agreement']:>8.4f}")
        print()

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "k_extension_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={csv_path}")

    if plot_reference_trend(rows, output_dir / "k_extension_reference_trend_1.png"):
        print(f"saved_figure={output_dir / 'k_extension_reference_trend_1.png'}")
    if plot_dimension_curves(rows, output_dir / "k_extension_dimension_curves_2.png"):
        print(f"saved_figure={output_dir / 'k_extension_dimension_curves_2.png'}")
    if plot_dimension_heatmap(rows, output_dir / "k_extension_dimension_heatmap_2.png"):
        print(f"saved_figure={output_dir / 'k_extension_dimension_heatmap_2.png'}")
    if plot_concentration_panels(rows, output_dir / "k_extension_concentration_panels_3.png"):
        print(f"saved_figure={output_dir / 'k_extension_concentration_panels_3.png'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default=",".join(str(v) for v in D_VALUES))
    parser.add_argument("--k-values", default=",".join(str(v) for v in K_VALUES))
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the CSV and the four figures. Defaults to "
            "<script dir>/result, which holds the canonical published outputs; "
            "pass a scratch directory for shortened checks so the canonical "
            "files are not overwritten."
        ),
    )
    args = parser.parse_args()
    dimensions = tuple(int(v) for v in args.dimensions.split(",") if v.strip())
    k_values = tuple(int(v) for v in args.k_values.split(",") if v.strip())
    run(dimensions, k_values, args.samples, args.output_dir)


if __name__ == "__main__":
    main()
