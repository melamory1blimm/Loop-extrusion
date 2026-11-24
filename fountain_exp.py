#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hi-C extrusion-fountain analysis for three vertebrates (Gaussian-only).

This is the consolidated runner for the *current* cohesin-extrusion paper.
It replaces xenopus.py, danio.py and medaka.py, removing all numerical/PDE
fountain-kernel fitting, mechanistic parameter scans and Colab notebook cells.

RETAINED analyses:
  * Aggregate observed/expected (O/E) Hi-C fountains at 5-kb resolution;
  * Phenomenological anisotropic Gaussian fit and rho from its widths;
  * Gaussian fit robustness over 60, 80, 100, 130-kb windows (Table S8);
  * Bootstrap over individual fountains (50 repeats by default);
  * Direct intensity-weighted Pearson correlations around the fountain base
    and empirical peak, plus empirical-vs-Gaussian window comparison (Fig. S15);
  * Optional Gaussian-only tolerance scan for Table S9.

The original scripts call the project module ``fountain_tools_clean.py``.
That file is still required, since its Hi-C O/E extraction and Gaussian fitting
routines were NOT supplied in these three scripts. This one Python *runner*
replaces all three species-specific scripts; it does not replace the module.

Example (Google Colab, after mounting Drive):
    !python /content/drive/MyDrive/fountains/fountain_analysis_three_species.py \\
        --base-dir /content/drive/MyDrive/fountains

Local example:
    python fountain_analysis_three_species.py --base-dir ./fountains \\
        --species xenopus danio --n-bootstrap 50

Install packages used here and by the project library as needed:
    pip install numpy pandas matplotlib scipy tqdm cooler cooltools

CAVEATS IMPORTANT FOR REPRODUCIBILITY:
  1. Keep the same fountain_tools_clean.py version used in the paper.
  2. The aggregate builder and individual-fountain collector are distinct
     calls, as in the notebooks. Their outputs are compared; a discrepancy
     is reported, NOT silently corrected or used to alter scientific results.
  3. Empirical windows are square; Gaussian local fitting windows are not
     necessarily identical in shape. Comparison is qualitative/diagnostic.
  4. Figure uncertainty bands are the explicitly supplied manuscript values,
     not newly computed confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESOLUTION_BP = 5_000
TABLE_WINDOWS_KB = (60, 80, 100, 130)


@dataclass(frozen=True)
class Species:
    name: str
    label: str
    mcool: str
    fountain_csv: str
    flank_kb: int
    windows_kb: tuple[int, ...]
    color: str
    reference_rho: float
    reference_sd: float
    image_vmax: float
    hic_contour_levels: tuple[float, ...]
    gauss_contour_levels: tuple[float, ...]
    plot_zoom_kb: int | None = None
    fit_contour_width: float = 3.0
    example_empirical_window_kb: int = 80


SPECIES: dict[str, Species] = {
    "oryzias": Species(
        name="oryzias", label="Oryzias latipes",
        mcool="data/HiC_st10.oryLat2.mapq_30.1000.mcool",
        fountain_csv="fountains_madoka.csv",  # Original file spelling.
        flank_kb=100, windows_kb=tuple(range(30, 111, 10)),
        color="#EA8548", reference_rho=0.25, reference_sd=0.11,
        image_vmax=1.20, hic_contour_levels=(0.50, 0.70, 0.85, 0.95),
        gauss_contour_levels=(0.70, 0.85, 0.95),
        fit_contour_width=1.8, example_empirical_window_kb=30,
    ),
    "xenopus": Species(
        name="xenopus", label="Xenopus tropicalis",
        mcool="data/stage11.xenTro10.mapq_30.1000.mcool",
        fountain_csv="fountains_xenopus.csv",
        flank_kb=250, windows_kb=tuple(range(40, 111, 10)),
        color="#1649C0", reference_rho=0.30, reference_sd=0.11,
        image_vmax=1.70, hic_contour_levels=(0.70, 0.85, 0.95),
        gauss_contour_levels=(0.70, 0.85, 0.95),
        plot_zoom_kb=160,
    ),
    "danio": Species(
        name="danio", label="Danio rerio",
        mcool="data/GSE195609_Wild-Type_5.3.1000.mcool",
        fountain_csv="fountains.csv",
        flank_kb=200, windows_kb=tuple(range(40, 111, 10)),
        color="#25834B", reference_rho=0.30, reference_sd=0.13,
        image_vmax=1.40, hic_contour_levels=(0.70, 0.82, 0.94),
        gauss_contour_levels=(0.70, 0.82, 0.94),
        plot_zoom_kb=150,
    ),
}

# The common Gaussian fitting parameters are copied from all three notebooks.
# The initial broad-window fit uses 100 kb, whereas the Table S8 protocol uses
# a preliminary 80-kb window. The FINAL local fitting radius is 80 kb in both.
GAUSSIAN_KWARGS = dict(
    res=RESOLUTION_BP,
    p0_kb=25,
    p_half_window_kb=50,
    fit_window_kb=80,
    diag_exclusion_kb=10,
    two_step_peak_refit=True,
    near_peak_radius_kb=80,
    positive_weight=True,
    positive_weight_strength=3,
    robust=True,
    downweight_major_tail=True,
    major_tail_side="positive",
    major_tail_start_kb=20,
    major_tail_transition_kb=40,
    major_tail_min_scale=0.00001,
    peak_penalty=False,
    peak_weight=150.0,
    peak_scale_kb=5.0,
    peak_search_window_kb=80,
    peak_quantile=0.80,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated fountain analysis: empirical correlations and Gaussian fits only."
    )
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/fountains",
                        help="Directory containing data/, the fountain CSVs and fountain_tools_clean.py")
    parser.add_argument("--tools-dir", default=None,
                        help="Directory with fountain_tools_clean.py, if different from --base-dir")
    parser.add_argument("--out-dir", default=None,
                        help="Default: <base-dir>/fountain_gaussian_results")
    parser.add_argument("--species", nargs="+", choices=tuple(SPECIES) + ("all",),
                        default=["all"], help="One, several or all species")
    parser.add_argument("--n-bootstrap", type=int, default=50,
                        help="Number of fountain-bootstrap replicates (default: 50; 0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=4,
                        help="Processes used for cooltools expected-distance calculation")
    parser.add_argument("--gaussian-tolerance-scan", action="store_true",
                        help="Run the expensive Gaussian a/b/p uncertainty scan for Table S9")
    parser.add_argument("--diagnostic-figures", action="store_true",
                        help="Additionally save smoothed aggregate and empirical-window mask figures")
    parser.add_argument("--dry-run", action="store_true",
                        help="List input files/settings without importing scientific libraries or processing data")
    args = parser.parse_args()
    if args.n_bootstrap < 0 or args.nproc < 1:
        parser.error("--n-bootstrap must be >= 0 and --nproc must be >= 1")
    return args


def selected_species(names: list[str]) -> list[str]:
    return list(SPECIES) if "all" in names else list(dict.fromkeys(names))


def require_project_tools(tools_dir: Path):
    sys.path.insert(0, str(tools_dir))
    try:
        import fountain_tools_clean as ft
    except ImportError as exc:
        raise RuntimeError(
            "Could not import fountain_tools_clean.py. "
            f"Place the project module in {tools_dir} or pass --tools-dir. "
            "Also check numpy, scipy, cooler and cooltools installations."
        ) from exc
    required = [
        "build_weighted_aggregate_fountain",
        "collect_individual_fountain_Zs",
        "nan_weighted_mean_stack",
        "fit_aggregate_fountain_old",
        "plot_aggregate_fountain_fit",
        "fit_aggregate_same_settings",
        "bootstrap_fountains_same_settings",
        "empirical_fountain_correlation",
    ]
    missing = [name for name in required if not hasattr(ft, name)]
    if missing:
        raise RuntimeError(
            "The installed fountain_tools_clean.py lacks functions used by the notebooks: "
            + ", ".join(missing)
            + ". Use the same updated module version as in your Colab analysis."
        )
    return ft


def arrays_match_reference(ft, aggregate, stack, used_df):
    """Cross-check the two *original* extraction routes; preserve their output."""
    import numpy as np
    import pandas as pd

    if len(used_df) != len(stack):
        raise ValueError("The used_df rows do not match the rows of Z_stack")
    weights = None
    if "weight" in used_df.columns:
        weights = pd.to_numeric(used_df["weight"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid weights in used_df['weight']")
    recomputed = ft.nan_weighted_mean_stack(stack, weights=weights)
    if recomputed.shape != aggregate.shape:
        warnings.warn("Shape differs between collected and aggregated fountains")
        return weights, None
    same = np.allclose(aggregate, recomputed, atol=1e-8, rtol=1e-6, equal_nan=True)
    finite = np.isfinite(aggregate) & np.isfinite(recomputed)
    max_abs_diff = float(np.max(np.abs(aggregate[finite] - recomputed[finite]))) if finite.any() else None
    if not same:
        warnings.warn(
            f"Aggregate differs from the aggregate reconstructed from individual "
            f"fountains; max absolute O/E discrepancy = {max_abs_diff!r}. "
            "Check the version of fountain_tools_clean and its weighting/shift settings. "
            "No automatic scientific-data changes have been made."
        )
    return weights, max_abs_diff


def extract_fountains(ft, species: Species, base_dir: Path, out: Path, nproc: int):
    import numpy as np
    import pandas as pd

    mcool = base_dir / species.mcool
    annotations = base_dir / species.fountain_csv
    if not mcool.is_file() or not annotations.is_file():
        missing = [str(p) for p in (mcool, annotations) if not p.is_file()]
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))

    # As in the notebooks, the published aggregate and bootstrap stack have
    # distinct extraction calls. Both use the same 5-kb resolution and flank.
    aggregate, aggregate_info, aggregate_used, aggregate_failed, _ = (
        ft.build_weighted_aggregate_fountain(
            cool_path=str(mcool), fountains_csv=str(annotations),
            flank=species.flank_kb * 1000, res=RESOLUTION_BP,
            weight_col="Fountain Score", z_transform="oe",
            expected_nproc=nproc, min_weight=0, require_full_window=True,
        )
    )
    individual_list, stack, used_df, failed_df, _, _ = ft.collect_individual_fountain_Zs(
        cool_path=str(mcool), fountains_csv=str(annotations),
        res=RESOLUTION_BP, flank=species.flank_kb * 1000,
        z_transform="oe", expected_nproc=1,
        weight_col=None, min_weight=None, require_full_window=True,
        base_shift_bp=0, max_fountains=None,
    )
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 2 or stack.shape[1] != stack.shape[2]:
        raise ValueError(f"Unexpected Z_stack shape {stack.shape}; at least two fountains needed")
    aggregate = np.asarray(aggregate, dtype=float)
    if aggregate.ndim != 2 or aggregate.shape[0] != aggregate.shape[1]:
        raise ValueError(f"Expected a square aggregate_Z, got {aggregate.shape}")

    # Deliberately save the raw, original aggregate (not the reconstructed one).
    np.save(out / "aggregate_Z.npy", aggregate)
    aggregate_used.to_csv(out / "aggregate_fountains_used.csv", index=False)
    aggregate_failed.to_csv(out / "aggregate_fountains_failed.csv", index=False)
    used_df.to_csv(out / "individual_fountains_used.csv", index=False)
    failed_df.to_csv(out / "individual_fountains_failed.csv", index=False)
    weights, diff = arrays_match_reference(ft, aggregate, stack, used_df)
    print(f"  aggregate: {len(aggregate_used)} used; stack: {len(stack)} individual fountains")
    return aggregate, stack, used_df, weights, diff, len(aggregate_used)


def fit_gaussian(ft, aggregate):
    settings = dict(GAUSSIAN_KWARGS)
    settings.update(
        fit_window_kb=100,
        compute_full_nrmse=True,
        full_nrmse_normalization="range",
        full_nrmse_mirror=True,
        full_nrmse_exclude_diagonal=False,
    )
    return ft.fit_aggregate_fountain_old(aggregate, **settings)


def plot_gaussian_fit(ft, species: Species, aggregate, fit, cache, output: Path):
    import matplotlib.pyplot as plt

    opts = dict(
        image_vmin=0.8, image_vmax=species.image_vmax,
        figsize=(6.5, 5.8), tick_fontsize=16,
        x_tick_step=100 if species.name != "oryzias" else 50,
        y_tick_step=100 if species.name != "oryzias" else 50,
        colorbar_tick_fontsize=16, colorbar_tick_step=0.1,
        output=str(output),
        show_hic_contours=True, hic_contour_smooth_sigma=1.5,
        norm_hic_levels=species.hic_contour_levels,
        norm_gaussian_levels=species.gauss_contour_levels,
        hic_contour_color="black", hic_contour_linewidth=3,
        hic_contour_linestyle="--", show_fit_contours=True,
        fit_contour_color="fuchsia", fit_contour_linewidth=species.fit_contour_width,
        show_fit_contours_mirror=True, fit_contour_linestyle="-",
    )
    if species.plot_zoom_kb:
        opts.update(xlim=(-species.plot_zoom_kb, species.plot_zoom_kb),
                    ylim=(species.plot_zoom_kb, -species.plot_zoom_kb))
    fig, _ = ft.plot_aggregate_fountain_fit(aggregate, fit, cache, **opts)
    plt.close(fig)


def get_peak_xy(ft, aggregate, fit):
    """Use the empirical peak already returned by the original Gaussian fit."""
    import numpy as np

    xy = (float(fit.get("hic_peak_x_kb", np.nan)),
          float(fit.get("hic_peak_y_kb", np.nan)))
    if np.isfinite(xy).all():
        return xy
    if not hasattr(ft, "estimate_empirical_fountain_peak"):
        raise ValueError("Gaussian fit has no empirical peak and helper cannot estimate it")
    result = ft.estimate_empirical_fountain_peak(
        aggregate, res=RESOLUTION_BP,
        search_center_xy_kb=(40, -40), search_half_window_kb=50,
        diag_exclusion_kb=10, peak_quantile=0.80, smooth_sigma_bins=1.0,
    )
    return tuple(map(float, result["peak_xy_kb"]))


def gaussian_table_species(ft, species: Species, stack, weights,
                           n_boot: int, seed: int):
    import numpy as np
    import pandas as pd

    rows, bootstrap_frames = [], []
    for window in TABLE_WINDOWS_KB:
        settings = dict(GAUSSIAN_KWARGS, near_peak_radius_kb=float(window),
                        compute_full_nrmse=False)
        result = ft.fit_aggregate_same_settings(
            Z_stack=stack, fit_kwargs=settings, weights=weights,
            nrmse_normalization="range", peak_radius_kb=60,
            full_nrmse_mirror=False,
        )
        row = result["row"]
        r_boot = pd.Series(dtype=float)
        if n_boot:
            boot = ft.bootstrap_fountains_same_settings(
                Z_stack=stack, fit_kwargs=settings, weights=weights,
                n_boot=n_boot, random_state=seed,
                nrmse_normalization="range", peak_radius_kb=60,
                full_nrmse_mirror=False, return_matrices=False,
            )["boot_df"].copy()
            if "r" not in boot.columns:
                raise RuntimeError(f"No 'r' column returned from bootstrap for {species.name}")
            okay = boot["status"].eq("ok") if "status" in boot else pd.Series(True, index=boot.index)
            r_boot = pd.to_numeric(boot.loc[okay, "r"], errors="coerce").dropna()
            if len(r_boot) != n_boot:
                warnings.warn(f"{species.name}/{window} kb: {len(r_boot)}/{n_boot} successful bootstrap fits")
            boot.insert(0, "Species", species.label)
            boot.insert(1, "Window (kb)", window)
            bootstrap_frames.append(boot)
        record = {
            "Species": species.label,
            "Window (kb)": window,
            "rho": float(row["r"]),
            "SE": float(row.get("r_se", np.nan)),
            "Bootstrap mean": float(r_boot.mean()) if len(r_boot) else np.nan,
            "Bootstrap SD": float(r_boot.std(ddof=1)) if len(r_boot) > 1 else np.nan,
            "n_fountains": len(stack),
            "n_boot_success": len(r_boot),
            "a_kb": float(row.get("a_kb", np.nan)),
            "b_kb": float(row.get("b_kb", np.nan)),
        }
        rows.append(record)
        print(f"    Table S8, {window:3d} kb: rho={record['rho']:.3f}, "
              f"bootstrap SD={record['Bootstrap SD']:.3f}")
    boot_all = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    return pd.DataFrame(rows), boot_all


def empirical_scan(ft, species: Species, aggregate, peak_xy):
    import numpy as np
    import pandas as pd

    # Compute direct empirical correlations at every window size; Gaussian
    # values at S8 windows are recycled instead of fitting twice.
    records = []
    for window in species.windows_kb:
        measured = ft.empirical_fountain_correlation(
            aggregate, res=RESOLUTION_BP, center="peak", peak_xy_kb=peak_xy,
            half_window_kb=window, diag_exclusion_kb=10, signal_mode="excess",
        )
        # Do NOT reuse Table S8 fit results here: the S8 bootstrap stack and
        # the published aggregate use separate extraction calls in the sources.
        # Fig. S15 must consistently use the SAME aggregate as empirical rho.
        settings = dict(GAUSSIAN_KWARGS,
                        near_peak_radius_kb=float(window), compute_full_nrmse=False)
        row, _ = ft.fit_aggregate_fountain_old(aggregate, **settings)
        r_gauss = float(row["r"])
        gauss_se = float(row.get("r_se", np.nan))
        gauss_source = "Gaussian fit (published aggregate)"
        records.append({
            "Species": species.label, "Window (kb)": window,
            "r_arms": float(measured["r_pearson_arms"]),
            "r_gaussian": r_gauss, "r_se": gauss_se,
            "n_weighted_pixels": int(measured["n_weighted_pixels"]),
            "peak_x_kb": peak_xy[0], "peak_y_kb": peak_xy[1],
            "gaussian_source": gauss_source,
        })
    return pd.DataFrame(records)


def empirical_base_peak_table(ft, species: Species, aggregate, peak_xy):
    import pandas as pd

    records = []
    for center in ("base", "peak"):
        for window in species.windows_kb:
            measured = ft.empirical_fountain_correlation(
                aggregate, res=RESOLUTION_BP, center=center, peak_xy_kb=peak_xy,
                half_window_kb=window, diag_exclusion_kb=10, signal_mode="excess",
            )
            records.append({
                "Species": species.label, "center": center,
                "half_window_kb": window,
                "r_XY": float(measured["r_pearson_xy"]),
                "r_arms": float(measured["r_pearson_arms"]),
                "rho_origin": float(measured["rho_origin_xy"]),
                "n_pixels": int(measured["n_weighted_pixels"]),
            })
    return pd.DataFrame(records)


def plot_window_scan(species: Species, scan, output: Path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    import numpy as np

    scan = scan.sort_values("Window (kb)")
    x = scan["Window (kb)"].to_numpy(dtype=float)
    r_arms = scan["r_arms"].to_numpy(dtype=float)
    r_gauss = scan["r_gaussian"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    low = species.reference_rho - species.reference_sd
    high = species.reference_rho + species.reference_sd
    ax.axhspan(low, high, facecolor=species.color, alpha=0.11,
               label="Reference uncertainty range")
    reference_row = scan.loc[scan["Window (kb)"].eq(80), "r_gaussian"]
    if len(reference_row) != 1:
        raise ValueError("An 80-kb Gaussian estimate is required")
    rho_80 = float(reference_row.iloc[0])
    ax.axhline(rho_80, color=species.color, lw=2,
               label="Gaussian fit (80 kb)")
    ax.plot(x, r_arms, color=species.color, marker="s", ms=5.8,
            lw=2.4, label="Empirical correlation")
    ax.plot(x, r_gauss, color=species.color, marker="o", ms=5.4,
            mfc="white", mew=1.2, lw=1.7, ls="--", dashes=(4, 3),
            label="Gaussian fit")
    ax.set(title=species.label, xlabel="Window size, kb", ylabel="Correlation",
           xlim=(0, 120), ylim=(0, 0.45))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="major", direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_combined_scans(species_scans, output: Path):
    import matplotlib.pyplot as plt

    if not species_scans:
        return
    n = len(species_scans)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.7), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (cfg, scan) in zip(axes[0], species_scans):
        sorted_df = scan.sort_values("Window (kb)")
        ax.axhspan(cfg.reference_rho - cfg.reference_sd,
                   cfg.reference_rho + cfg.reference_sd,
                   color=cfg.color, alpha=0.10)
        reference = sorted_df.loc[sorted_df["Window (kb)"].eq(80), "r_gaussian"]
        if len(reference) != 1:
            raise ValueError(f"No 80-kb Gaussian fit for {cfg.name}")
        ax.axhline(float(reference.iloc[0]), color=cfg.color, lw=1.6)
        ax.plot(sorted_df["Window (kb)"], sorted_df["r_arms"],
                color=cfg.color, lw=2, marker="s", markersize=4,
                label="Empirical")
        ax.plot(sorted_df["Window (kb)"], sorted_df["r_gaussian"],
                color=cfg.color, ls="--", lw=1.7, marker="o", markersize=4,
                mfc="white", label="Gaussian")
        ax.set(title=cfg.label, xlim=(20, 120), ylim=(0, 0.5), xlabel="Window size, kb")
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[0][0].set_ylabel("Arm correlation, rho")
    axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_s8_latex(table, filename: Path, n_bootstrap: int):
    import pandas as pd

    lines = [
        r"\begin{table}[p]", r"\centering",
        r"\caption{Robustness of Gaussian-fit correlation estimates. "
        r"For each species, the fit was repeated over different local fitting "
        r"windows. The table lists the fitted correlation, its local standard "
        r"error, and the bootstrap mean and standard deviation "
        + f"({n_bootstrap} replicates)." + r"}",
        r"\label{tab:gaussian_fit_robustness}",
        r"\begin{tabular}{llllll}", r"\hline",
        r"Species & Window (kb) & $\rho$ & SE & Bootstrap mean & Bootstrap SD \\",
        r"\hline",
    ]
    for name in SPECIES:
        group = table.loc[table["Species"] == SPECIES[name].label]
        if group.empty:
            continue
        for i, (_, row) in enumerate(group.sort_values("Window (kb)").iterrows()):
            label = rf"\textit{{{SPECIES[name].label}}}" if i == 0 else ""
            vals = [row["rho"], row["SE"], row["Bootstrap mean"], row["Bootstrap SD"]]
            fmt = [f"{v:.3f}" if pd.notna(v) else "--" for v in vals]
            lines.append(label + f" & {int(row['Window (kb)'])} & " + " & ".join(fmt) + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table}"])
    filename.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_gaussian_tolerance(ft, aggregate, fit, cache, out: Path):
    import pandas as pd

    if not hasattr(ft, "scan_gaussian_nrmse_tolerance_region"):
        raise RuntimeError("Project module lacks scan_gaussian_nrmse_tolerance_region")
    scan = ft.scan_gaussian_nrmse_tolerance_region(
        aggregate_Z=aggregate, fit=fit, fit_cache=cache,
        fit_kwargs=dict(GAUSSIAN_KWARGS),
        nrmse_delta=0.03, score_mode="peak60", nrmse_normalization="range",
        full_mode="finite_all", peak_radius_kb=60,
        full_nrmse_mirror=False, a_rel_span=(0.5, 2.0),
        b_rel_span=(0.5, 2.0), p_span_kb=50,
        n_a=45, n_b=45, n_p=41,
    )
    summary = scan.get("summary", {})
    pd.Series(summary).to_csv(out / "gaussian_tolerance_summary.csv", header=["value"])
    return summary


def run_species(ft, cfg: Species, base_dir: Path, out_dir: Path, args):
    import matplotlib.pyplot as plt
    import numpy as np

    species_out = out_dir / cfg.name
    species_out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {cfg.label} ===", flush=True)
    aggregate, stack, used_df, weights, aggregate_diff, n_agg = extract_fountains(
        ft, cfg, base_dir, species_out, args.nproc,
    )
    fit, fit_cache = fit_gaussian(ft, aggregate)
    peak_xy = get_peak_xy(ft, aggregate, fit)
    print(f"  Gaussian rho={float(fit['r']):.4f}, peak={peak_xy}")
    plot_gaussian_fit(ft, cfg, aggregate, fit, fit_cache,
                      species_out / f"{cfg.name}_gaussian_fountain.pdf")

    table, boot = gaussian_table_species(
        ft, cfg, stack, weights, args.n_bootstrap, args.seed,
    )
    table.to_csv(species_out / "gaussian_window_robustness.csv", index=False)
    if not boot.empty:
        boot.to_csv(species_out / "gaussian_bootstrap_raw.csv", index=False)

    comparisons = empirical_scan(ft, cfg, aggregate, peak_xy)
    comparisons.to_csv(species_out / "empirical_vs_gaussian_windows.csv", index=False)
    plot_window_scan(cfg, comparisons, species_out / f"{cfg.name}_correlation_windows.pdf")

    direct = empirical_base_peak_table(ft, cfg, aggregate, peak_xy)
    direct.to_csv(species_out / "direct_correlations_base_vs_peak.csv", index=False)
    if args.diagnostic_figures:
        if hasattr(ft, "plot_smoothed_hic_contours"):
            fig, _, _, _ = ft.plot_smoothed_hic_contours(
                aggregate, res=RESOLUTION_BP, sigma=1.2,
                cmap="coolwarm", vmin=0.8, vmax=cfg.image_vmax,
                linecolor="blue", linewidth=1.5, show_image=True,
                levels=[1.04, 1.1, 1.15, 1.2, 1.25, 1.5, 1.6],
            )
            fig.savefig(species_out / "aggregate_diagnostic.pdf", bbox_inches="tight")
            plt.close(fig)
        if hasattr(ft, "plot_fountain_correlation_window"):
            result = ft.empirical_fountain_correlation(
                aggregate, res=RESOLUTION_BP, center="peak", peak_xy_kb=peak_xy,
                half_window_kb=cfg.example_empirical_window_kb,
                diag_exclusion_kb=10, signal_mode="excess", return_mask=True,
            )
            fig, _ = ft.plot_fountain_correlation_window(
                aggregate, result, res=RESOLUTION_BP,
            )
            fig.savefig(species_out / "empirical_window_mask.pdf", bbox_inches="tight")
            plt.close(fig)

    if args.gaussian_tolerance_scan:
        scan_gaussian_tolerance(ft, aggregate, fit, fit_cache, species_out)

    summary = {
        "Species": cfg.label,
        "n_fountains_aggregate": int(n_agg),
        "n_fountains_stack": int(len(stack)),
        "Gaussian rho": float(fit["r"]),
        "Gaussian SE": float(fit.get("r_se", np.nan)),
        "Gaussian NRMSE percent": float(fit.get("nrmse_percent", np.nan)),
        "Empirical peak x (kb)": float(peak_xy[0]),
        "Empirical peak y (kb)": float(peak_xy[1]),
        "Max aggregate discrepancy": aggregate_diff,
    }
    # Keep settings as a machine-readable record, separate from published values.
    with (species_out / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump({"species": asdict(cfg), "n_bootstrap": args.n_bootstrap,
                   "seed": args.seed, "resolution_bp": RESOLUTION_BP,
                   "gaussian_kwargs": GAUSSIAN_KWARGS,
                   "gaussian_tolerance_scan": args.gaussian_tolerance_scan}, f, indent=2)
    return summary, table, comparisons


def main():
    args = parse_args()
    names = selected_species(args.species)
    base_dir = Path(args.base_dir).expanduser().resolve()
    tools_dir = Path(args.tools_dir).expanduser().resolve() if args.tools_dir else base_dir
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else base_dir / "fountain_gaussian_results")
    if args.dry_run:
        for name in names:
            cfg = SPECIES[name]
            print(f"{cfg.label}: flank={cfg.flank_kb} kb, "
                  f"Hi-C={base_dir / cfg.mcool} (exists={(base_dir/cfg.mcool).exists()}), "
                  f"CSV={base_dir / cfg.fountain_csv} (exists={(base_dir/cfg.fountain_csv).exists()})")
        print(f"Project module expected: {tools_dir / 'fountain_tools_clean.py'}")
        print("No analyses were run (--dry-run).")
        return
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"The data directory does not exist: {base_dir}. "
            "In Google Colab, mount Drive first or specify --base-dir."
        )

    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    ft = require_project_tools(tools_dir)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"cooltools.*")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, tables, comparisons = [], [], []
    plotted = []
    for name in names:
        cfg = SPECIES[name]
        summary, table, correlation_scan = run_species(ft, cfg, base_dir, out_dir, args)
        summaries.append(summary)
        tables.append(table)
        comparisons.append(correlation_scan)
        plotted.append((cfg, correlation_scan))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "species_summary.csv", index=False)
    full_table = pd.concat(tables, ignore_index=True)
    full_table.to_csv(out_dir / "table_S8_gaussian_robustness.csv", index=False)
    write_s8_latex(full_table, out_dir / "table_S8_gaussian_robustness.tex", args.n_bootstrap)
    pd.concat(comparisons, ignore_index=True).to_csv(
        out_dir / "figure_S15_correlation_windows.csv", index=False,
    )
    plot_combined_scans(plotted, out_dir / "figure_S15_correlation_windows.pdf")
    print(f"\nSaved Gaussian/empirical results to: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
