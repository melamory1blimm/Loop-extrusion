"""
fountain_tools.py

Utilities for working with Hi-C fountains, aggregate fountain fitting,
Dome PADRE/ChromHMM enhancer/promoter annotation, and E-P visualization.
Keep this file as a library: no analysis should run at import time.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from scipy.optimize import least_squares
from scipy.special import erf

from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PatchCollection

import cooler
import cooltools
from tqdm.auto import tqdm
from matplotlib.lines import Line2D


def load_fountains_csv(path, sep=';'):
    df = pd.read_csv(path, sep=sep)
    required = ['Fountain index', 'chrom', 'start', 'end', 'Fountain Score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'В CSV не хватает колонок: {missing}')
    df = df.copy()
    df['chrom'] = df['chrom'].astype(str)
    df['base_bp'] = (df['start'].astype(int) + df['end'].astype(int)) // 2
    return df


def pick_expected_column(expected):
    if 'balanced.avg' in expected.columns:
        return 'balanced.avg'
    if 'balanced.avg.smoothed' in expected.columns:
        return 'balanced.avg.smoothed'
    if 'balanced.avg.smoothed.agg' in expected.columns:
        return 'balanced.avg.smoothed.agg'
    if 'count.avg' in expected.columns:
        return 'count.avg'
    raise ValueError(f'Не нашла подходящую колонку expected. Доступные колонки: {list(expected.columns)}')


def compute_expected_by_chrom(clr, chroms, smooth=False, aggregate_smoothed=False, nproc=1, chunksize=1000000):
    """
    Считает expected_cis один раз для всех нужных хромосом.
    Возвращает dict:
        expected_by_chrom["chr6"] = Series, index=dist, values=expected
    """
    chroms = list(dict.fromkeys(chroms))
    view_df = pd.DataFrame({'chrom': chroms, 'start': [0] * len(chroms), 'end': [int(clr.chromsizes[c]) for c in chroms], 'name': chroms})
    expected = cooltools.expected_cis(clr, view_df=view_df, smooth=smooth, aggregate_smoothed=aggregate_smoothed, nproc=nproc, chunksize=chunksize)
    exp_col = pick_expected_column(expected)
    expected_by_chrom = {}
    for chrom in chroms:
        exp_chr = expected[(expected['region1'] == chrom) & (expected['region2'] == chrom)].copy()
        s = exp_chr.set_index('dist')[exp_col]
        expected_by_chrom[chrom] = s
    return (expected_by_chrom, expected, exp_col)


class ExpectedMatrixCache:

    def __init__(self, expected_by_chrom):
        self.expected_by_chrom = expected_by_chrom
        self.diag_cache = {}
        self.exp_mat_cache = {}

    def get_diag_ids(self, n):
        if n not in self.diag_cache:
            idx = np.arange(n)
            self.diag_cache[n] = np.abs(np.subtract.outer(idx, idx))
        return self.diag_cache[n]

    def get_exp_mat(self, chrom, n):
        key = (chrom, n)
        if key in self.exp_mat_cache:
            return self.exp_mat_cache[key]
        exp_by_diag = self.expected_by_chrom[chrom]
        diag_ids = self.get_diag_ids(n)
        exp_mat = exp_by_diag.reindex(diag_ids.ravel()).to_numpy().reshape(n, n)
        exp_mat = exp_mat.astype(float)
        exp_mat[~np.isfinite(exp_mat)] = np.nan
        exp_mat[exp_mat <= 0] = np.nan
        self.exp_mat_cache[key] = exp_mat
        return exp_mat


class FountainFitCache:
    """
    Кэширует всё, что зависит только от размера окна n:
    - координаты X, Y;
    - маску верхнего треугольника;
    - локальное окно вокруг ожидаемого фонтана.
    """

    def __init__(self, bin_kb=10, p0_kb=50, p_half_window_kb=40, diag_exclusion_kb=10, fit_window_kb=180):
        self.bin_kb = bin_kb
        self.p0_kb = p0_kb
        self.p_half_window_kb = p_half_window_kb
        self.diag_exclusion_kb = diag_exclusion_kb
        self.fit_window_kb = fit_window_kb
        self.cache = {}

    def get(self, n):
        if n in self.cache:
            return self.cache[n]
        x_kb = make_centered_coords(n, self.bin_kb)
        y_kb = make_centered_coords(n, self.bin_kb)
        X, Y = np.meshgrid(x_kb, y_kb)
        mask = Y < X - self.diag_exclusion_kb
        u0 = X + Y
        v0 = X - Y - 2 * self.p0_kb
        mask &= np.abs(u0) <= self.fit_window_kb
        mask &= np.abs(v0) <= self.fit_window_kb
        context = {'x_kb': x_kb, 'y_kb': y_kb, 'X': X, 'Y': Y, 'base_mask': mask}
        self.cache[n] = context
        return context


import numpy as np
from scipy.optimize import least_squares


def fit_one_fountain_matrix(
    Z,
    fit_cache,
    p0_kb=50,
    p_half_window_kb=40,
    positive_weight=True,
    positive_weight_strength=2.0,
    robust=True,
    min_points=20,
    near_peak_radius_kb=90,
    two_step_peak_refit=True,

    # Подавление асимметричного хвоста вдоль major-axis
    downweight_major_tail=True,
    major_tail_side="positive",
    major_tail_start_kb=20,
    major_tail_transition_kb=40,
    major_tail_min_scale=0.25,

    # Peak anchor penalty
    peak_penalty=True,
    peak_weight=100.0,
    peak_scale_kb=5.0,
    peak_search_window_kb=70,
    peak_quantile=0.80,
    peak_center_kb=None,
    peak_min_points=3,
):
    """
    Фитит один фонтан 2D-гауссианом.

    Peak penalty:
    -------------
    Сначала находим центр масс верхушки Hi-C пика.
    Затем штрафуем положение гауссовского пика.

    В данной параметризации:
        Gaussian peak = (p, -p)

    RMSE:
    -----
    rmse считается на финальном окне фита как:

        sqrt(mean((G - H)^2))

    где G — Gaussian fit, H — Hi-C signal.

    rmse не включает веса и peak penalty.
    weighted_rmse учитывает веса пикселей.
    objective_rms_residual учитывает всё, включая peak penalty.
    """

    Z = np.asarray(Z, dtype=float)

    n, m = Z.shape
    if n != m:
        raise ValueError(f"Матрица не квадратная: {Z.shape}")

    if major_tail_side not in {"positive", "negative"}:
        raise ValueError("major_tail_side должен быть 'positive' или 'negative'")

    if not 0 < major_tail_min_scale <= 1:
        raise ValueError("major_tail_min_scale должен находиться в интервале (0, 1]")

    if major_tail_transition_kb < 0:
        raise ValueError("major_tail_transition_kb не может быть отрицательным")

    if positive_weight_strength < 0:
        raise ValueError("positive_weight_strength не может быть отрицательным")

    if peak_weight < 0:
        raise ValueError("peak_weight не может быть отрицательным")

    if peak_scale_kb <= 0:
        raise ValueError("peak_scale_kb должен быть положительным")

    if not 0 < peak_quantile < 1:
        raise ValueError("peak_quantile должен быть в интервале (0, 1)")

    ctx = fit_cache.get(n)
    X = ctx["X"]
    Y = ctx["Y"]

    finite_mask = np.isfinite(Z)

    # Верхний треугольник без главной диагонали
    base_mask = finite_mask.copy()
    base_mask &= Y < X - fit_cache.diag_exclusion_kb

    # Широкое окно вокруг ожидаемого положения p0
    U0_raw = X + Y
    V0_raw = X - Y - 2 * p0_kb

    broad_mask = base_mask.copy()
    broad_mask &= np.abs(U0_raw) <= fit_cache.fit_window_kb
    broad_mask &= np.abs(V0_raw) <= fit_cache.fit_window_kb

    coord_span = max(
        ctx["x_kb"].max() - ctx["x_kb"].min(),
        ctx["y_kb"].max() - ctx["y_kb"].min(),
    )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def resolve_peak_center():
        if peak_center_kb is None:
            return float(p0_kb), float(-p0_kb)

        if np.isscalar(peak_center_kb):
            p = float(peak_center_kb)
            return p, -p

        cx, cy = peak_center_kb
        return float(cx), float(cy)

    def peak_center_of_mass(values, x, y):
        """
        Центр масс верхушки пика:
        берем пиксели выше peak_quantile и взвешиваем excess над порогом.
        """

        values = np.asarray(values, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = (
            np.isfinite(values)
            & np.isfinite(x)
            & np.isfinite(y)
        )

        if valid.sum() < peak_min_points:
            return np.nan, np.nan, int(valid.sum())

        vv = values[valid]
        xx = x[valid]
        yy = y[valid]

        thr = np.nanquantile(vv, peak_quantile)
        top = vv >= thr

        if top.sum() < peak_min_points:
            return np.nan, np.nan, int(top.sum())

        weights_peak = vv[top] - thr

        if np.sum(weights_peak) <= 0:
            weights_peak = np.ones_like(weights_peak)

        cx = np.average(xx[top], weights=weights_peak)
        cy = np.average(yy[top], weights=weights_peak)

        return float(cx), float(cy), int(top.sum())

    def calculate_tail_scale(x_fit, y_fit, p_center):
        """
        Рассчитывает плавный множитель веса вдоль major-axis.
        major_coord = 0 соответствует центру найденного пика.
        """

        major_coord = (
            x_fit - y_fit - 2 * p_center
        ) / np.sqrt(2)

        if major_tail_side == "positive":
            signed_coord = major_coord
        else:
            signed_coord = -major_coord

        if major_tail_transition_kb > 0:
            t = (
                signed_coord - major_tail_start_kb
            ) / major_tail_transition_kb

            t = np.clip(t, 0.0, 1.0)

            smooth_t = t**2 * (3.0 - 2.0 * t)

            tail_scale = (
                1.0
                - (1.0 - major_tail_min_scale) * smooth_t
            )

        else:
            tail_scale = np.where(
                signed_coord > major_tail_start_kb,
                major_tail_min_scale,
                1.0,
            )

        return tail_scale, major_coord

    # ----------------------------------------------------------
    # Inner fit
    # ----------------------------------------------------------

    def run_fit(
        mask,
        theta0=None,
        tail_center_p=None,
    ):
        """
        Выполняет один фит.

        Если tail_center_p is None, хвост специально не подавляется.
        Если передан p, рассчитываются веса относительно этого центра.
        """

        mask = mask.copy()
        mask &= finite_mask

        x_fit = X[mask]
        y_fit = Y[mask]
        z_fit = Z[mask]

        if len(z_fit) < min_points:
            raise ValueError(f"Слишком мало точек для фита: {len(z_fit)}")

        C0 = np.nanmedian(z_fit)
        high = np.nanpercentile(z_fit, 95)
        A0 = max(high - C0, 0.001)

        if theta0 is None:
            a0 = fit_cache.fit_window_kb / 3
            b0 = fit_cache.fit_window_kb / 3

            theta0 = np.array(
                [a0, b0, p0_kb, A0, C0],
                dtype=float,
            )
        else:
            theta0 = np.asarray(theta0, dtype=float).copy()

        z_min = np.nanmin(z_fit)
        z_max = np.nanmax(z_fit)
        z_range = max(z_max - z_min, 0.001)

        lower_bounds = np.array(
            [
                fit_cache.bin_kb / 2,
                fit_cache.bin_kb / 2,
                p0_kb - p_half_window_kb,
                0.0,
                z_min - z_range,
            ],
            dtype=float,
        )

        upper_bounds = np.array(
            [
                coord_span * 2,
                coord_span * 2,
                p0_kb + p_half_window_kb,
                z_range * 5,
                z_max + z_range,
            ],
            dtype=float,
        )

        # Гарантируем нахождение theta0 строго внутри bounds
        eps_bound = 1e-9

        theta0 = np.maximum(theta0, lower_bounds + eps_bound)
        theta0 = np.minimum(theta0, upper_bounds - eps_bound)

        # ------------------------------------------------------
        # Усиление положительно обогащенных пикселей
        # ------------------------------------------------------

        if positive_weight:
            q70 = np.nanpercentile(z_fit, 70)
            q95 = np.nanpercentile(z_fit, 95)

            denom = max(q95 - q70, 1e-6)

            enrichment = np.clip(
                (z_fit - q70) / denom,
                0.0,
                1.0,
            )

            positive_bonus = positive_weight_strength * enrichment
        else:
            positive_bonus = np.zeros_like(z_fit)

        # ------------------------------------------------------
        # Подавление major-хвоста
        # ------------------------------------------------------

        if downweight_major_tail and tail_center_p is not None:
            tail_scale, major_coord = calculate_tail_scale(
                x_fit,
                y_fit,
                tail_center_p,
            )
        else:
            tail_scale = np.ones_like(z_fit)
            major_coord = (
                x_fit - y_fit - 2 * p0_kb
            ) / np.sqrt(2)

        weights = tail_scale * (
            1.0 + positive_bonus * tail_scale
        )

        # ------------------------------------------------------
        # Hi-C peak anchor
        # ------------------------------------------------------

        peak_is_valid = False

        if peak_penalty and peak_weight > 0:
            peak_cx, peak_cy = resolve_peak_center()

            peak_mask_fit = (
                (np.abs(x_fit - peak_cx) <= peak_search_window_kb)
                & (np.abs(y_fit - peak_cy) <= peak_search_window_kb)
                & np.isfinite(z_fit)
            )

            if peak_mask_fit.sum() >= peak_min_points:
                hic_peak_x, hic_peak_y, hic_peak_n = peak_center_of_mass(
                    z_fit[peak_mask_fit],
                    x_fit[peak_mask_fit],
                    y_fit[peak_mask_fit],
                )

                if np.isfinite(hic_peak_x) and np.isfinite(hic_peak_y):
                    peak_is_valid = True
            else:
                hic_peak_x = np.nan
                hic_peak_y = np.nan
                hic_peak_n = int(peak_mask_fit.sum())

        else:
            peak_mask_fit = None
            hic_peak_x = np.nan
            hic_peak_y = np.nan
            hic_peak_n = 0

        # ------------------------------------------------------
        # Residuals
        # ------------------------------------------------------

        def residuals(theta):
            pred = gaussian_fountain_model(
                theta,
                x_fit,
                y_fit,
            )

            base_resid = weights * (pred - z_fit)

            if not peak_is_valid:
                return base_resid

            # В этой параметризации Gaussian peak = (p, -p)
            p_model = theta[2]

            gaussian_peak_x = p_model
            gaussian_peak_y = -p_model

            dx = (gaussian_peak_x - hic_peak_x) / peak_scale_kb
            dy = (gaussian_peak_y - hic_peak_y) / peak_scale_kb

            # Масштабируем на sqrt(n), чтобы peak penalty был сопоставим
            # с суммой пиксельных residuals.
            peak_resid_scale = np.sqrt(max(len(z_fit), 1) * peak_weight)

            peak_resid = peak_resid_scale * np.array([dx, dy])

            return np.concatenate([base_resid, peak_resid])

        opt = least_squares(
            residuals,
            theta0,
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1" if robust else "linear",
            f_scale=0.1,
            max_nfev=20000,
        )

        if not opt.success:
            raise RuntimeError(opt.message)

        # ------------------------------------------------------
        # Final prediction and RMSE diagnostics
        # ------------------------------------------------------

        pred_final = gaussian_fountain_model(
            opt.x,
            x_fit,
            y_fit,
        )

        raw_resid = pred_final - z_fit
        weighted_resid = weights * raw_resid
        objective_resid = residuals(opt.x)

        rmse = float(np.sqrt(np.mean(raw_resid**2)))
        mae = float(np.mean(np.abs(raw_resid)))
        weighted_rmse = float(np.sqrt(np.mean(weighted_resid**2)))
        objective_rms_residual = float(np.sqrt(np.mean(objective_resid**2)))

        z_q05, z_q995 = np.nanquantile(z_fit, [0.05, 0.995])
        z_robust_range = float(max(z_q995 - z_q05, 1e-12))

        nrmse_robust = float(rmse / z_robust_range)
        nrmse_percent = float(100.0 * nrmse_robust)

        # ------------------------------------------------------
        # Final peak diagnostics
        # ------------------------------------------------------

        if peak_is_valid:
            p_final = float(opt.x[2])

            gaussian_peak_x = p_final
            gaussian_peak_y = -p_final

            peak_dist_kb = float(
                np.sqrt(
                    (gaussian_peak_x - hic_peak_x) ** 2
                    + (gaussian_peak_y - hic_peak_y) ** 2
                )
            )

            peak_penalty_value = float(
                peak_weight * (peak_dist_kb / peak_scale_kb) ** 2
            )

        else:
            gaussian_peak_x = np.nan
            gaussian_peak_y = np.nan
            peak_dist_kb = np.nan
            peak_penalty_value = 0.0

        diagnostics = {
            "n_points": len(z_fit),

            "n_tail_downweighted": int(
                np.count_nonzero(tail_scale < 1.0 - 1e-12)
            ),
            "mean_tail_scale": float(np.mean(tail_scale)),
            "min_tail_scale": float(np.min(tail_scale)),
            "mean_fit_weight": float(np.mean(weights)),
            "max_fit_weight": float(np.max(weights)),
            "major_coord_min": float(np.min(major_coord)),
            "major_coord_max": float(np.max(major_coord)),

            # RMSE diagnostics
            "rmse": rmse,
            "mae": mae,
            "weighted_rmse": weighted_rmse,
            "objective_rms_residual": objective_rms_residual,
            "z_robust_range": z_robust_range,
            "nrmse_robust": nrmse_robust,
            "nrmse_percent": nrmse_percent,

            # Peak diagnostics
            "peak_penalty_enabled": bool(peak_penalty),
            "peak_is_valid": bool(peak_is_valid),
            "peak_weight": float(peak_weight),
            "peak_scale_kb": float(peak_scale_kb),
            "peak_search_window_kb": float(peak_search_window_kb),
            "peak_quantile": float(peak_quantile),

            "hic_peak_x_kb": float(hic_peak_x)
            if np.isfinite(hic_peak_x)
            else np.nan,

            "hic_peak_y_kb": float(hic_peak_y)
            if np.isfinite(hic_peak_y)
            else np.nan,

            "hic_peak_n_pixels": int(hic_peak_n),

            "gaussian_peak_x_kb": float(gaussian_peak_x)
            if np.isfinite(gaussian_peak_x)
            else np.nan,

            "gaussian_peak_y_kb": float(gaussian_peak_y)
            if np.isfinite(gaussian_peak_y)
            else np.nan,

            "peak_dist_kb": float(peak_dist_kb)
            if np.isfinite(peak_dist_kb)
            else peak_dist_kb,

            "peak_penalty_value": float(peak_penalty_value)
            if np.isfinite(peak_penalty_value)
            else peak_penalty_value,
        }

        return opt, diagnostics

    # ==========================================================
    # 1. Грубый фит
    # ==========================================================

    opt_broad, diag_broad = run_fit(
        broad_mask,
        tail_center_p=None,
    )

    n_fit_broad = diag_broad["n_points"]
    p_broad = float(opt_broad.x[2])

    # ==========================================================
    # 2. Финальный фит
    # ==========================================================

    if two_step_peak_refit and near_peak_radius_kb is not None:
        U = (X + Y) / np.sqrt(2)
        V = (
            X - Y - 2 * p_broad
        ) / np.sqrt(2)

        near_peak_mask = base_mask.copy()
        near_peak_mask &= (
            U**2 + V**2
        ) <= near_peak_radius_kb**2

        opt, diag_final = run_fit(
            near_peak_mask,
            theta0=opt_broad.x,
            tail_center_p=p_broad,
        )

        fit_mode = "near_peak_refit"
        fit_peak_center_kb = p_broad

    elif downweight_major_tail:
        opt, diag_final = run_fit(
            broad_mask,
            theta0=opt_broad.x,
            tail_center_p=p_broad,
        )

        fit_mode = "broad_tail_refit"
        fit_peak_center_kb = p_broad

    else:
        opt = opt_broad
        diag_final = diag_broad

        fit_mode = "broad"
        fit_peak_center_kb = p_broad

    n_fit_pixels = diag_final["n_points"]

    a, b, p, A, C = opt.x

    if not np.all(np.isfinite([a, b, p, A, C])):
        raise ValueError("Fit returned non-finite parameters")

    eps = a / b

    # Старый показатель
    r_old_linear = (1 - eps) / (1 + eps)

    # Gaussian/Pearson-аналог
    r_gaussian = (
        1 - eps**2
    ) / (
        1 + eps**2
    )

    uncertainty = estimate_r_uncertainty(
        opt=opt,
        a=a,
        b=b,
        n_observations=n_fit_pixels,
    )

    return {
        "a_kb": a,
        "b_kb": b,
        "p_kb": p,
        "A": A,
        "C": C,
        "a_over_b": eps,

        "r": r_gaussian,
        "r_gaussian": r_gaussian,
        "r_old_linear": r_old_linear,

        "n_fit_pixels": n_fit_pixels,
        "n_fit_pixels_broad": n_fit_broad,

        "fit_mode": fit_mode,
        "near_peak_radius_kb": near_peak_radius_kb,
        "fit_peak_center_kb": fit_peak_center_kb,

        # Результаты предварительного широкого фита
        "a_kb_broad": float(opt_broad.x[0]),
        "b_kb_broad": float(opt_broad.x[1]),
        "p_kb_broad": float(opt_broad.x[2]),

        # RMSE diagnostics
        "rmse": diag_final["rmse"],
        "rmse_broad": diag_broad["rmse"],
        "mae": diag_final["mae"],
        "weighted_rmse": diag_final["weighted_rmse"],
        "objective_rms_residual": diag_final["objective_rms_residual"],
        "z_robust_range": diag_final["z_robust_range"],
        "nrmse_robust": diag_final["nrmse_robust"],
        "nrmse_percent": diag_final["nrmse_percent"],

        # Параметры подавления хвоста
        "downweight_major_tail": downweight_major_tail,
        "major_tail_side": major_tail_side,
        "major_tail_start_kb": major_tail_start_kb,
        "major_tail_transition_kb": major_tail_transition_kb,
        "major_tail_min_scale": major_tail_min_scale,

        # Диагностика весов
        "n_tail_downweighted": diag_final["n_tail_downweighted"],
        "mean_tail_scale": diag_final["mean_tail_scale"],
        "min_tail_scale": diag_final["min_tail_scale"],
        "mean_fit_weight": diag_final["mean_fit_weight"],
        "max_fit_weight": diag_final["max_fit_weight"],

        # Peak diagnostics
        "peak_penalty_enabled": diag_final["peak_penalty_enabled"],
        "peak_is_valid": diag_final["peak_is_valid"],
        "peak_weight": diag_final["peak_weight"],
        "peak_scale_kb": diag_final["peak_scale_kb"],
        "peak_search_window_kb": diag_final["peak_search_window_kb"],
        "peak_quantile": diag_final["peak_quantile"],

        "hic_peak_x_kb": diag_final["hic_peak_x_kb"],
        "hic_peak_y_kb": diag_final["hic_peak_y_kb"],
        "hic_peak_n_pixels": diag_final["hic_peak_n_pixels"],

        "gaussian_peak_x_kb": diag_final["gaussian_peak_x_kb"],
        "gaussian_peak_y_kb": diag_final["gaussian_peak_y_kb"],

        "peak_dist_kb": diag_final["peak_dist_kb"],
        "peak_penalty_value": diag_final["peak_penalty_value"],

        # Стандартные ошибки параметров
        "a_se_kb": uncertainty["a_se"],
        "b_se_kb": uncertainty["b_se"],
        "p_se_kb": uncertainty["p_se"],
        "A_se": uncertainty["A_se"],
        "C_se": uncertainty["C_se"],

        # Погрешность r
        "r_se": uncertainty["r_se"],
        "r_ci95_low": uncertainty["r_ci95_low"],
        "r_ci95_high": uncertainty["r_ci95_high"],

        # Диагностика оценки погрешности
        "corr_a_b": uncertainty["corr_a_b"],
        "degrees_of_freedom": uncertainty["degrees_of_freedom"],
        "residual_variance": uncertainty["residual_variance"],
        "jacobian_condition_number": uncertainty["jacobian_condition_number"],

        # Полная ковариационная матрица параметров
        "parameter_covariance": uncertainty["covariance"],

        "cost": opt.cost,
        "optimality": opt.optimality,
    }

def make_region_around_base(chrom, base_bp, flank, res, chromsizes):
    """
    Возвращает region, start, end.

    Если фонтан слишком близко к краю хромосомы, регион обрезается.
    Такие случаи потом можно либо фитить отдельно, либо пропускать.
    """
    chrom_len = int(chromsizes[chrom])
    start = (base_bp - flank) // res * res
    end = (base_bp + flank + res - 1) // res * res
    start = max(0, int(start))
    end = min(chrom_len, int(end))
    region = f'{chrom}:{start}-{end}'
    return (region, start, end)


def gaussian_fountain_model(theta, X, Y, model_kind='anti_diag_peak'):
    """
    theta = [a, b, p, A, C]

    a, b, p — основные fit-параметры.
    A — амплитуда.
    C — фон.

    model_kind="anti_diag_peak":
        пик около (p, -p), то есть фонтан на побочной диагонали.

        G = C + A * exp(-((x+y)^2/a^2 + (x-y-2p)^2/b^2))
    """
    a, b, p, A, C = theta
    if model_kind == 'anti_diag_peak':
        u = X + Y
        v = X - Y - 2 * p
    else:
        raise ValueError("model_kind должен быть 'anti_diag_peak' или 'as_written'.")
    return C + A * np.exp(-(u ** 2 / a ** 2 + v ** 2 / b ** 2))


import numpy as np
from scipy.optimize import least_squares


def fit_fountain_gaussian(
    Z,
    x_kb=None,
    y_kb=None,
    bin_kb=10,
    p0_kb=50,
    p_half_window_kb=40,
    model_kind='anti_diag_peak',
    upper_triangle=True,
    diag_exclusion_kb=0,
    fit_window_kb=180,
    positive_weight=True,
    positive_weight_strength=2.0,
    robust=True,

    # Новые параметры для подавления асимметричного хвоста
    downweight_major_tail=True,
    major_tail_side='positive',
    major_tail_start_kb=20,
    major_tail_transition_kb=35,
    major_tail_min_scale=0.25
):
    """
    Фитит фонтан 2D-гауссианом.

    Parameters
    ----------
    Z : 2D np.ndarray
        Матрица сигнала, например log2(observed / expected).

    x_kb, y_kb : 1D arrays or None
        Координаты столбцов и строк в kb относительно центра фонтана.

    bin_kb : float
        Размер бина в kb, если координаты не переданы явно.

    p0_kb : float
        Начальное предположение для положения p.

    p_half_window_kb : float
        Диапазон поиска p:
        [p0_kb - p_half_window_kb,
         p0_kb + p_half_window_kb].

    model_kind : str
        'anti_diag_peak' — пик около (p, -p).
        'as_written' — альтернативная параметризация.

    upper_triangle : bool
        Если True, используются только пиксели y < x.

    diag_exclusion_kb : float
        Ширина исключаемой области около главной диагонали.

    fit_window_kb : float
        Окно фита в повернутых координатах.

    positive_weight : bool
        Увеличивать ли вес положительно обогащённых пикселей.

    positive_weight_strength : float
        Сила увеличения веса положительных пикселей.

        При значении 2.0 множитель остатка изменяется от 1 до 3.
        Поскольку минимизируется квадрат остатка, максимальный вклад
        пикселя увеличивается до 9 раз.

        Если хвост имеет высокий положительный сигнал, разумно попробовать
        значения 0.5–1.0 или полностью отключить positive_weight.

    robust : bool
        Использовать ли soft_l1 loss.

    downweight_major_tail : bool
        Уменьшать ли вклад асимметричного хвоста вдоль major-axis.

    major_tail_side : {'positive', 'negative'}
        С какой стороны находится хвост.

        Для хвоста справа на твоём графике:
        major_tail_side='positive'.

    major_tail_start_kb : float
        Начиная с какого расстояния от центра major-профиля уменьшать вес.

    major_tail_transition_kb : float
        Длина области, на которой вес плавно уменьшается.

    major_tail_min_scale : float
        Минимальный множитель остатка для дальнего хвоста.

        Например:
        0.5 -> вклад в сумму квадратов уменьшается в 4 раза;
        0.25 -> вклад уменьшается в 16 раз.

    Returns
    -------
    result : dict
        a, b, p, A, C
        model_grid
        fit_mask
        fit_weights
        major_coord_fit
        tail_scale
        opt
        initial_opt
    """

    Z = np.asarray(Z, dtype=float)
    ny, nx = Z.shape

    if x_kb is None:
        x_kb = make_centered_coords(nx, bin_kb)
    else:
        x_kb = np.asarray(x_kb, dtype=float)

    if y_kb is None:
        y_kb = make_centered_coords(ny, bin_kb)
    else:
        y_kb = np.asarray(y_kb, dtype=float)

    X, Y = np.meshgrid(x_kb, y_kb)

    mask = np.isfinite(Z)

    if upper_triangle:
        mask &= Y < X - diag_exclusion_kb

    if model_kind == 'anti_diag_peak':
        u0 = X + Y
        v0 = X - Y - 2 * p0_kb

    elif model_kind == 'as_written':
        u0 = X - Y
        v0 = X + Y - 2 * p0_kb

    else:
        raise ValueError(
            "model_kind должен быть 'anti_diag_peak' или 'as_written'"
        )

    mask &= np.abs(u0) <= fit_window_kb
    mask &= np.abs(v0) <= fit_window_kb

    x_fit = X[mask]
    y_fit = Y[mask]
    z_fit = Z[mask]

    if len(z_fit) < 20:
        raise ValueError(
            'Слишком мало точек для фита. '
            'Попробуй увеличить fit_window_kb или проверить координаты.'
        )

    # Начальные значения
    C0 = np.nanmedian(z_fit)
    high = np.nanpercentile(z_fit, 95)
    A0 = max(high - C0, 0.001)

    a0 = fit_window_kb / 3
    b0 = fit_window_kb / 3

    theta0 = np.array(
        [a0, b0, p0_kb, A0, C0],
        dtype=float
    )

    coord_span = max(
        np.nanmax(x_kb) - np.nanmin(x_kb),
        np.nanmax(y_kb) - np.nanmin(y_kb)
    )

    z_min = np.nanmin(z_fit)
    z_max = np.nanmax(z_fit)
    z_range = max(z_max - z_min, 0.001)

    lower_bounds = np.array([
        bin_kb / 2,
        bin_kb / 2,
        p0_kb - p_half_window_kb,
        0.0,
        z_min - z_range
    ])

    upper_bounds = np.array([
        coord_span * 2,
        coord_span * 2,
        p0_kb + p_half_window_kb,
        z_range * 5,
        z_max + z_range
    ])

    # Базовый множитель остатка
    base_weights = np.ones_like(z_fit, dtype=float)

    if positive_weight:
        q70 = np.nanpercentile(z_fit, 70)
        q95 = np.nanpercentile(z_fit, 95)

        denom = max(q95 - q70, 1e-6)

        enrichment = np.clip(
            (z_fit - q70) / denom,
            0,
            1
        )

        base_weights *= (
            1.0
            + positive_weight_strength * enrichment
        )

    loss = 'soft_l1' if robust else 'linear'

    def residuals(theta, fit_weights):
        pred = gaussian_fountain_model(
            theta,
            x_fit,
            y_fit,
            model_kind=model_kind
        )

        return fit_weights * (pred - z_fit)

    # ----------------------------------------------------------
    # 1. Предварительный фит без специального подавления хвоста
    # ----------------------------------------------------------

    initial_opt = least_squares(
        lambda theta: residuals(theta, base_weights),
        theta0,
        bounds=(lower_bounds, upper_bounds),
        loss=loss,
        f_scale=0.1,
        max_nfev=20000
    )

    fit_weights = base_weights.copy()
    tail_scale = np.ones_like(z_fit, dtype=float)

    # ----------------------------------------------------------
    # 2. Определяем major-координату относительно найденного p
    # ----------------------------------------------------------

    p_initial = initial_opt.x[2]

    if model_kind == 'anti_diag_peak':
        # Координата, связанная с параметром b
        major_coord_fit = (
            x_fit - y_fit - 2 * p_initial
        ) / np.sqrt(2)

    else:
        # Для модели as_written параметру b соответствует X + Y - 2p
        major_coord_fit = (
            x_fit + y_fit - 2 * p_initial
        ) / np.sqrt(2)

    # ----------------------------------------------------------
    # 3. Плавно уменьшаем вклад только с одной стороны
    # ----------------------------------------------------------

    if downweight_major_tail:

        if major_tail_side == 'positive':
            signed_major_coord = major_coord_fit

        elif major_tail_side == 'negative':
            signed_major_coord = -major_coord_fit

        else:
            raise ValueError(
                "major_tail_side должен быть 'positive' или 'negative'"
            )

        if major_tail_transition_kb > 0:
            t = (
                signed_major_coord - major_tail_start_kb
            ) / major_tail_transition_kb

            t = np.clip(t, 0.0, 1.0)

            # Smoothstep:
            # плавный переход без скачка производной
            smooth_t = t**2 * (3.0 - 2.0 * t)

            tail_scale = (
                1.0
                - (1.0 - major_tail_min_scale) * smooth_t
            )

        else:
            # Жёсткое уменьшение веса после заданной границы
            tail_scale = np.where(
                signed_major_coord > major_tail_start_kb,
                major_tail_min_scale,
                1.0
            )

        fit_weights *= tail_scale

        # ------------------------------------------------------
        # 4. Повторный фит с уменьшенным весом хвоста
        # ------------------------------------------------------

        opt = least_squares(
            lambda theta: residuals(theta, fit_weights),
            initial_opt.x,
            bounds=(lower_bounds, upper_bounds),
            loss=loss,
            f_scale=0.1,
            max_nfev=20000
        )

    else:
        opt = initial_opt

    a, b, p, A, C = opt.x

    model_grid = gaussian_fountain_model(
        opt.x,
        X,
        Y,
        model_kind=model_kind
    )

    return {
        'a': a,
        'b': b,
        'p': p,
        'A': A,
        'C': C,

        'model_grid': model_grid,
        'fit_mask': mask,

        'x_kb': x_kb,
        'y_kb': y_kb,

        'opt': opt,
        'initial_opt': initial_opt,

        'model_kind': model_kind,

        # Для диагностики весов
        'fit_weights': fit_weights,
        'base_weights': base_weights,
        'tail_scale': tail_scale,
        'major_coord_fit': major_coord_fit
    }

def plot_fountain_fit(Z, fit_result, title=None):
    """
    Рисует исходную карту и контуры fitted Gaussian.
    """
    x_kb = fit_result['x_kb']
    y_kb = fit_result['y_kb']
    model_grid = fit_result['model_grid']
    X, Y = np.meshgrid(x_kb, y_kb)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, origin='upper', extent=[x_kb.min(), x_kb.max(), y_kb.max(), y_kb.min()], aspect='equal', interpolation='nearest', cmap='coolwarm', vmin=0, vmax=2)
    plt.colorbar(im, ax=ax, label='observed / expected')
    A = fit_result['A']
    C = fit_result['C']
    levels = C + A * np.array([0.25, 0.5, 0.75])
    ax.contour(X, Y, model_grid, levels=levels, linewidths=2, cmap='ocean')
    p = fit_result['p']
    ax.axline((0, 0), slope=1, linestyle='--', linewidth=1, label='main diagonal')
    ax.axline((0, 0), slope=-1, linestyle=':', linewidth=1, label='anti-diagonal')
    ax.set_xlabel('distance from fountain base, kb')
    ax.set_ylabel('distance from fountain base, kb')
    if title is None:
        title = f"a={fit_result['a']:.1f} kb, b={fit_result['b']:.1f} kb, p={fit_result['p']:.1f} kb"
    plt.tight_layout()
    return (fig, ax)


def compute_quadrant_moment_correlation(Z, bin_kb=10, max_arm_kb=200, min_arm_kb=0, z_transform='oe', background=None, weight_mode='positive_excess', weight_clip_quantile=0.99, min_pixels=20, min_total_weight=1e-09):
    """
    Считает model-free коэффициент корреляции фонтана
    по первой четверти в координатах плеч экструзии: l > 0, r > 0.

    Parameters
    ----------
    Z : 2D array
        Матрица observed/expected или log2(observed/expected)
        вокруг fountain base.

    bin_kb : float
        Размер бина в kb.

    max_arm_kb : float
        Максимальная длина плеча, которую включаем в анализ.
        Например, 200 kb.

    min_arm_kb : float
        Минимальная длина плеча. Обычно 0 или bin_kb.

    z_transform : {"oe", "log_oe"}
        Тип матрицы Z.

    background : float or None
        Фон, который вычитаем перед построением весов.
        Для oe по умолчанию 1.0.
        Для log_oe по умолчанию 0.0.

    weight_mode : {"positive_excess", "raw_positive"}
        positive_excess:
            w = max(Z - background, 0)
        raw_positive:
            w = max(Z, 0)

    weight_clip_quantile : float or None
        Обрезает слишком большие веса сверху, чтобы один пиксель
        не определял всю корреляцию.

    Returns
    -------
    dict
        q1_r_pearson:
            прямой weighted Pearson correlation между l и r.

        q1_axis_ratio:
            отношение малой и большой оси из моментной ковариационной матрицы.

        q1_r_axis_linear:
            аналог твоего старого r = (1 - a/b) / (1 + a/b),
            но рассчитанный из моментных осей без фита гауссианом.

        q1_r_axis_squared:
            корреляция, соответствующая эллиптическому Gaussian-моменту:
            (lambda_major - lambda_minor) / (lambda_major + lambda_minor).
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f'Z должна быть квадратной 2D-матрицей, получено {Z.shape}')
    n = Z.shape[0]
    coords_kb = make_centered_coords(n, bin_kb)
    X, Y = np.meshgrid(coords_kb, coords_kb)
    L = X
    R = -Y
    mask = np.isfinite(Z) & (L > min_arm_kb) & (R > min_arm_kb) & (L <= max_arm_kb) & (R <= max_arm_kb)
    l = L[mask].astype(float)
    r = R[mask].astype(float)
    z = Z[mask].astype(float)
    if len(z) < min_pixels:
        return {'q1_r_pearson': np.nan, 'q1_axis_ratio': np.nan, 'q1_r_axis_linear': np.nan, 'q1_r_axis_squared': np.nan, 'q1_n_pixels': len(z), 'q1_total_weight': np.nan, 'q1_status': 'too_few_pixels'}
    if background is None:
        if z_transform == 'oe':
            background = 1.0
        elif z_transform == 'log_oe':
            background = 0.0
        else:
            raise ValueError("z_transform должен быть 'oe' или 'log_oe'.")
    if weight_mode == 'positive_excess':
        w = np.clip(z - background, 0, None)
    elif weight_mode == 'raw_positive':
        w = np.clip(z, 0, None)
    else:
        raise ValueError("weight_mode должен быть 'positive_excess' или 'raw_positive'.")
    valid = np.isfinite(w) & (w > 0)
    l = l[valid]
    r = r[valid]
    w = w[valid]
    if len(w) < min_pixels:
        return {'q1_r_pearson': np.nan, 'q1_axis_ratio': np.nan, 'q1_r_axis_linear': np.nan, 'q1_r_axis_squared': np.nan, 'q1_n_pixels': len(w), 'q1_total_weight': float(np.nansum(w)) if len(w) else 0.0, 'q1_status': 'too_few_positive_pixels'}
    if weight_clip_quantile is not None:
        w_max = np.nanquantile(w, weight_clip_quantile)
        w = np.minimum(w, w_max)
    W = np.sum(w)
    if not np.isfinite(W) or W < min_total_weight:
        return {'q1_r_pearson': np.nan, 'q1_axis_ratio': np.nan, 'q1_r_axis_linear': np.nan, 'q1_r_axis_squared': np.nan, 'q1_n_pixels': len(w), 'q1_total_weight': float(W), 'q1_status': 'too_low_weight'}
    mean_l = np.sum(w * l) / W
    mean_r = np.sum(w * r) / W
    dl = l - mean_l
    dr = r - mean_r
    var_l = np.sum(w * dl ** 2) / W
    var_r = np.sum(w * dr ** 2) / W
    cov_lr = np.sum(w * dl * dr) / W
    if var_l <= 0 or var_r <= 0:
        r_pearson = np.nan
    else:
        r_pearson = cov_lr / np.sqrt(var_l * var_r)
    cov_mat = np.array([[var_l, cov_lr], [cov_lr, var_r]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals = np.sort(eigvals)
    lambda_minor = eigvals[0]
    lambda_major = eigvals[1]
    if lambda_minor <= 0 or lambda_major <= 0:
        axis_ratio = np.nan
        r_axis_linear = np.nan
        r_axis_squared = np.nan
    else:
        axis_ratio = np.sqrt(lambda_minor / lambda_major)
        r_axis_linear = (1 - axis_ratio) / (1 + axis_ratio)
        r_axis_squared = (lambda_major - lambda_minor) / (lambda_major + lambda_minor)
    return {'q1_r_pearson': float(r_pearson), 'q1_axis_ratio': float(axis_ratio), 'q1_r_axis_linear': float(r_axis_linear), 'q1_r_axis_squared': float(r_axis_squared), 'q1_mean_l_kb': float(mean_l), 'q1_mean_r_kb': float(mean_r), 'q1_var_l_kb2': float(var_l), 'q1_var_r_kb2': float(var_r), 'q1_cov_lr_kb2': float(cov_lr), 'q1_n_pixels': int(len(w)), 'q1_total_weight': float(W), 'q1_status': 'ok'}


def process_fountains_batch(cool_path, fountains_csv, output_csv='fountain_fit_results.csv', failed_csv='fountain_fit_failed.csv', res=10000, flank=200000, p0_kb=50, p_half_window_kb=40, fit_window_kb=180, diag_exclusion_kb=10, z_transform='oe', expected_nproc=1):
    """
    z_transform:
        "oe"     — фитить observed / expected
        "log_oe" — фитить log2(observed / expected)

    В итоговую выборку попадают только успешно зафиченные фонтаны.
    Неуспешные сохраняются отдельно в failed_csv.
    """
    clr = cooler.Cooler(f'{cool_path}::resolutions/{res}')
    fountains = load_fountains_csv(fountains_csv)
    fountains = harmonize_fountain_chroms_to_cooler(fountains, clr)
    chroms = sorted(fountains['chrom'].unique())
    expected_by_chrom, expected_raw, exp_col = compute_expected_by_chrom(clr, chroms=chroms, smooth=False, aggregate_smoothed=False, nproc=expected_nproc, chunksize=1000000)
    exp_cache = ExpectedMatrixCache(expected_by_chrom)
    fit_cache = FountainFitCache(bin_kb=res / 1000, p0_kb=p0_kb, p_half_window_kb=p_half_window_kb, diag_exclusion_kb=diag_exclusion_kb, fit_window_kb=fit_window_kb)
    results = []
    failed = []
    matrix_selector = clr.matrix(balance=True)
    for _, row in tqdm(fountains.iterrows(), total=len(fountains), desc='Fitting fountains'):
        fountain_index = row['Fountain index']
        chrom = row['chrom']
        base_bp = int(row['base_bp'])
        try:
            region, region_start, region_end = make_region_around_base(chrom=chrom, base_bp=base_bp, flank=flank, res=res, chromsizes=clr.chromsizes)
            mat = matrix_selector.fetch(region)
            mat = np.asarray(mat, dtype=float)
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(f'Non-square matrix for {region}: {mat.shape}')
            n = mat.shape[0]
            expected_n = int(2 * flank / res) + 1
            if n != expected_n:
                raise ValueError(f'Window near chromosome edge or unexpected size: n={n}, expected={expected_n}, region={region}')
            exp_mat = exp_cache.get_exp_mat(chrom, n)
            oe = mat / exp_mat
            oe[~np.isfinite(oe)] = np.nan
            np.fill_diagonal(oe, np.nan)
            if z_transform == 'oe':
                Z = oe
            elif z_transform == 'log_oe':
                Z = np.log2(oe)
                Z[~np.isfinite(Z)] = np.nan
                np.fill_diagonal(Z, np.nan)
            else:
                raise ValueError("z_transform должен быть 'oe' или 'log_oe'.")
            q1_corr = compute_quadrant_moment_correlation(Z, bin_kb=res / 1000, max_arm_kb=flank / 1000, min_arm_kb=0, z_transform=z_transform, background=None, weight_mode='positive_excess', weight_clip_quantile=0.99, min_pixels=20)
            fit = fit_one_fountain_matrix(Z, fit_cache=fit_cache, p0_kb=p0_kb, p_half_window_kb=p_half_window_kb, positive_weight=True, robust=True, min_points=20)
            result_row = {'Fountain index': fountain_index, 'chrom': chrom, 'start': int(row['start']), 'end': int(row['end']), 'base_bp': base_bp, 'region': region, 'Fountain Score': row['Fountain Score'], 'expected_column': exp_col, 'z_transform': z_transform, **fit, **q1_corr}
            results.append(result_row)
        except Exception as e:
            failed.append({'Fountain index': fountain_index, 'chrom': chrom, 'start': row.get('start', np.nan), 'end': row.get('end', np.nan), 'base_bp': base_bp, 'Fountain Score': row.get('Fountain Score', np.nan), 'error': repr(e)})
            continue
    results_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)
    results_df.to_csv(output_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)
    return (results_df, failed_df, expected_raw)


def extract_fountain_Z(row, clr, matrix_selector, exp_cache, flank=200000, res=5000, z_transform='oe', require_full_window=True, base_shift_bp=-6000):
    """
    Достаёт окно вокруг одного фонтана и возвращает Z:
        Z = observed / expected
    или
        Z = log2(observed / expected)

    row должен содержать:
        chrom, start, end
    либо уже готовый base_bp.
    """
    chrom = row['chrom']
    if 'base_bp' in row:
        base_bp = int(row['base_bp']) + int(base_shift_bp)
    else:
        base_bp = int((int(row['start']) + int(row['end'])) // 2) + int(base_shift_bp)
    chrom_len = int(clr.chromsizes[chrom])
    start = (base_bp - flank) // res * res
    end = (base_bp + flank + res - 1) // res * res
    start = max(0, int(start))
    end = min(chrom_len, int(end))
    region = f'{chrom}:{start}-{end}'
    mat = matrix_selector.fetch(region)
    mat = np.asarray(mat, dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f'Non-square matrix for {region}: {mat.shape}')
    n = mat.shape[0]
    expected_n = int(2 * flank / res) + 1
    if require_full_window and n != expected_n:
        raise ValueError(f'Window has unexpected size: n={n}, expected={expected_n}, region={region}')
    exp_mat = exp_cache.get_exp_mat(chrom, n)
    oe = mat / exp_mat
    oe[~np.isfinite(oe)] = np.nan
    np.fill_diagonal(oe, np.nan)
    if z_transform == 'oe':
        Z = oe
    elif z_transform == 'log_oe':
        Z = np.log2(oe)
        Z[~np.isfinite(Z)] = np.nan
        np.fill_diagonal(Z, np.nan)
    else:
        raise ValueError("z_transform должен быть 'oe' или 'log_oe'.")
    meta = {'chrom': chrom, 'base_bp': base_bp, 'region': region, 'region_start': start, 'region_end': end, 'n': n}
    return (Z, meta)


def chrom_key(chrom):
    """
    Универсальный ключ для сопоставления названий хромосом:
        chr1, Chr1, CHR1, 1 -> 1
        chr10, Chr10 -> 10
        chrM, ChrM, MT -> m / mt
    """
    s = str(chrom).strip()
    s = re.sub('^chr', '', s, flags=re.IGNORECASE)
    s = s.strip()
    return s.lower()


def make_chrom_name_map(input_chroms, target_chroms):
    """
    Строит map:
        название из fountains.csv -> название в cooler/mcool.

    Например:
        chr1 -> Chr1
        1    -> Chr1
        Chr1 -> Chr1
    """
    target_by_key = {}
    for chrom in target_chroms:
        key = chrom_key(chrom)
        if key in target_by_key:
            raise ValueError(f'Неоднозначное сопоставление для ключа {key}: {target_by_key[key]} и {chrom}')
        target_by_key[key] = chrom
    chrom_map = {}
    missing = []
    for chrom in input_chroms:
        key = chrom_key(chrom)
        if key in target_by_key:
            chrom_map[chrom] = target_by_key[key]
        else:
            missing.append(chrom)
    if missing:
        raise ValueError(f'Не удалось сопоставить эти хромосомы из fountains.csv с хромосомами в mcool: {missing}\n\nХромосомы в mcool: {list(target_chroms)[:30]}')
    return chrom_map


def harmonize_fountain_chroms_to_cooler(fountains, clr, chrom_col='chrom'):
    """
    Приводит fountains[chrom_col] к названиям хромосом в clr.

    Добавляет:
        chrom_original — исходное название
        chrom          — название как в mcool
    """
    out = fountains.copy()
    cooler_chroms = list(clr.chromnames)
    chrom_map = make_chrom_name_map(input_chroms=out[chrom_col].astype(str).unique(), target_chroms=cooler_chroms)
    out['chrom_original'] = out[chrom_col].astype(str)
    out[chrom_col] = out[chrom_col].astype(str).map(chrom_map)
    return out


def build_weighted_aggregate_fountain(cool_path, fountains_csv, flank=300000, res=10000, weight_col='Fountain Score', z_transform='oe', expected_nproc=1, min_weight=0, require_full_window=True, base_shift_bp=-6000):
    """
    Строит агрегированный фонтан:

        Z_agg[x, y] = sum_i(score_i * Z_i[x, y]) / sum_i(score_i)

    Если отдельный фонтан не удалось извлечь, он пропускается.

    Возвращает:
        aggregate_Z
        aggregate_info
        used_df
        failed_df
        expected_raw
    """
    clr = cooler.Cooler(f'{cool_path}::resolutions/{res}')
    fountains = load_fountains_csv(fountains_csv)
    fountains = harmonize_fountain_chroms_to_cooler(fountains, clr)
    if 'base_bp' not in fountains.columns:
        fountains['base_bp'] = (fountains['start'].astype(int) + fountains['end'].astype(int)) // 2
    chroms = sorted(fountains['chrom'].unique())
    expected_by_chrom, expected_raw, exp_col = compute_expected_by_chrom(clr, chroms=chroms, smooth=False, aggregate_smoothed=False, nproc=expected_nproc, chunksize=1000000)
    exp_cache = ExpectedMatrixCache(expected_by_chrom)
    matrix_selector = clr.matrix(balance=True)
    sum_weighted_Z = None
    sum_weights = None
    n_contributors = None
    used_rows = []
    failed_rows = []
    expected_n = int(2 * flank / res) + 1
    for _, row in tqdm(fountains.iterrows(), total=len(fountains), desc='Building aggregate fountain'):
        fountain_index = row['Fountain index']
        try:
            weight = 1 #float(row[weight_col])
            if not np.isfinite(weight):
                raise ValueError(f'Non-finite weight: {weight}')
            if weight <= min_weight:
                raise ValueError(f'Weight <= min_weight: {weight}')
            Z, meta = extract_fountain_Z(row=row, clr=clr, matrix_selector=matrix_selector, exp_cache=exp_cache, flank=flank, res=res, z_transform=z_transform, require_full_window=require_full_window, base_shift_bp=base_shift_bp)
            if Z.shape != (expected_n, expected_n):
                raise ValueError(f'Unexpected Z shape: {Z.shape}')
            if sum_weighted_Z is None:
                sum_weighted_Z = np.zeros_like(Z, dtype=float)
                sum_weights = np.zeros_like(Z, dtype=float)
                n_contributors = np.zeros_like(Z, dtype=int)
            valid = np.isfinite(Z)
            if valid.sum() == 0:
                raise ValueError('No finite pixels in Z')
            sum_weighted_Z[valid] += weight * Z[valid]
            sum_weights[valid] += weight
            n_contributors[valid] += 1
            used_rows.append({'Fountain index': fountain_index, 'chrom': row['chrom'], 'start': int(row['start']), 'end': int(row['end']), 'base_bp': int(row['base_bp']), 'Fountain Score': row[weight_col], 'weight': weight, 'region': meta['region']})
        except Exception as e:
            failed_rows.append({'Fountain index': fountain_index, 'chrom': row.get('chrom', np.nan), 'start': row.get('start', np.nan), 'end': row.get('end', np.nan), 'Fountain Score': row.get(weight_col, np.nan), 'error': repr(e)})
            continue
    if sum_weighted_Z is None:
        raise RuntimeError('Не удалось добавить ни одного фонтана в aggregate.')
    aggregate_Z = np.full_like(sum_weighted_Z, np.nan, dtype=float)
    valid_agg = sum_weights > 0
    aggregate_Z[valid_agg] = sum_weighted_Z[valid_agg] / sum_weights[valid_agg]
    np.fill_diagonal(aggregate_Z, np.nan)
    used_df = pd.DataFrame(used_rows)
    failed_df = pd.DataFrame(failed_rows)
    aggregate_info = {'n_used': len(used_df), 'n_failed': len(failed_df), 'z_transform': z_transform, 'weight_col': weight_col, 'expected_column': exp_col, 'flank': flank, 'res': res, 'n_pixels': expected_n, 'sum_weights_total': float(np.nansum(used_df['weight'])) if len(used_df) else 0, 'n_contributors': n_contributors, 'sum_weights': sum_weights}
    return (aggregate_Z, aggregate_info, used_df, failed_df, expected_raw)




def estimate_r_uncertainty(
    opt,
    a,
    b,
    n_observations,
    confidence_z=1.96,
):
    """
    Оценивает локальную статистическую погрешность параметров фита
    и коэффициента

        r = (b^2 - a^2) / (b^2 + a^2)

    по якобиану scipy.optimize.least_squares.

    Parameters
    ----------
    opt : scipy.optimize.OptimizeResult
        Результат least_squares.

    a, b : float
        Параметры ширины гауссианы.

    n_observations : int
        Число пикселей, использованных в финальном фите.

    confidence_z : float
        Коэффициент для доверительного интервала.
        1.96 соответствует приблизительно 95% ДИ.

    Returns
    -------
    dict
        Ковариационная матрица, стандартные ошибки
        и доверительный интервал для r.
    """

    jac = np.asarray(opt.jac, dtype=float)

    if jac.ndim != 2:
        raise ValueError("Якобиан opt.jac должен быть двумерным")

    n_parameters = jac.shape[1]
    dof = int(n_observations - n_parameters)

    if dof <= 0:
        raise ValueError(
            "Недостаточно степеней свободы для оценки погрешности: "
            f"n_observations={n_observations}, "
            f"n_parameters={n_parameters}"
        )

    # Для linear loss:
    #     2 * cost = сумма квадратов остатка.
    #
    # Для soft_l1 это локальная приближенная оценка масштаба ошибки.
    residual_variance = 2.0 * float(opt.cost) / dof

    jtj = jac.T @ jac

    # pinv устойчивее обычного inv при близкой вырожденности
    covariance = (
        residual_variance
        * np.linalg.pinv(jtj, rcond=1e-12)
    )

    variances = np.diag(covariance)
    parameter_se = np.sqrt(
        np.maximum(variances, 0.0)
    )

    # Порядок параметров:
    # theta = [a, b, p, A, C]
    a_se = float(parameter_se[0])
    b_se = float(parameter_se[1])

    denominator = a**2 + b**2

    if denominator <= 0:
        raise ValueError(
            "Невозможно вычислить погрешность r: a^2 + b^2 <= 0"
        )

    # Производные:
    #
    # dr/da = -4*a*b^2 / (a^2+b^2)^2
    # dr/db =  4*a^2*b / (a^2+b^2)^2

    dr_da = (
        -4.0 * a * b**2
        / denominator**2
    )

    dr_db = (
        4.0 * a**2 * b
        / denominator**2
    )

    gradient_r = np.zeros(n_parameters, dtype=float)
    gradient_r[0] = dr_da
    gradient_r[1] = dr_db

    r_variance = float(
        gradient_r
        @ covariance
        @ gradient_r
    )

    r_variance = max(r_variance, 0.0)
    r_se = np.sqrt(r_variance)

    r_value = (
        b**2 - a**2
    ) / (
        b**2 + a**2
    )

    r_ci_low = max(
        -1.0,
        r_value - confidence_z * r_se,
    )

    r_ci_high = min(
        1.0,
        r_value + confidence_z * r_se,
    )

    # Корреляция ошибок a и b
    if a_se > 0 and b_se > 0:
        corr_ab = (
            covariance[0, 1]
            / (a_se * b_se)
        )
    else:
        corr_ab = np.nan

    condition_number = float(
        np.linalg.cond(jtj)
    )

    return {
        "covariance": covariance,
        "parameter_se": parameter_se,

        "a_se": a_se,
        "b_se": b_se,
        "p_se": float(parameter_se[2]),
        "A_se": float(parameter_se[3]),
        "C_se": float(parameter_se[4]),

        "corr_a_b": float(corr_ab),

        "r_value": float(r_value),
        "r_se": float(r_se),
        "r_ci95_low": float(r_ci_low),
        "r_ci95_high": float(r_ci_high),

        "degrees_of_freedom": dof,
        "residual_variance": residual_variance,
        "jacobian_condition_number": condition_number,
    }


def fit_aggregate_fountain_old(
    aggregate_Z,
    res=10_000,
    p0_kb=50,
    p_half_window_kb=40,
    fit_window_kb=180,
    diag_exclusion_kb=10,

    # Параметры основного фита
    positive_weight=True,
    positive_weight_strength=0.75,
    robust=True,
    min_points=20,

    # Двухэтапный локальный фит
    two_step_peak_refit=True,
    near_peak_radius_kb=90,

    # Уменьшение влияния асимметричного major-хвоста
    downweight_major_tail=True,
    major_tail_side="positive",
    major_tail_start_kb=20,
    major_tail_transition_kb=40,
    major_tail_min_scale=0.50,

    # Peak-aware penalty
    peak_penalty=True,
    peak_weight=20.0,
    peak_scale_kb=10.0,
    peak_search_window_kb=70,
    peak_quantile=0.80,
    peak_center_kb=None,
    peak_min_points=3,
):
    """
    Фитит агрегированный фонтан 2D-гауссианом.

    Добавлен peak-aware penalty:
    модель дополнительно штрафуется за несовпадение центра масс
    верхушки Gaussian-пика с центром масс верхушки Hi-C-пика.
    """

    bin_kb = res / 1000.0

    fit_cache = FountainFitCache(
        bin_kb=bin_kb,
        p0_kb=p0_kb,
        p_half_window_kb=p_half_window_kb,
        diag_exclusion_kb=diag_exclusion_kb,
        fit_window_kb=fit_window_kb,
    )

    fit = fit_one_fountain_matrix(
        aggregate_Z,
        fit_cache=fit_cache,

        p0_kb=p0_kb,
        p_half_window_kb=p_half_window_kb,

        positive_weight=positive_weight,
        positive_weight_strength=positive_weight_strength,
        robust=robust,
        min_points=min_points,

        two_step_peak_refit=two_step_peak_refit,
        near_peak_radius_kb=near_peak_radius_kb,

        downweight_major_tail=downweight_major_tail,
        major_tail_side=major_tail_side,
        major_tail_start_kb=major_tail_start_kb,
        major_tail_transition_kb=major_tail_transition_kb,
        major_tail_min_scale=major_tail_min_scale,

        peak_penalty=peak_penalty,
        peak_weight=peak_weight,
        peak_scale_kb=peak_scale_kb,
        peak_search_window_kb=peak_search_window_kb,
        peak_quantile=peak_quantile,
        peak_center_kb=peak_center_kb,
        peak_min_points=peak_min_points,
    )

    return fit, fit_cache
    
    
import numpy as np
from scipy.ndimage import gaussian_filter


def _get_fit_param(fit, *names, default=None):
    for name in names:
        if name in fit:
            return fit[name]
    return default


def _nan_smooth(A, sigma=1.0):
    A = np.asarray(A, dtype=float)

    if sigma is None or sigma <= 0:
        return A.copy()

    finite = np.isfinite(A)

    if not finite.any():
        return A.copy()

    fill_value = np.nanmedian(A[finite])
    A_fill = np.where(finite, A, fill_value)

    return gaussian_filter(A_fill, sigma=sigma)


def _robust01(A, mask=None, q_low=0.05, q_high=0.995):
    A = np.asarray(A, dtype=float)

    if mask is None:
        mask = np.isfinite(A)
    else:
        mask = mask & np.isfinite(A)

    if mask.sum() < 5:
        return np.full_like(A, np.nan, dtype=float), np.nan, np.nan

    lo = np.nanquantile(A[mask], q_low)
    hi = np.nanquantile(A[mask], q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(A, np.nan, dtype=float), lo, hi

    A01 = (A - lo) / (hi - lo)
    A01 = np.clip(A01, 0.0, 1.0)

    return A01, lo, hi


def _gaussian_model_from_fit(fit, X, Y):
    A = float(_get_fit_param(fit, "A", "amp", "amplitude", default=1.0))
    C = float(_get_fit_param(fit, "C", "background", "offset", default=0.0))

    a = float(_get_fit_param(fit, "a_kb", "a", default=np.nan))
    b = float(_get_fit_param(fit, "b_kb", "b", default=np.nan))
    p = float(_get_fit_param(fit, "p_kb", "p", default=np.nan))

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(p):
        raise ValueError("Cannot build Gaussian model: fit must contain a_kb, b_kb and p_kb.")

    a = max(a, 1e-9)
    b = max(b, 1e-9)

    G = C + A * np.exp(
        -(
            ((X + Y) ** 2) / (a ** 2)
            + ((X - Y - 2.0 * p) ** 2) / (b ** 2)
        )
    )

    return G


def _peak_center_of_mass(F, X, Y, mask, peak_quantile=0.80, min_points=3):
    F = np.asarray(F, dtype=float)

    mask = mask & np.isfinite(F) & np.isfinite(X) & np.isfinite(Y)

    if mask.sum() < min_points:
        return np.nan, np.nan, 0

    values = F[mask]
    x = X[mask]
    y = Y[mask]

    thr = np.nanquantile(values, peak_quantile)
    top = values >= thr

    if top.sum() < min_points:
        return np.nan, np.nan, int(top.sum())

    weights = values[top] - thr

    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)

    cx = np.average(x[top], weights=weights)
    cy = np.average(y[top], weights=weights)

    return float(cx), float(cy), int(top.sum())


def _compute_gaussian_peak_aware_score(
    aggregate_Z,
    fit,
    res,
    p0_kb=50,
    fit_window_kb=180,
    diag_exclusion_kb=10,

    peak_search_center=None,
    peak_search_window_kb=60,
    peak_quantile=0.80,
    peak_weight=3.0,
    peak_scale_kb=20.0,

    smooth_sigma=1.0,
    norm_q_low=0.05,
    norm_q_high=0.995,
):
    """
    Считает score для Gaussian fit:

        score = normalized_rmse + peak_penalty

    peak_penalty считается по расстоянию между центрами масс верхушки
    Hi-C пика и Gaussian пика.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    n = Z.shape[0]
    bin_kb = res / 1000.0

    coords_kb = (np.arange(n) - n // 2) * bin_kb
    X, Y = np.meshgrid(coords_kb, coords_kb)

    G = _gaussian_model_from_fit(fit, X, Y)

    Z_smooth = _nan_smooth(Z, sigma=smooth_sigma)
    G_smooth = _nan_smooth(G, sigma=smooth_sigma)

    fit_mask = (
        (X >= 0) &
        (Y <= 0) &
        (np.abs(X) <= fit_window_kb) &
        (np.abs(Y) <= fit_window_kb) &
        (np.abs(X - Y) >= diag_exclusion_kb) &
        np.isfinite(Z_smooth) &
        np.isfinite(G_smooth)
    )

    Z01, z_lo, z_hi = _robust01(
        Z_smooth,
        mask=fit_mask,
        q_low=norm_q_low,
        q_high=norm_q_high,
    )

    G01, g_lo, g_hi = _robust01(
        G_smooth,
        mask=fit_mask,
        q_low=norm_q_low,
        q_high=norm_q_high,
    )

    valid = fit_mask & np.isfinite(Z01) & np.isfinite(G01)

    if valid.sum() < 5:
        normalized_rmse = np.inf
    else:
        normalized_rmse = float(
            np.sqrt(np.mean((Z01[valid] - G01[valid]) ** 2))
        )

    if peak_search_center is None:
        cx, cy = p0_kb, -p0_kb
    else:
        cx, cy = peak_search_center

    peak_mask = (
        (np.abs(X - cx) <= peak_search_window_kb) &
        (np.abs(Y - cy) <= peak_search_window_kb) &
        valid
    )

    z_peak_x, z_peak_y, z_n = _peak_center_of_mass(
        Z01,
        X,
        Y,
        peak_mask,
        peak_quantile=peak_quantile,
    )

    g_peak_x, g_peak_y, g_n = _peak_center_of_mass(
        G01,
        X,
        Y,
        peak_mask,
        peak_quantile=peak_quantile,
    )

    if not np.isfinite(z_peak_x) or not np.isfinite(g_peak_x):
        peak_dist_kb = np.inf
        peak_penalty = np.inf
        fit_score = np.inf
    else:
        peak_dist_kb = float(
            np.sqrt(
                (g_peak_x - z_peak_x) ** 2
                + (g_peak_y - z_peak_y) ** 2
            )
        )

        peak_penalty = float(
            peak_weight * (peak_dist_kb / peak_scale_kb) ** 2
        )

        fit_score = float(normalized_rmse + peak_penalty)

    return {
        "fit_score": fit_score,
        "normalized_rmse": normalized_rmse,
        "peak_dist_kb": peak_dist_kb,
        "peak_penalty": peak_penalty,

        "hic_peak_x_kb": z_peak_x,
        "hic_peak_y_kb": z_peak_y,
        "gaussian_peak_x_kb": g_peak_x,
        "gaussian_peak_y_kb": g_peak_y,

        "hic_peak_n_pixels": z_n,
        "gaussian_peak_n_pixels": g_n,

        "z_norm_lo": z_lo,
        "z_norm_hi": z_hi,
        "g_norm_lo": g_lo,
        "g_norm_hi": g_hi,
    }

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_aggregate_fountain_fit(
    aggregate_Z,
    fit,
    fit_cache,
    image_vmin=None,
    image_vmax=None,
    cmap="coolwarm",
    figsize=(6, 6),
    output=None,
    dpi=300,

    # unified contour normalization
    normalize_contours=True,
    norm_hic_levels=(0.35, 0.50, 0.70, 0.85),
    norm_gaussian_levels=(0.35, 0.50, 0.70, 0.85),
    contour_norm_orientation="upper",
    contour_norm_window_kb=None,
    contour_norm_q_low=0.05,
    contour_norm_q_high=0.995,

    # Hi-C contours
    show_hic_contours=False,
    hic_contour_levels=None,
    hic_contour_quantiles=(0.70, 0.80, 0.90, 0.96),
    hic_contour_smooth_sigma=1.0,
    hic_contour_color="black",
    hic_contour_linewidth=1.2,
    hic_contour_linestyle="-",
    label_hic_contours=False,

    # Gaussian fit contours
    show_fit_contours=True,
    fit_contour_fractions=(0.35, 0.50, 0.70, 0.85),
    fit_contour_levels=None,
    fit_contour_color="white",
    fit_contour_linewidth=1.5,
    fit_contour_linestyle="--",
    label_fit_contours=False,
    show_fit_contours_mirror=True,   # <--- добавить

    # guides
    show_diagonals=True,
    title=None,
    colorbar_label="Hi-C intensity",
):
    """
    Рисует агрегированный фонтан и контуры.

    Если normalize_contours=True, то Hi-C и Gaussian fit
    сначала независимо нормируются в 0..1 по одной и той же маске,
    а затем рисуются с common_contour_levels.
    """

    def _valid_levels(F, levels):
        levels = np.asarray(levels, dtype=float)
        fmin = np.nanmin(F)
        fmax = np.nanmax(F)
        return np.sort(levels[(levels > fmin) & (levels < fmax)])

    def _make_norm_mask(X, Y, orientation="upper", window_kb=None):
        if orientation == "upper":
            mask = (X >= 0) & (Y <= 0)
        elif orientation == "lower":
            mask = (X <= 0) & (Y >= 0)
        elif orientation == "both":
            mask = ((X >= 0) & (Y <= 0)) | ((X <= 0) & (Y >= 0))
        elif orientation is None:
            mask = np.ones_like(X, dtype=bool)
        else:
            raise ValueError(
                "contour_norm_orientation должен быть 'upper', 'lower', 'both' или None."
            )

        if window_kb is not None:
            mask = (
                mask
                & (np.abs(X) <= window_kb)
                & (np.abs(Y) <= window_kb)
            )

        return mask

    def _normalize01(F, mask=None, q_low=0.05, q_high=0.995):
        F = np.asarray(F, dtype=float)

        if mask is None:
            vals = F[np.isfinite(F)]
        else:
            vals = F[np.isfinite(F) & mask]

        if len(vals) == 0:
            return np.full_like(F, np.nan)

        lo = np.nanquantile(vals, q_low)
        hi = np.nanquantile(vals, q_high)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.full_like(F, np.nan)

        out = (F - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)

        return out

    # ------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------

    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]

    ctx = fit_cache.get(n)

    if "X" in ctx and "Y" in ctx:
        X = ctx["X"]
        Y = ctx["Y"]
        x_kb = ctx.get("x_kb", X[0, :])
        y_kb = ctx.get("y_kb", Y[:, 0])
    else:
        x_kb = ctx["x_kb"]
        y_kb = ctx["y_kb"]
        X, Y = np.meshgrid(x_kb, y_kb)

    extent = [
        x_kb.min(),
        x_kb.max(),
        y_kb.max(),
        y_kb.min(),
    ]

    # ------------------------------------------------------------
    # Base heatmap
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        Z,
        origin="upper",
        extent=extent,
        cmap=cmap,
        vmin=image_vmin,
        vmax=image_vmax,
        interpolation="nearest",
        aspect="equal",
        zorder=0,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label(colorbar_label, fontsize=11)

    # ------------------------------------------------------------
    # Prepare Hi-C contour field
    # ------------------------------------------------------------

    Z_hic = Z.copy()
    finite = np.isfinite(Z_hic)

    if np.any(finite):
        med = np.nanmedian(Z_hic[finite])
    else:
        med = 0.0

    Z_hic[~finite] = med

    if hic_contour_smooth_sigma is not None and hic_contour_smooth_sigma > 0:
        Z_hic = gaussian_filter(Z_hic, sigma=hic_contour_smooth_sigma)

    # ------------------------------------------------------------
    # Prepare Gaussian model
    # ------------------------------------------------------------

    gaussian_model = None

    if fit is not None:
        gaussian_model = gaussian_fountain_from_fit_on_grid(
            gaussian_fit=fit,
            X=X,
            Y=Y,
        )

    # ------------------------------------------------------------
    # Unified contour normalization
    # ------------------------------------------------------------

    if normalize_contours:
        norm_mask = _make_norm_mask(
            X,
            Y,
            orientation=contour_norm_orientation,
            window_kb=contour_norm_window_kb,
        )

        Z_hic_for_contour = _normalize01(
            Z_hic,
            mask=norm_mask,
            q_low=contour_norm_q_low,
            q_high=contour_norm_q_high,
        )

        if gaussian_model is not None:
            gaussian_for_contour = _normalize01(
                gaussian_model,
                mask=norm_mask,
                q_low=contour_norm_q_low,
                q_high=contour_norm_q_high,
            )
        else:
            gaussian_for_contour = None

        hic_levels_use = _valid_levels(
            Z_hic_for_contour,
            norm_hic_levels,
        )

        fit_levels_use = (
            _valid_levels(gaussian_for_contour, norm_gaussian_levels)
            if gaussian_for_contour is not None
            else np.array([])
        )

    else:
        Z_hic_for_contour = Z_hic
        gaussian_for_contour = gaussian_model

        if hic_contour_levels is None:
            vals = Z_hic[np.isfinite(Z_hic)]
            hic_levels_use = np.quantile(vals, hic_contour_quantiles)
        else:
            hic_levels_use = np.asarray(hic_contour_levels, dtype=float)

        hic_levels_use = _valid_levels(Z_hic_for_contour, hic_levels_use)

        if gaussian_model is not None:
            if fit_contour_levels is not None:
                fit_levels_use = np.asarray(fit_contour_levels, dtype=float)
            else:
                A = float(
                    fit.get(
                        "A",
                        np.nanmax(gaussian_model) - np.nanmin(gaussian_model),
                    )
                )
                C = float(
                    fit.get(
                        "C",
                        np.nanmin(gaussian_model),
                    )
                )
                fit_levels_use = C + A * np.asarray(
                    fit_contour_fractions,
                    dtype=float,
                )

            fit_levels_use = _valid_levels(gaussian_model, fit_levels_use)
        else:
            fit_levels_use = np.array([])

    # ------------------------------------------------------------
    # Gaussian contours
    # ------------------------------------------------------------

    fit_cs = None
    fit_cs_mirror = None

    if (
        show_fit_contours
        and gaussian_for_contour is not None
        and len(fit_levels_use) > 0
    ):
        # основной гауссовский фит
        fit_cs = ax.contour(
            X,
            Y,
            gaussian_for_contour,
            levels=fit_levels_use,
            colors=fit_contour_color,
            linewidths=fit_contour_linewidth,
            linestyles=fit_contour_linestyle,
            zorder=20,
        )

        # отражение относительно главной диагонали y = x
        if show_fit_contours_mirror:
            gaussian_for_contour_mirror = gaussian_for_contour.T

            fit_cs_mirror = ax.contour(
                X,
                Y,
                gaussian_for_contour_mirror,
                levels=fit_levels_use,
                colors=fit_contour_color,
                linewidths=fit_contour_linewidth,
                linestyles=fit_contour_linestyle,
                zorder=20,
            )

        if label_fit_contours:
            ax.clabel(fit_cs, inline=True, fontsize=8, fmt="%.2f")
            if fit_cs_mirror is not None:
                ax.clabel(fit_cs_mirror, inline=True, fontsize=8, fmt="%.2f")

    # ------------------------------------------------------------
    # Hi-C contours
    # ------------------------------------------------------------

    hic_cs = None

    if show_hic_contours and len(hic_levels_use) > 0:
        hic_cs = ax.contour(
            X,
            Y,
            Z_hic_for_contour,
            levels=hic_levels_use,
            colors=hic_contour_color,
            linewidths=hic_contour_linewidth,
            linestyles=hic_contour_linestyle,
            zorder=10,
        )

        if label_hic_contours:
            ax.clabel(hic_cs, inline=True, fontsize=8, fmt="%.2f")



    # ------------------------------------------------------------
    # Guides
    # ------------------------------------------------------------

    if show_diagonals:
        ax.axline(
            (0, 0),
            slope=1,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.5,
            zorder=30,
        )
        ax.axline(
            (0, 0),
            slope=-1,
            linestyle=":",
            linewidth=1,
            color="gray",
            alpha=0.5,
            zorder=30,
        )

    #ax.set_xlabel("Distance from fountain base, kb")
    #ax.set_ylabel("Distance from fountain base, kb")

    if title is None:
        if normalize_contours:
            title = "Aggregated fountain and Gaussian fit, normalized contours"
        else:
            title = "Aggregated fountain and Gaussian fit"

    #ax.set_title(title)

    plt.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_smoothed_hic_contours(
    aggregate_Z,
    res=5_000,
    sigma=1.0,
    levels=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    linecolor="black",
    linewidth=1.5,
    show_image=True,
    figsize=(6, 6),
):
    """
    Рисует линии уровня для агрегированного фонтана после Gaussian smoothing.

    Parameters
    ----------
    aggregate_Z : 2D array
        Агрегированный фонтан.
    res : int
        Разрешение в bp.
    sigma : float
        Ширина gaussian filter в пикселях.
    levels : list or None
        Уровни contour. Если None, будут выбраны автоматически.
    show_image : bool
        Если True, подложить саму Hi-C карту.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    # заменим nan на медиану только для сглаживания
    Z_fill = Z.copy()
    med = np.nanmedian(Z_fill[np.isfinite(Z_fill)])
    Z_fill[~np.isfinite(Z_fill)] = med

    Z_smooth = gaussian_filter(Z_fill, sigma=sigma)

    n = Z.shape[0]
    bin_kb = res / 1000.0
    coords_kb = (np.arange(n) - (n - 1) / 2.0) * bin_kb

    extent = [
        coords_kb.min(),
        coords_kb.max(),
        coords_kb.max(),
        coords_kb.min(),
    ]

    X, Y = np.meshgrid(coords_kb, coords_kb)

    if levels is None:
        # можно менять процентили под задачу
        vals = Z_smooth[np.isfinite(Z_smooth)]
        levels = np.quantile(vals, [0.70, 0.80, 0.88, 0.94, 0.97])

    fig, ax = plt.subplots(figsize=figsize)

    if show_image:
        im = ax.imshow(
            Z,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cs = ax.contour(
        X,
        Y,
        Z_smooth,
        levels=levels,
        colors=linecolor,
        linewidths=linewidth,
    )

    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.axline((0, 0), slope=1, linestyle="--", linewidth=1, color="gray", alpha=0.6)
    ax.axline((0, 0), slope=-1, linestyle=":", linewidth=1, color="gray", alpha=0.6)

    ax.set_xlabel("Distance from fountain base, kb")
    ax.set_ylabel("Distance from fountain base, kb")
    ax.set_title(f"Smoothed contours of aggregated fountain (sigma={sigma})")

    plt.tight_layout()
    return fig, ax, Z_smooth, levels


def extrusion_fountain_kernel(l, r, sigma, gamma0):
    """
    Теоретическая функция F(l, r; sigma, gamma0).

    В этой версии используется выражение:

        x2  = l
        xi1 = r

    ВАЖНО:
    В формуле sigma входит как дисперсионный параметр:
        exp(-(x2 - xi1)^2 / sigma)
        sqrt(sigma)

    То есть это НЕ старая сигма из exp(-x^2 / sigma^2),
    если ты раньше под sigma понимала стандартное отклонение.
    """
    x2 = np.asarray(l, dtype=float)
    xi1 = np.asarray(r, dtype=float)
    s = float(sigma)
    g = float(gamma0)
    if s <= 0:
        raise ValueError('sigma должна быть положительной, потому что используется sqrt(sigma).')
    sqrt_pi = np.sqrt(np.pi)
    sqrt_s = np.sqrt(s)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        term1 = np.exp(-(x2 - xi1) ** 2 / s - (x2 + xi1) * (1.0 + g))
        term2 = np.exp(s - xi1 * g - x2 * (2.0 + g)) * sqrt_pi * sqrt_s * g * (erf((-2.0 * s + x2 - xi1) / (2.0 * sqrt_s)) + erf((s + xi1) / sqrt_s))
        term3 = np.exp(s - x2 * g - xi1 * (2.0 + g)) * sqrt_pi * sqrt_s * g * (erf((s + x2) / sqrt_s) + erf((-2.0 * s - x2 + xi1) / (2.0 * sqrt_s)))
        term4 = 0.5 * np.exp(s - (x2 + xi1) * (2.0 + g)) * sqrt_pi * sqrt_s * g ** 2 * (np.exp(2.0 * x2) * erf((s + x2) / sqrt_s) + np.exp(2.0 * xi1) * erf((-2.0 * s + x2 - xi1) / (2.0 * sqrt_s)) + np.exp(2.0 * xi1) * erf((s + xi1) / sqrt_s) + np.exp(2.0 * x2) * erf((-2.0 * s - x2 + xi1) / (2.0 * sqrt_s)))
        F = term1 + term2 + term3 + term4
    F = np.asarray(F, dtype=float)
    F[~np.isfinite(F)] = np.nan
    return F


def fit_aggregate_fountain_extrusion_model(aggregate_Z, res=10000, orientation='upper', fit_max_arm_kb=200, fit_min_arm_kb=0, lp_bounds=(200, 900), sigma_bounds=(0.01, 0.05), gamma_bounds=(1, 10), normalize_shape=True, positive_weight=True, weight_clip_quantile=0.99, robust=True, min_points=30, n_starts=True):
    """
    Фитит агрегированный фонтан теоретической функцией:

        Z = C + A * F(L/lp, R/lp; sigma, gamma0)

    Parameters
    ----------
    aggregate_Z : 2D array
        Агрегированный фонтан, например O/E.

    res : int
        Разрешение Hi-C карты в bp.

    orientation : str
        Обычно "upper": верхний фонтан, L=X, R=-Y.

    fit_max_arm_kb : float
        Максимальная длина плеча, включаемая в фит.

    fit_min_arm_kb : float
        Минимальная длина плеча. Можно поставить 10–20 kb,
        если хочешь исключить область около основания/диагонали.

    lp_bounds : tuple
        Границы lp в kb.

    sigma_bounds : tuple
        Границы sigma.

    gamma_bounds : tuple
        Границы gamma0.

    normalize_shape : bool
        Если True, теоретическая форма нормируется на максимум
        внутри fit mask. Это сильно стабилизирует фит.

    Returns
    -------
    fit : dict
        Параметры фита и качество.
    """
    Z = np.asarray(aggregate_Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f'aggregate_Z должна быть квадратной матрицей, получено {Z.shape}')
    n = Z.shape[0]
    X, Y, L, R = make_extrusion_arm_coords(n=n, res=res, orientation=orientation)
    fit_mask = np.isfinite(Z) & (L >= fit_min_arm_kb) & (R >= fit_min_arm_kb) & (L <= fit_max_arm_kb) & (R <= fit_max_arm_kb)
    L_fit = L[fit_mask].astype(float)
    R_fit = R[fit_mask].astype(float)
    z_fit = Z[fit_mask].astype(float)
    if len(z_fit) < min_points:
        raise ValueError(f'Слишком мало точек для фита: {len(z_fit)}')
    C0 = float(np.nanmedian(z_fit))
    high = float(np.nanpercentile(z_fit, 95))
    A0 = max(high - C0, 1e-06)
    z_min = float(np.nanmin(z_fit))
    z_max = float(np.nanmax(z_fit))
    z_range = max(z_max - z_min, 1e-06)
    weights = np.ones_like(z_fit, dtype=float)
    if positive_weight:
        q70 = np.nanpercentile(z_fit, 70)
        q95 = np.nanpercentile(z_fit, 95)
        denom = max(q95 - q70, 1e-09)
        enrichment = np.clip((z_fit - q70) / denom, 0, 1)
        weights = 1.0 + 2.0 * enrichment
    if weight_clip_quantile is not None:
        w_max = np.nanquantile(weights, weight_clip_quantile)
        weights = np.minimum(weights, w_max)
    lower = np.array([lp_bounds[0], sigma_bounds[0], gamma_bounds[0], 0.0, z_min - z_range], dtype=float)
    upper = np.array([lp_bounds[1], sigma_bounds[1], gamma_bounds[1], 10 * z_range, z_max + z_range], dtype=float)

    def model_vector(theta):
        lp_kb, sigma, gamma0, A, C = theta
        l = L_fit / lp_kb
        r = R_fit / lp_kb
        F = extrusion_fountain_kernel(l=l, r=r, sigma=sigma, gamma0=gamma0)
        if normalize_shape:
            F_max = np.nanmax(F)
            if np.isfinite(F_max) and F_max > 0:
                F = F / F_max
        pred = C + A * F
        return pred

    def residuals(theta):
        pred = model_vector(theta)
        resids = pred - z_fit
        resids[~np.isfinite(resids)] = 0.0
        return weights * resids
    if n_starts:
        lp_starts = [lp_bounds[0], 0.5 * (lp_bounds[0] + lp_bounds[1]), lp_bounds[1]]
        sigma_starts = [sigma_bounds[0], 0.5 * (sigma_bounds[0] + sigma_bounds[1]), sigma_bounds[1]]
        gamma_starts = [gamma_bounds[0], 0.5 * (gamma_bounds[0] + gamma_bounds[1]), gamma_bounds[1]]
        starts = []
        for lp0 in lp_starts:
            for sigma0 in sigma_starts:
                for gamma0 in gamma_starts:
                    starts.append(np.array([lp0, sigma0, gamma0, A0, C0], dtype=float))
    else:
        starts = [np.array([0.5 * (lp_bounds[0] + lp_bounds[1]), 0.5 * (sigma_bounds[0] + sigma_bounds[1]), 0.5 * (gamma_bounds[0] + gamma_bounds[1]), A0, C0], dtype=float)]
    best_opt = None
    best_cost = np.inf
    failed_messages = []
    for theta0 in starts:
        try:
            opt = least_squares(residuals, theta0, bounds=(lower, upper), loss='soft_l1' if robust else 'linear', f_scale=0.1, max_nfev=50000)
            if opt.cost < best_cost:
                best_cost = opt.cost
                best_opt = opt
        except Exception as e:
            failed_messages.append(repr(e))
            continue
    if best_opt is None:
        raise RuntimeError('Все попытки фита упали. Примеры ошибок: ' + '; '.join(failed_messages[:3]))
    lp_kb, sigma, gamma0, A, C = best_opt.x
    pred_fit = model_vector(best_opt.x)
        # ----------------------------
    # Parameter uncertainty estimate
    # ----------------------------

    param_names = ["lp_kb", "sigma", "gamma0", "A", "C"]
    theta_hat = best_opt.x
    n_obs = len(z_fit)
    n_params = len(theta_hat)
    dof = max(n_obs - n_params, 1)

    # Остатки в том же виде, в котором они оптимизировались
    weighted_resid = residuals(theta_hat)

    # Для linear loss это стандартная оценка дисперсии.
    # Для robust loss это локальная приближенная оценка.
    s_sq = np.sum(weighted_resid ** 2) / dof

    J = best_opt.jac

    JTJ = J.T @ J

    # Проверяем обусловленность: если матрица плохо обусловлена,
    # параметры плохо различимы.
    cond_JTJ = np.linalg.cond(JTJ)

    try:
        cov = s_sq * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = s_sq * np.linalg.pinv(JTJ)

    param_se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    # 95% доверительный интервал, грубо через нормальное приближение
    param_ci95_low = theta_hat - 1.96 * param_se
    param_ci95_high = theta_hat + 1.96 * param_se

    # Корреляционная матрица параметров
    denom = np.outer(param_se, param_se)
    with np.errstate(divide="ignore", invalid="ignore"):
        param_corr = cov / denom
    param_corr[~np.isfinite(param_corr)] = np.nan


    ss_res = np.nansum((z_fit - pred_fit) ** 2)
    ss_tot = np.nansum((z_fit - np.nanmean(z_fit)) ** 2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan
    rmse = np.sqrt(np.nanmean((z_fit - pred_fit) ** 2))
    mae = np.nanmean(np.abs(z_fit - pred_fit))
    F_final = extrusion_fountain_kernel(l=L_fit / lp_kb, r=R_fit / lp_kb, sigma=sigma, gamma0=gamma0)
    if normalize_shape:
        shape_norm = np.nanmax(F_final)
    else:
        shape_norm = 1.0
    fit = {'lp_kb': float(lp_kb), 'sigma': float(sigma), 'gamma0': float(gamma0), 'A': float(A), 'C': float(C), 'cost': float(best_opt.cost), 'optimality': float(best_opt.optimality), 'success': bool(best_opt.success), 'message': best_opt.message, 'n_fit_pixels': int(len(z_fit)), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'fit_max_arm_kb': float(fit_max_arm_kb), 'fit_min_arm_kb': float(fit_min_arm_kb), 'orientation': orientation, 'normalize_shape': bool(normalize_shape), 'shape_norm': float(shape_norm)}
    eps = 1e-06
    fit['lp_at_lower_bound'] = abs(lp_kb - lp_bounds[0]) < eps
    fit['lp_at_upper_bound'] = abs(lp_kb - lp_bounds[1]) < eps
    fit['sigma_at_lower_bound'] = abs(sigma - sigma_bounds[0]) < eps
    fit['sigma_at_upper_bound'] = abs(sigma - sigma_bounds[1]) < eps
    fit['gamma_at_lower_bound'] = abs(gamma0 - gamma_bounds[0]) < eps
    fit['gamma_at_upper_bound'] = abs(gamma0 - gamma_bounds[1]) < eps
    fit.update({
        "param_names": param_names,

        "param_se": {
            name: float(se)
            for name, se in zip(param_names, param_se)
        },

        "param_ci95": {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(param_names, param_ci95_low, param_ci95_high)
        },

        "param_cov": cov,
        "param_corr": param_corr,

        "dof": int(dof),
        "residual_variance": float(s_sq),
        "JTJ_condition_number": float(cond_JTJ),
    })
    return fit


def plot_extrusion_model_fit(aggregate_Z, fit, res=10000, orientation=None, vmax_quantile=0.99):
    Z = np.asarray(aggregate_Z, dtype=float)
    pred = predict_extrusion_fountain_grid(aggregate_Z=Z, fit=fit, res=res, orientation=orientation)
    residual = Z - pred
    finite_vals = Z[np.isfinite(Z)]
    if len(finite_vals) > 0:
        vmax = np.nanquantile(np.abs(finite_vals), vmax_quantile)
    else:
        vmax = 1.0
    n = Z.shape[0]
    bin_kb = res / 1000
    coords = make_centered_coords(n, bin_kb)
    extent = [coords.min(), coords.max(), coords.max(), coords.min()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(Z, extent=extent, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Aggregated fountain')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pred, extent=extent, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Extrusion model fit')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    res_vmax = np.nanquantile(np.abs(residual[np.isfinite(residual)]), 0.99)
    im2 = axes[2].imshow(residual, extent=extent, cmap='coolwarm', vmin=-res_vmax, vmax=res_vmax)
    axes[2].set_title('Residual')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel('x, kb')
        ax.set_ylabel('y, kb')
    title = f"lp={fit['lp_kb']:.1f} kb, sigma={fit['sigma']:.4f}, gamma0={fit['gamma0']:.2f}, R²={fit['r2']:.3f}"
    fig.suptitle(title)
    plt.tight_layout()
    return (fig, axes)


def make_centered_coords(n, bin_kb):
    return (np.arange(n) - (n - 1) / 2) * bin_kb


def make_extrusion_arm_coords(n, res, orientation='upper'):
    """
    Создаёт координаты X, Y и соответствующие координаты плеч экструзии L, R.

    orientation="upper":
        верхний фонтан в Hi-C окне:
            L = X
            R = -Y
        валидная область: L >= 0, R >= 0

    orientation="lower":
        нижний фонтан:
            L = -X
            R = Y

    orientation="first_quadrant":
        если матрица уже в координатах L>0, R>0:
            L = X
            R = Y
    """
    bin_kb = res / 1000
    coords_kb = make_centered_coords(n, bin_kb)
    X, Y = np.meshgrid(coords_kb, coords_kb)
    if orientation == 'upper':
        L = X
        R = -Y
    elif orientation == 'lower':
        L = -X
        R = Y
    elif orientation == 'first_quadrant':
        L = X
        R = Y
    else:
        raise ValueError("orientation должен быть 'upper', 'lower' или 'first_quadrant'.")
    return (X, Y, L, R)


def predict_extrusion_fountain_grid(aggregate_Z, fit, res=10000, orientation=None):
    """
    Строит фитированную поверхность:
        Z_pred = C + A * F(L/lp, R/lp; sigma, gamma0)

    fit должен содержать:
        lp_kb, sigma, gamma0, A, C
    """
    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]
    if orientation is None:
        orientation = fit.get('orientation', 'upper')
    X, Y, L, R = make_extrusion_arm_coords(n=n, res=res, orientation=orientation)
    lp_kb = float(fit['lp_kb'])
    sigma = float(fit['sigma'])
    gamma0 = float(fit['gamma0'])
    A = float(fit.get('A', 1.0))
    C = float(fit.get('C', 0.0))
    l = L / lp_kb
    r = R / lp_kb
    F = extrusion_fountain_kernel(l=l, r=r, sigma=sigma, gamma0=gamma0)
    if fit.get('normalize_shape', True):
        shape_norm = float(fit.get('shape_norm', np.nan))
        if np.isfinite(shape_norm) and shape_norm > 0:
            F = F / shape_norm
    Z_pred = C + A * F
    valid_mask = (L >= 0) & (R >= 0)
    Z_pred[~valid_mask] = np.nan
    return Z_pred


def plot_aggregate_fountain_with_theory_contours(aggregate_Z, fit, res=10000, orientation=None, cmap='coolwarm', image_vmin=None, image_vmax=None, contour_levels=None, contour_color='black', contour_linewidth=1.2, contour_alpha=0.9, show_labels=True, title=None, output_png=None):
    """
    Рисует агрегированный фонтан как heatmap
    и накладывает contour lines теоретического фита.
    """
    Z = np.asarray(aggregate_Z, dtype=float)
    Z_pred = predict_extrusion_fountain_grid(aggregate_Z=Z, fit=fit, res=res, orientation=orientation)
    n = Z.shape[0]
    bin_kb = res / 1000
    coords_kb = make_centered_coords(n, bin_kb)
    X, Y = np.meshgrid(coords_kb, coords_kb)
    extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
    if image_vmin is None or image_vmax is None:
        finite_vals = Z[np.isfinite(Z)]
        if len(finite_vals) == 0:
            image_vmin, image_vmax = (-1, 1)
        else:
            q = np.nanquantile(np.abs(finite_vals), 0.99)
            if image_vmin is None:
                image_vmin = -q
            if image_vmax is None:
                image_vmax = q
    if contour_levels is None:
        C = float(fit.get('C', 0.0))
        A = float(fit.get('A', 1.0))
        contour_levels = C + A * np.array([0.65, 0.8, 0.9])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(Z, extent=extent, cmap=cmap, vmin=image_vmin, vmax=image_vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Aggregated fountain signal')
    cs = ax.contour(X, Y, Z_pred, levels=contour_levels, colors=contour_color, linewidths=contour_linewidth, alpha=contour_alpha)
    if show_labels:
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.set_xlabel('l, kb')
    ax.set_ylabel('r, kb')
    if title is None:
        title = f"Aggregated fountain with theoretical-fit contours\nlp={fit['lp_kb']:.1f} kb, sigma={fit['sigma']:.4f}, gamma0={fit['gamma0']:.2f}"
    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
    return (fig, ax, Z_pred)
    
    
def _get_fit_param(fit, name):
    """
    Позволяет читать параметры как из:
        a_kb, b_kb, p_kb
    так и из:
        a, b, p
    """
    if f"{name}_kb" in fit:
        return float(fit[f"{name}_kb"])
    return float(fit[name])


def plot_aggregate_with_gaussian_fit_and_theory(
    aggregate_Z,
    gaussian_fit=None,
    gaussian_fit_cache=None,
    theory_fit=None,
    res=5_000,
    orientation="upper",
    image_vmin=0.5,
    image_vmax=1.6,
    cmap="coolwarm",
    level_fractions=(0.45, 0.60, 0.75),
    gaussian_levels_frac=(0.45, 0.60, 0.75),
    theory_levels_frac=(0.45, 0.60, 0.75),
    gaussian_color="black",
    theory_color="black",
    gaussian_linewidth=1.6,
    theory_linewidth=1.6,
    title=None,
    output_png=None,
):
    """
    Рисует агрегированный фонтан и накладывает две модели:

        1. gaussian_fit — сплошные контуры
        2. theory_fit   — пунктирные контуры

    Parameters
    ----------
    aggregate_Z : 2D array
        Агрегированный фонтан, например observed/expected.

    gaussian_fit : dict or None
        Результат ft.fit_aggregate_fountain(...).
        Ожидает ключи:
            a_kb, b_kb, p_kb, A, C
        или:
            a, b, p, A, C

    gaussian_fit_cache : FountainFitCache or None
        Второй объект, который возвращает ft.fit_aggregate_fountain(...).
        Если None, координаты будут построены напрямую.

    theory_fit : dict or None
        Результат ft.fit_aggregate_fountain_extrusion_model(...).
        Ожидает ключи:
            lp_kb, sigma, gamma0, A, C

    res : int
        Разрешение карты в bp.

    level_fractions : tuple
        На каких долях амплитуды рисовать контуры.
        Например, 0.45 означает C + 0.45 A.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f"aggregate_Z должна быть квадратной матрицей, получено {Z.shape}")

    n = Z.shape[0]
    bin_kb = res / 1000

    coords_kb = make_centered_coords(n, bin_kb)
    X, Y = np.meshgrid(coords_kb, coords_kb)

    extent = [
        coords_kb.min(),
        coords_kb.max(),
        coords_kb.max(),
        coords_kb.min(),
    ]

    fig, ax = plt.subplots(figsize=(7, 6.5))

    im = ax.imshow(
        Z,
        origin="upper",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=image_vmin,
        vmax=image_vmax,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Observed / expected", fontsize=14)

    # ----------------------------
    # 1. Gaussian fit, solid contours
    # ----------------------------
    gaussian_model = None

    if gaussian_fit is not None:
        a = _get_fit_param(gaussian_fit, "a")
        b = _get_fit_param(gaussian_fit, "b")
        p = _get_fit_param(gaussian_fit, "p")
        A = float(gaussian_fit["A"])
        C = float(gaussian_fit["C"])

        theta = np.array([a, b, p, A, C], dtype=float)

        # В разных версиях функции gaussian_fountain_model мог быть
        # или не быть аргумент model_kind, поэтому делаем устойчиво.
        try:
            gaussian_model = gaussian_fountain_model(theta, X, Y)
        except TypeError:
            gaussian_model = gaussian_fountain_model(
                theta,
                X,
                Y,
                model_kind="anti_diag_peak",
            )

        gaussian_levels = C + A * np.array(gaussian_levels_frac)

        ax.contour(
            X,
            Y,
            gaussian_model,
            levels=gaussian_levels,
            colors=gaussian_color,
            linewidths=gaussian_linewidth,
            linestyles="solid",
        )

    # ----------------------------
    # 2. Theoretical fit, dashed contours
    # ----------------------------
    theory_model = None

    if theory_fit is not None:
        theory_model = predict_extrusion_fountain_grid(
            aggregate_Z=Z,
            fit=theory_fit,
            res=res,
            orientation=orientation,
        )

        C_th = float(theory_fit.get("C", 0.0))
        A_th = float(theory_fit.get("A", 1.0))

        theory_levels = C_th + A_th * np.array(theory_levels_frac)

        ax.contour(
            X,
            Y,
            theory_model,
            levels=theory_levels,
            colors=theory_color,
            linewidths=theory_linewidth,
            linestyles="dashed",
        )

    # ----------------------------
    # Reference lines
    # ----------------------------
    ax.axline((0, 0), slope=1, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.axline((0, 0), slope=-1, linestyle=":", linewidth=1, color="gray", alpha=0.7)
    ax.axvline(0, linewidth=0.7, color="gray", alpha=0.4)
    ax.axhline(0, linewidth=0.7, color="gray", alpha=0.4)

    #ax.set_xlabel("Distance from fountain base, kb")
    #ax.set_ylabel("Distance from fountain base, kb")

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)

    legend_handles = []

    if gaussian_fit is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=gaussian_color,
                linewidth=gaussian_linewidth,
                linestyle="solid",
                label="Gaussian fit",
            )
        )

    if theory_fit is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=theory_color,
                linewidth=theory_linewidth,
                linestyle="dashed",
                label="Theoretical fit",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=14)

    if title is None:
        title_parts = ["Aggregate fountain"]

        if gaussian_fit is not None:
            title_parts.append(
                f"Gaussian: a={_get_fit_param(gaussian_fit, 'a'):.1f}, "
                f"b={_get_fit_param(gaussian_fit, 'b'):.1f}, "
                f"p={_get_fit_param(gaussian_fit, 'p'):.1f} kb"
            )

        if theory_fit is not None:
            title_parts.append(
                f"Theory: lp={theory_fit['lp_kb']:.1f} kb, "
                f"sigma={theory_fit['sigma']:.4g}, "
                f"gamma0={theory_fit['gamma0']:.2f}"
            )

        title = "\n".join(title_parts)

    #ax.set_title(title)

    plt.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight")

    return fig, ax, {
        "gaussian_model": gaussian_model,
        "theory_model": theory_model,
    }


def get_fit_value(fit, name):
    """
    Позволяет использовать fit dict с ключами:
        a_kb, b_kb, p_kb
    или:
        a, b, p
    """
    if f'{name}_kb' in fit:
        return float(fit[f'{name}_kb'])
    return float(fit[name])


def gaussian_axis_profile(s_kb, width_kb, A, C):
    """
    Для нашей 2D-модели:

        G = C + A exp[-((x+y)^2/a^2 + (x-y-2p)^2/b^2)]

    Если перейти к ортонормированным координатам:

        U = (x+y)/sqrt(2)
        V = (x-y-2p)/sqrt(2)

    то вдоль оси U:

        G(U) = C + A exp[-2 U^2 / a^2]

    а вдоль V:

        G(V) = C + A exp[-2 V^2 / b^2]
    """
    return C + A * np.exp(-2 * s_kb ** 2 / width_kb ** 2)


def binned_strip_profile(Z, axis_coord, perp_coord, base_mask, profile_range_kb, bin_width_kb, strip_halfwidth_kb):
    """
    Строит экспериментальный профиль вдоль оси.

    axis_coord — координата вдоль оси.
    perp_coord — координата поперёк оси.

    Берём не строго линию, а узкую полосу:
        abs(perp_coord) <= strip_halfwidth_kb

    Это устойчивее для пиксельной Hi-C карты.
    """
    edges = np.arange(-profile_range_kb, profile_range_kb + bin_width_kb, bin_width_kb)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for left, right, center in zip(edges[:-1], edges[1:], centers):
        mask = base_mask & (np.abs(perp_coord) <= strip_halfwidth_kb) & (axis_coord >= left) & (axis_coord < right)
        values = Z[mask]
        values = values[np.isfinite(values)]
        n = len(values)
        if n == 0:
            mean = np.nan
            std = np.nan
            sem = np.nan
        elif n == 1:
            mean = float(values[0])
            std = np.nan
            sem = np.nan
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            sem = std / np.sqrt(n)
        rows.append({'distance_kb': center, 'signal_mean': mean, 'signal_std': std, 'signal_sem': sem, 'n_pixels': n})
    return pd.DataFrame(rows)


def make_ellipse_axis_profile_data(aggregate_Z, fit, res=10000, profile_range_kb=None, bin_width_kb=None, strip_halfwidth_kb=None, upper_triangle=True, diag_exclusion_kb=10, n_gaussian_points=500):
    """
    Делает таблицы для графика профилей вдоль большой и малой осей эллипса.

    Возвращает:
        profile_df  — экспериментальные binned-профили
        gaussian_df — гладкие fitted Gaussian-профили
        meta        — параметры и описание осей
    """
    Z = np.asarray(aggregate_Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError('aggregate_Z должен быть квадратной 2D-матрицей.')
    n = Z.shape[0]
    bin_kb = res / 1000
    a = get_fit_value(fit, 'a')
    b = get_fit_value(fit, 'b')
    p = get_fit_value(fit, 'p')
    A = float(fit['A'])
    C = float(fit['C'])
    if bin_width_kb is None:
        bin_width_kb = bin_kb
    if strip_halfwidth_kb is None:
        strip_halfwidth_kb = 1.5 * bin_kb
    if profile_range_kb is None:
        profile_range_kb = 2.5 * max(a, b)
    x_kb = make_centered_coords(n, bin_kb)
    y_kb = make_centered_coords(n, bin_kb)
    X, Y = np.meshgrid(x_kb, y_kb)
    U = (X + Y) / np.sqrt(2)
    V = (X - Y - 2 * p) / np.sqrt(2)
    base_mask = np.isfinite(Z)
    if upper_triangle:
        base_mask &= Y < X - diag_exclusion_kb
    axis_specs = {'U': {'axis_coord': U, 'perp_coord': V, 'width_kb': a, 'description': 'U = (x + y) / sqrt(2), parallel to main diagonal'}, 'V': {'axis_coord': V, 'perp_coord': U, 'width_kb': b, 'description': 'V = (x - y - 2p) / sqrt(2), parallel to anti-diagonal'}}
    if a >= b:
        major_key = 'U'
        minor_key = 'V'
    else:
        major_key = 'V'
        minor_key = 'U'
    labelled_axes = {'major': major_key, 'minor': minor_key}
    profile_rows = []
    gaussian_rows = []
    s_grid = np.linspace(-profile_range_kb, profile_range_kb, n_gaussian_points)
    for axis_label, key in labelled_axes.items():
        spec = axis_specs[key]
        profile = binned_strip_profile(Z=Z, axis_coord=spec['axis_coord'], perp_coord=spec['perp_coord'], base_mask=base_mask, profile_range_kb=profile_range_kb, bin_width_kb=bin_width_kb, strip_halfwidth_kb=strip_halfwidth_kb)
        profile['axis'] = axis_label
        profile['raw_axis'] = key
        profile['width_kb'] = spec['width_kb']
        profile['axis_description'] = spec['description']
        profile_rows.append(profile)
        g = gaussian_axis_profile(s_kb=s_grid, width_kb=spec['width_kb'], A=A, C=C)
        gaussian_rows.append(pd.DataFrame({'axis': axis_label, 'raw_axis': key, 'distance_kb': s_grid, 'gaussian_signal': g, 'width_kb': spec['width_kb'], 'axis_description': spec['description']}))
    profile_df = pd.concat(profile_rows, ignore_index=True)
    gaussian_df = pd.concat(gaussian_rows, ignore_index=True)
    meta = {'a_kb': a, 'b_kb': b, 'p_kb': p, 'A': A, 'C': C, 'major_raw_axis': major_key, 'minor_raw_axis': minor_key, 'profile_range_kb': profile_range_kb, 'bin_width_kb': bin_width_kb, 'strip_halfwidth_kb': strip_halfwidth_kb, 'upper_triangle': upper_triangle, 'diag_exclusion_kb': diag_exclusion_kb}
    return (profile_df, gaussian_df, meta)


def plot_ellipse_axis_profiles_barplot(profile_df, gaussian_df, meta=None, show_sem=True, title=None, ax=None, bar_width_kb=None, bar_alpha=0.45, bar_from_background=False):
    """
    Рисует 4 кривые/серии на одной картинке:

        barplot experimental major axis
        line fitted Gaussian major axis
        barplot experimental minor axis
        line fitted Gaussian minor axis

    Parameters
    ----------
    profile_df : DataFrame
        Таблица экспериментальных профилей, созданная make_ellipse_axis_profile_data.

    gaussian_df : DataFrame
        Таблица fitted Gaussian-профилей.

    meta : dict or None
        Метаданные, возвращённые make_ellipse_axis_profile_data.

    show_sem : bool
        Если True, рисует error bars для experimental barplot.

    bar_width_kb : float or None
        Ширина столбиков в kb. Если None, берётся из meta["bin_width_kb"].

    bar_alpha : float
        Прозрачность столбиков.

    bar_from_background : bool
        Если True, столбики рисуются относительно fitted background C.
        Это удобно, если хочется видеть именно excess над фоном.
        Если False, столбики идут от нуля.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    if bar_width_kb is None:
        if meta is not None and 'bin_width_kb' in meta:
            bar_width_kb = meta['bin_width_kb']
        else:
            sorted_x = np.sort(profile_df['distance_kb'].dropna().unique())
            if len(sorted_x) > 1:
                bar_width_kb = np.median(np.diff(sorted_x))
            else:
                bar_width_kb = 10
    offsets = {'major': -0.18 * bar_width_kb, 'minor': 0.18 * bar_width_kb}
    bar_width_each = 0.36 * bar_width_kb
    for axis_label in ['major', 'minor']:
        data = profile_df[(profile_df['axis'] == axis_label) & np.isfinite(profile_df['signal_mean'])].copy()
        gauss = gaussian_df[gaussian_df['axis'] == axis_label].copy()
        if len(data) == 0:
            continue
        x = data['distance_kb'].to_numpy() + offsets[axis_label]
        y = data['signal_mean'].to_numpy()
        if bar_from_background and meta is not None:
            baseline = meta['C']
            heights = y - baseline
            bottom = np.full_like(heights, baseline, dtype=float)
        else:
            heights = y
            bottom = None
        if show_sem and 'signal_sem' in data.columns:
            yerr = data['signal_sem'].to_numpy()
        else:
            yerr = None
        bars = ax.bar(x, heights, width=bar_width_each, bottom=bottom, yerr=yerr, capsize=2 if yerr is not None else 0, alpha=bar_alpha, label=f'experimental {axis_label}')
        bar_color = bars.patches[0].get_facecolor()
        ax.plot(gauss['distance_kb'], gauss['gaussian_signal'], linewidth=2, color=bar_color, label=f'fitted Gaussian {axis_label}')
    if meta is not None:
        ax.axhline(meta['C'], linestyle='--', linewidth=1, label='fitted background C')
    ax.set_xlabel('Distance along ellipse axis, kb')
    ax.set_ylabel('Aggregated signal')
    if title is None:
        if meta is not None:
            title = f"Aggregated fountain profiles along fitted ellipse axes\na={meta['a_kb']:.1f} kb, b={meta['b_kb']:.1f} kb, p={meta['p_kb']:.1f} kb"
        else:
            title = 'Aggregated fountain profiles along fitted ellipse axes'
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return (fig, ax)


def save_experimental_profile_points(profile_df, meta=None, output_csv='experimental_axis_profile_points.csv', experiment_name=None, aggregate_name=None, include_empty_bins=False, sep=','):
    """
    Сохраняет именно экспериментальные точки профилей вдоль осей эллипса.

    profile_df — таблица из make_ellipse_axis_profile_data:
        distance_kb
        signal_mean
        signal_std
        signal_sem
        n_pixels
        axis
        raw_axis
        width_kb
        axis_description

    gaussian_df здесь специально не используется, потому что это fitted curve,
    а не экспериментальные точки.
    """
    out = profile_df.copy()
    if not include_empty_bins:
        out = out[np.isfinite(out['signal_mean']) & (out['n_pixels'] > 0)].copy()
    if experiment_name is not None:
        out.insert(0, 'experiment', experiment_name)
    if aggregate_name is not None:
        out.insert(1 if experiment_name is not None else 0, 'aggregate', aggregate_name)
    if meta is not None:
        meta_fields = ['a_kb', 'b_kb', 'p_kb', 'A', 'C', 'major_raw_axis', 'minor_raw_axis', 'profile_range_kb', 'bin_width_kb', 'strip_halfwidth_kb', 'upper_triangle', 'diag_exclusion_kb']
        for key in meta_fields:
            if key in meta:
                out[key] = meta[key]
    preferred_cols = ['experiment', 'aggregate', 'axis', 'raw_axis', 'distance_kb', 'signal_mean', 'signal_std', 'signal_sem', 'n_pixels', 'width_kb', 'axis_description', 'a_kb', 'b_kb', 'p_kb', 'A', 'C', 'major_raw_axis', 'minor_raw_axis', 'profile_range_kb', 'bin_width_kb', 'strip_halfwidth_kb', 'upper_triangle', 'diag_exclusion_kb']
    cols = [c for c in preferred_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in cols]
    out = out[cols + other_cols]
    output_csv = Path(output_csv)
    out.to_csv(output_csv, index=False, sep=sep)
    return out


def load_baranasic_promoters(path, sheet_name=0, chrom_col='Chromosome', start_col='Start', end_col='End'):
    """
    Загружает таблицу промоторов Baranasic Supplementary Table 4.

    Ожидаемые минимальные колонки:
        Chromosome, Start, End

    Остальные колонки сохраняются.
    """
    path = Path(path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path, sep=None, engine='python')
    df = df.copy()
    required = [chrom_col, start_col, end_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'В таблице промоторов не хватает колонок: {missing}. Доступные колонки: {list(df.columns)}')
    df['promoter_chrom'] = df[chrom_col].astype(str)
    df['promoter_start'] = df[start_col].astype(int)
    df['promoter_end'] = df[end_col].astype(int)
    df['promoter_center'] = ((df['promoter_start'] + df['promoter_end']) // 2).astype(int)
    if 'promoter_id' not in df.columns:
        df['promoter_id'] = np.arange(len(df))
    return df


def harmonize_promoter_chrom_style(promoters_df, fountains_df):
    """
    Приводит promoter_chrom к стилю chrom в fountains_df.

    Например:
        promoters: 1, 2, 3
        fountains: chr1, chr2, chr3
    или наоборот.
    """
    promoters = promoters_df.copy()
    fountain_has_chr = fountains_df['chrom'].astype(str).str.startswith('chr').mean() > 0.5
    promoter_has_chr = promoters['promoter_chrom'].astype(str).str.startswith('chr').mean() > 0.5
    if fountain_has_chr and (not promoter_has_chr):
        promoters['promoter_chrom'] = 'chr' + promoters['promoter_chrom'].astype(str)
    elif promoter_has_chr and (not fountain_has_chr):
        promoters['promoter_chrom'] = promoters['promoter_chrom'].astype(str).str.replace('^chr', '', regex=True)
    return promoters


def find_promoters_near_fountains(fountains_df, promoters_df, window_bp=200000, fountain_id_col='Fountain index', base_col='base_bp'):
    """
    Для каждого фонтана ищет все промоторы в окне +/- window_bp.

    Возвращает:
        hits_df:
            long-table, одна строка = один promoter около одного fountain.

        summary_df:
            одна строка = один fountain, summary по промоторам.
    """
    fountains = fountains_df.copy()
    promoters = promoters_df.copy()
    if base_col not in fountains.columns:
        if {'start', 'end'}.issubset(fountains.columns):
            fountains[base_col] = (fountains['start'].astype(int) + fountains['end'].astype(int)) // 2
        else:
            raise ValueError(f'В fountains_df нет {base_col}, и нельзя восстановить из start/end.')
    promoters = harmonize_promoter_chrom_style(promoters, fountains)
    promoters_by_chrom = {chrom: g.sort_values('promoter_start').reset_index(drop=True) for chrom, g in promoters.groupby('promoter_chrom', sort=False)}
    hit_rows = []
    summary_rows = []
    for _, f in tqdm(fountains.iterrows(), total=len(fountains), desc='Finding promoters near fountains'):
        chrom = str(f['chrom'])
        base = int(f[base_col])
        fountain_id = f[fountain_id_col] if fountain_id_col in f.index else f.name
        window_start = base - window_bp
        window_end = base + window_bp
        if chrom not in promoters_by_chrom:
            summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, 'n_promoters_200kb': 0, 'mean_abs_distance_to_promoter_bp': np.nan, 'median_abs_distance_to_promoter_bp': np.nan, 'nearest_abs_distance_to_promoter_bp': np.nan, 'nearest_promoter_id': np.nan, 'promoter_ids_200kb': ''})
            continue
        p = promoters_by_chrom[chrom]
        hits = p[(p['promoter_end'] >= window_start) & (p['promoter_start'] <= window_end)].copy()
        if len(hits) == 0:
            summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, 'n_promoters_200kb': 0, 'mean_abs_distance_to_promoter_bp': np.nan, 'median_abs_distance_to_promoter_bp': np.nan, 'nearest_abs_distance_to_promoter_bp': np.nan, 'nearest_promoter_id': np.nan, 'promoter_ids_200kb': ''})
            continue
        hits['distance_to_promoter_interval_bp'] = [point_to_interval_distance(base, int(s), int(e)) for s, e in zip(hits['promoter_start'], hits['promoter_end'])]
        hits['signed_distance_to_promoter_center_bp'] = hits['promoter_center'].astype(int) - base
        hits['abs_distance_to_promoter_center_bp'] = hits['signed_distance_to_promoter_center_bp'].abs()
        hits = hits.sort_values(['distance_to_promoter_interval_bp', 'abs_distance_to_promoter_center_bp', 'promoter_start']).copy()
        distances = hits['distance_to_promoter_interval_bp'].to_numpy(dtype=float)
        nearest = hits.iloc[0]
        summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, 'n_promoters_200kb': int(len(hits)), 'mean_abs_distance_to_promoter_bp': float(np.mean(distances)), 'median_abs_distance_to_promoter_bp': float(np.median(distances)), 'nearest_abs_distance_to_promoter_bp': float(np.min(distances)), 'nearest_promoter_id': nearest['promoter_id'], 'nearest_promoter_start': int(nearest['promoter_start']), 'nearest_promoter_end': int(nearest['promoter_end']), 'nearest_promoter_center': int(nearest['promoter_center']), 'promoter_ids_200kb': ';'.join(hits['promoter_id'].astype(str))})
        for rank, (_, h) in enumerate(hits.iterrows(), start=1):
            row = {fountain_id_col: fountain_id, 'fountain_chrom': chrom, 'fountain_base_bp': base, 'promoter_rank_by_distance': rank, 'distance_to_promoter_interval_bp': int(h['distance_to_promoter_interval_bp']), 'signed_distance_to_promoter_center_bp': int(h['signed_distance_to_promoter_center_bp']), 'abs_distance_to_promoter_center_bp': int(h['abs_distance_to_promoter_center_bp'])}
            for col in hits.columns:
                row[col] = h[col]
            hit_rows.append(row)
    hits_df = pd.DataFrame(hit_rows)
    summary_df = pd.DataFrame(summary_rows)
    return (hits_df, summary_df)


def load_dome_chromhmm_bed(path, sep=';'):
    """
    Загружает BED-like файл с Dome ChromHMM/PADRE annotation.

    Ожидаемый формат:
        #chrom chromStart chromEnd name score strand thickStart thickEnd reserved
        chr1   11016      12769    1_TssA1 ...

    Возвращает таблицу с нормализованными колонками:
        chrom, start, end, state, score, feature_id
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.lstrip('#') for c in df.columns]
    required = ['chrom', 'chromStart', 'chromEnd', 'name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Не хватает колонок {missing}. Доступные колонки: {list(df.columns)}')
    out = df.copy()
    out['chrom'] = out['chrom'].astype(str)
    out['start'] = out['chromStart'].astype(int)
    out['end'] = out['chromEnd'].astype(int)
    out['state'] = out['name'].astype(str)
    if 'score' in out.columns:
        out['state_score'] = pd.to_numeric(out['score'], errors='coerce')
    else:
        out['state_score'] = np.nan
    out['feature_id'] = np.arange(len(out))
    return out


def select_chromhmm_features(reg, feature_type='enhancer', mode='strict'):
    """
    feature_type:
        "enhancer" или "promoter"

    mode:
        для enhancer:
            "strict" = 5_EnhA1
            "broad"  = 5_EnhA1 + 6_EnhFlank + 7_EnhWk1

        для promoter:
            "strict" = 1_TssA1 + 2_TssA2
            "broad"  = 1_TssA1 + 2_TssA2 + 3_TssFlank1 + 4_TssFlank2
    """
    if feature_type == 'enhancer':
        if mode == 'strict':
            states = ['5_EnhA1']
        elif mode == 'broad':
            states = ['5_EnhA1', '6_EnhFlank', '7_EnhWk1']
        else:
            raise ValueError("Для enhancer mode должен быть 'strict' или 'broad'.")
    elif feature_type == 'promoter':
        if mode == 'strict':
            states = ['1_TssA1', '2_TssA2']
        elif mode == 'broad':
            states = ['1_TssA1', '2_TssA2', '3_TssFlank1', '4_TssFlank2']
        else:
            raise ValueError("Для promoter mode должен быть 'strict' или 'broad'.")
    else:
        raise ValueError("feature_type должен быть 'enhancer' или 'promoter'.")
    features = reg[reg['state'].isin(states)].copy()
    features['feature_type'] = feature_type
    features['feature_mode'] = mode
    features['feature_center'] = ((features['start'].astype(int) + features['end'].astype(int)) // 2).astype(int)
    return features


def prepare_fountains_for_annotation(fountains_df):
    fountains = fountains_df.copy()
    if 'chrom' not in fountains.columns:
        raise ValueError('В таблице фонтанов нет колонки chrom.')
    fountains['chrom'] = fountains['chrom'].astype(str)
    if 'base_bp' not in fountains.columns:
        if {'start', 'end'}.issubset(fountains.columns):
            fountains['base_bp'] = (fountains['start'].astype(int) + fountains['end'].astype(int)) // 2
        else:
            raise ValueError('В таблице фонтанов нет base_bp, start/end тоже нет.')
    if 'Fountain index' not in fountains.columns:
        fountains['Fountain index'] = np.arange(len(fountains))
    return fountains


def harmonize_chrom_style(df, reference_df, chrom_col='chrom'):
    """
    Приводит стиль хромосом df к стилю reference_df.
    Например:
        chr1 -> 1
        или
        1 -> chr1
    """
    out = df.copy()
    df_has_chr = out[chrom_col].astype(str).str.startswith('chr').mean() > 0.5
    ref_has_chr = reference_df[chrom_col].astype(str).str.startswith('chr').mean() > 0.5
    if ref_has_chr and (not df_has_chr):
        out[chrom_col] = 'chr' + out[chrom_col].astype(str)
    elif df_has_chr and (not ref_has_chr):
        out[chrom_col] = out[chrom_col].astype(str).str.replace('^chr', '', regex=True)
    return out


def point_to_interval_distance(point, start, end):
    """
    Расстояние от точки до BED-интервала [start, end).
    Если точка внутри интервала, расстояние 0.
    """
    if point < start:
        return start - point
    if point >= end:
        return point - end
    return 0


def annotate_fountains_with_nearby_features(fountains_df, features_df, feature_name='enhancer', max_distance_bp=0, fountain_id_col='Fountain index', base_col='base_bp'):
    """
    Для каждого фонтана считает ближайший feature и число feature,
    пересекающих fountain bin.

    max_distance_bp:
        0       -> feature должен прямо пересекать точку base_bp
        10_000  -> feature может быть в пределах 10 kb от base_bp
        и т.д.

    Но дополнительно считается overlap с самим fountain bin [start, end].
    """
    fountains = fountains_df.copy()
    features = harmonize_chrom_style(features_df, fountains)
    features_by_chrom = {chrom: g.sort_values('start').reset_index(drop=True) for chrom, g in features.groupby('chrom', sort=False)}
    summary_rows = []
    hit_rows = []
    for _, f in tqdm(fountains.iterrows(), total=len(fountains), desc=f'Annotating fountains by {feature_name}s'):
        chrom = str(f['chrom'])
        base = int(f[base_col])
        fountain_id = f[fountain_id_col]
        if {'start', 'end'}.issubset(f.index):
            f_start = int(f['start'])
            f_end = int(f['end'])
        else:
            f_start = base - 5000
            f_end = base + 5000
        if chrom not in features_by_chrom:
            summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, f'n_{feature_name}s_overlapping_fountain_bin': 0, f'n_{feature_name}s_within_{max_distance_bp}_bp': 0, f'nearest_{feature_name}_distance_bp': np.nan, f'nearest_{feature_name}_state': np.nan, f'nearest_{feature_name}_start': np.nan, f'nearest_{feature_name}_end': np.nan})
            continue
        g = features_by_chrom[chrom].copy()
        overlap_bin = g[(g['end'] > f_start) & (g['start'] < f_end)].copy()
        g[f'distance_to_{feature_name}_bp'] = [point_to_interval_distance(base, int(s), int(e)) for s, e in zip(g['start'], g['end'])]
        near = g[g[f'distance_to_{feature_name}_bp'] <= max_distance_bp].copy()
        g_sorted = g.sort_values([f'distance_to_{feature_name}_bp', 'start', 'end'])
        nearest = g_sorted.iloc[0]
        summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, f'n_{feature_name}s_overlapping_fountain_bin': int(len(overlap_bin)), f'n_{feature_name}s_within_{max_distance_bp}_bp': int(len(near)), f'nearest_{feature_name}_distance_bp': int(nearest[f'distance_to_{feature_name}_bp']), f'nearest_{feature_name}_state': nearest['state'], f'nearest_{feature_name}_start': int(nearest['start']), f'nearest_{feature_name}_end': int(nearest['end']), f'nearest_{feature_name}_score': nearest.get('state_score', np.nan)})
        for rank, (_, h) in enumerate(overlap_bin.iterrows(), start=1):
            hit_rows.append({fountain_id_col: fountain_id, 'fountain_chrom': chrom, 'fountain_start': f_start, 'fountain_end': f_end, 'fountain_base_bp': base, f'{feature_name}_rank_in_bin': rank, f'{feature_name}_chrom': h['chrom'], f'{feature_name}_start': int(h['start']), f'{feature_name}_end': int(h['end']), f'{feature_name}_state': h['state'], f'{feature_name}_score': h.get('state_score', np.nan)})
    summary_df = pd.DataFrame(summary_rows)
    hits_df = pd.DataFrame(hit_rows)
    return (summary_df, hits_df)


def find_features_in_window_around_fountains(fountains_df, features_df, feature_name='promoter', window_bp=200000, fountain_id_col='Fountain index', base_col='base_bp'):
    """
    Для каждого фонтана ищет все features в окне +/- window_bp
    вокруг base_bp.

    Возвращает:
        hits_df    — long table, один ряд = один promoter около одного fountain
        summary_df — summary на один fountain
    """
    fountains = fountains_df.copy()
    features = harmonize_chrom_style(features_df, fountains)
    features = features.copy()
    features['feature_center'] = ((features['start'].astype(int) + features['end'].astype(int)) // 2).astype(int)
    features_by_chrom = {chrom: g.sort_values('start').reset_index(drop=True) for chrom, g in features.groupby('chrom', sort=False)}
    hit_rows = []
    summary_rows = []
    for _, f in tqdm(fountains.iterrows(), total=len(fountains), desc=f'Finding {feature_name}s in ±{window_bp // 1000} kb'):
        chrom = str(f['chrom'])
        base = int(f[base_col])
        fountain_id = f[fountain_id_col]
        window_start = base - window_bp
        window_end = base + window_bp
        if chrom not in features_by_chrom:
            hits = pd.DataFrame()
        else:
            g = features_by_chrom[chrom]
            hits = g[(g['end'] > window_start) & (g['start'] < window_end)].copy()
        if len(hits) == 0:
            summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, f'n_{feature_name}s_{window_bp // 1000}kb': 0, f'mean_abs_distance_to_{feature_name}_bp': np.nan, f'median_abs_distance_to_{feature_name}_bp': np.nan, f'nearest_abs_distance_to_{feature_name}_bp': np.nan, f'nearest_{feature_name}_state': np.nan, f'nearest_{feature_name}_start': np.nan, f'nearest_{feature_name}_end': np.nan})
            continue
        hits[f'distance_to_{feature_name}_interval_bp'] = [point_to_interval_distance(base, int(s), int(e)) for s, e in zip(hits['start'], hits['end'])]
        hits[f'signed_distance_to_{feature_name}_center_bp'] = hits['feature_center'].astype(int) - base
        hits[f'abs_distance_to_{feature_name}_center_bp'] = hits[f'signed_distance_to_{feature_name}_center_bp'].abs()
        hits = hits.sort_values([f'distance_to_{feature_name}_interval_bp', f'abs_distance_to_{feature_name}_center_bp', 'start', 'end']).copy()
        distances = hits[f'distance_to_{feature_name}_interval_bp'].to_numpy(dtype=float)
        nearest = hits.iloc[0]
        summary_rows.append({fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, f'n_{feature_name}s_{window_bp // 1000}kb': int(len(hits)), f'mean_abs_distance_to_{feature_name}_bp': float(np.mean(distances)), f'median_abs_distance_to_{feature_name}_bp': float(np.median(distances)), f'nearest_abs_distance_to_{feature_name}_bp': float(np.min(distances)), f'nearest_{feature_name}_state': nearest['state'], f'nearest_{feature_name}_start': int(nearest['start']), f'nearest_{feature_name}_end': int(nearest['end']), f'nearest_{feature_name}_score': nearest.get('state_score', np.nan)})
        for rank, (_, h) in enumerate(hits.iterrows(), start=1):
            row = {fountain_id_col: fountain_id, 'fountain_chrom': chrom, 'fountain_base_bp': base, f'{feature_name}_rank_by_distance': rank, f'{feature_name}_chrom': h['chrom'], f'{feature_name}_start': int(h['start']), f'{feature_name}_end': int(h['end']), f'{feature_name}_center': int(h['feature_center']), f'{feature_name}_state': h['state'], f'{feature_name}_score': h.get('state_score', np.nan), f'distance_to_{feature_name}_interval_bp': int(h[f'distance_to_{feature_name}_interval_bp']), f'signed_distance_to_{feature_name}_center_bp': int(h[f'signed_distance_to_{feature_name}_center_bp']), f'abs_distance_to_{feature_name}_center_bp': int(h[f'abs_distance_to_{feature_name}_center_bp'])}
            hit_rows.append(row)
    hits_df = pd.DataFrame(hit_rows)
    summary_df = pd.DataFrame(summary_rows)
    return (hits_df, summary_df)


def plot_signed_promoter_distances(promoter_hits_df, bin_width_kb=10, window_kb=200, title=None, output_png=None):
    df = promoter_hits_df.copy()
    x_kb = df['signed_distance_to_promoter_center_bp'].to_numpy(dtype=float) / 1000
    bins = np.arange(-window_kb, window_kb + bin_width_kb, bin_width_kb)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x_kb, bins=bins, alpha=0.75)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Signed distance from fountain base to promoter center, kb')
    ax.set_ylabel('Number of fountain-promoter pairs')
    if title is None:
        title = 'Promoter distances around fountain bases'
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
    return (fig, ax)


def plot_abs_promoter_distances(promoter_hits_df, bin_width_kb=10, window_kb=200, title=None, output_png=None):
    df = promoter_hits_df.copy()
    x_kb = df['abs_distance_to_promoter_center_bp'].to_numpy(dtype=float) / 1000
    bins = np.arange(0, window_kb + bin_width_kb, bin_width_kb)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x_kb, bins=bins, alpha=0.75)
    ax.set_xlabel('Absolute distance from fountain base to promoter center, kb')
    ax.set_ylabel('Number of fountain-promoter pairs')
    if title is None:
        title = 'Absolute promoter distances around fountain bases'
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
    return (fig, ax)


def ensure_feature_center(df, start_col='start', end_col='end', center_col='feature_center'):
    out = df.copy()
    if center_col not in out.columns:
        out[center_col] = ((out[start_col].astype(int) + out[end_col].astype(int)) // 2).astype(int)
    return out


def point_to_intervals_distance(point, starts, ends):
    """
    Расстояние от точки до массива BED-интервалов [start, end).
    Если точка внутри интервала, расстояние 0.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    point = int(point)
    return np.where(point < starts, starts - point, np.where(point >= ends, point - ends, 0))


def prepare_fountains_for_ep_plot(fountains_df, base_col='base_bp'):
    fountains = fountains_df.copy()
    if 'chrom' not in fountains.columns:
        raise ValueError('В fountains_df нет колонки chrom.')
    fountains['chrom'] = fountains['chrom'].astype(str)
    if base_col not in fountains.columns:
        if {'start', 'end'}.issubset(fountains.columns):
            fountains[base_col] = (fountains['start'].astype(int) + fountains['end'].astype(int)) // 2
        else:
            raise ValueError(f'Нет {base_col}, и нельзя восстановить из start/end.')
    if 'Fountain index' not in fountains.columns:
        fountains['Fountain index'] = np.arange(len(fountains))
    return fountains


def find_enhancers_near_fountains(fountains_df, enhancers_df, max_enhancer_distance_bp=10000, fountain_id_col='Fountain index', base_col='base_bp'):
    """
    Ищет все enhancers, находящиеся не дальше max_enhancer_distance_bp
    от базы фонтана.

    Возвращает long-table:
        одна строка = fountain-enhancer pair.
    """
    fountains = prepare_fountains_for_ep_plot(fountains_df, base_col=base_col)
    enhancers = enhancers_df.copy()
    enhancers = harmonize_chrom_style(enhancers, fountains)
    enhancers = ensure_feature_center(enhancers)
    enhancers_by_chrom = {chrom: g.sort_values('start').reset_index(drop=True) for chrom, g in enhancers.groupby('chrom', sort=False)}
    rows = []
    for _, f in tqdm(fountains.iterrows(), total=len(fountains), desc='Finding enhancers near fountains'):
        chrom = str(f['chrom'])
        base = int(f[base_col])
        fountain_id = f[fountain_id_col]
        if chrom not in enhancers_by_chrom:
            continue
        e = enhancers_by_chrom[chrom]
        candidates = e[(e['end'] >= base - max_enhancer_distance_bp) & (e['start'] <= base + max_enhancer_distance_bp)].copy()
        if len(candidates) == 0:
            continue
        candidates['distance_to_fountain_base_bp'] = point_to_intervals_distance(base, candidates['start'], candidates['end'])
        candidates = candidates[candidates['distance_to_fountain_base_bp'] <= max_enhancer_distance_bp].copy()
        if len(candidates) == 0:
            continue
        for _, h in candidates.iterrows():
            row = {fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, 'enhancer_feature_id': h.get('feature_id', np.nan), 'enhancer_start': int(h['start']), 'enhancer_end': int(h['end']), 'enhancer_center': int(h['feature_center']), 'enhancer_state': h.get('state', np.nan), 'enhancer_score': h.get('state_score', np.nan), 'enhancer_distance_to_base_bp': int(h['distance_to_fountain_base_bp']), 'enhancer_rel_start_kb': (int(h['start']) - base) / 1000, 'enhancer_rel_end_kb': (int(h['end']) - base) / 1000, 'enhancer_rel_center_kb': (int(h['feature_center']) - base) / 1000}
            for col in ['r', 'q1_r_pearson', 'Fountain Score', 'a_kb', 'b_kb', 'p_kb']:
                if col in f.index:
                    row[col] = f[col]
            rows.append(row)
    return pd.DataFrame(rows)


def build_enhancer_promoter_contacts_around_fountains(enhancer_fountain_hits, promoters_df, promoter_window_bp=200000, fountain_id_col='Fountain index'):
    """
    Для каждого enhancer-fountain pair ищет все promoters в окне ±promoter_window_bp
    вокруг базы фонтана.

    Возвращает long-table:
        одна строка = один candidate enhancer-promoter contact
        в координатах относительно базы фонтана.
    """
    hits = enhancer_fountain_hits.copy()
    promoters = promoters_df.copy()
    promoters = harmonize_chrom_style(promoters, hits)
    promoters = ensure_feature_center(promoters)
    promoters_by_chrom = {chrom: g.sort_values('start').reset_index(drop=True) for chrom, g in promoters.groupby('chrom', sort=False)}
    rows = []
    for _, e in tqdm(hits.iterrows(), total=len(hits), desc='Building enhancer-promoter contact table'):
        chrom = str(e['chrom'])
        base = int(e['base_bp'])
        fountain_id = e[fountain_id_col]
        if chrom not in promoters_by_chrom:
            continue
        p = promoters_by_chrom[chrom]
        window_start = base - promoter_window_bp
        window_end = base + promoter_window_bp
        promoter_candidates = p[(p['end'] > window_start) & (p['start'] < window_end)].copy()
        if len(promoter_candidates) == 0:
            continue
        promoter_candidates['promoter_rel_start_kb'] = (promoter_candidates['start'].astype(int) - base) / 1000
        promoter_candidates['promoter_rel_end_kb'] = (promoter_candidates['end'].astype(int) - base) / 1000
        promoter_candidates['promoter_rel_center_kb'] = (promoter_candidates['feature_center'].astype(int) - base) / 1000
        for _, p_row in promoter_candidates.iterrows():
            contact_row = {fountain_id_col: fountain_id, 'chrom': chrom, 'base_bp': base, 'enhancer_feature_id': e.get('enhancer_feature_id', np.nan), 'enhancer_start': int(e['enhancer_start']), 'enhancer_end': int(e['enhancer_end']), 'enhancer_center': int(e['enhancer_center']), 'enhancer_state': e.get('enhancer_state', np.nan), 'enhancer_score': e.get('enhancer_score', np.nan), 'enhancer_distance_to_base_bp': int(e['enhancer_distance_to_base_bp']), 'enhancer_rel_start_kb': float(e['enhancer_rel_start_kb']), 'enhancer_rel_end_kb': float(e['enhancer_rel_end_kb']), 'enhancer_rel_center_kb': float(e['enhancer_rel_center_kb']), 'promoter_feature_id': p_row.get('feature_id', np.nan), 'promoter_start': int(p_row['start']), 'promoter_end': int(p_row['end']), 'promoter_center': int(p_row['feature_center']), 'promoter_state': p_row.get('state', np.nan), 'promoter_score': p_row.get('state_score', np.nan), 'promoter_rel_start_kb': float(p_row['promoter_rel_start_kb']), 'promoter_rel_end_kb': float(p_row['promoter_rel_end_kb']), 'promoter_rel_center_kb': float(p_row['promoter_rel_center_kb']), 'ep_signed_distance_bp': int(p_row['feature_center']) - int(e['enhancer_center']), 'ep_abs_distance_bp': abs(int(p_row['feature_center']) - int(e['enhancer_center'])), 'promoter_window_bp': int(promoter_window_bp)}
            for col in ['r', 'q1_r_pearson', 'Fountain Score', 'a_kb', 'b_kb', 'p_kb']:
                if col in e.index:
                    contact_row[col] = e[col]
            for extra_col in ['gene_id', 'gene_name', 'transcript_id', 'gene_strand', 'nearest_tss', 'distance_to_tss_bp', 'tss_annotation_status']:
                if extra_col in p_row.index:
                    contact_row[f'promoter_{extra_col}'] = p_row[extra_col]
            rows.append(contact_row)
    return pd.DataFrame(rows)


def plot_ep_contacts_on_aggregate_fountain(ep_contacts, aggregate_Z=None, res=5000, plot_window_kb=200, background_vmin=0.5, background_vmax=1.6, background_cmap='coolwarm', draw_symmetric=True, max_contacts_to_draw=50000, enhancer_lw=3.0, promoter_lw=2.0, contact_alpha=0.04, contact_lw=0.0, title=None, output_png=None):
    """
    Рисует:
      1. aggregate_Z как Hi-C background, если передан.
      2. enhancer intervals как полоски на диагонали.
      3. promoter intervals как полоски на диагонали.
      4. enhancer-promoter contact rectangles.

    Координаты — kb относительно базы фонтана.
    """
    df = ep_contacts.copy()
    keep = (df['enhancer_rel_end_kb'] >= -plot_window_kb) & (df['enhancer_rel_start_kb'] <= plot_window_kb) & (df['promoter_rel_end_kb'] >= -plot_window_kb) & (df['promoter_rel_start_kb'] <= plot_window_kb)
    df = df[keep].copy()
    if len(df) == 0:
        raise ValueError('После фильтрации по plot_window_kb не осталось контактов.')
    if len(df) > max_contacts_to_draw:
        df_draw = df.sample(max_contacts_to_draw, random_state=1).copy()
        print(f'Drawing sampled contacts: {len(df_draw)} of {len(df)}')
    else:
        df_draw = df.copy()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    if aggregate_Z is not None:
        Z = np.asarray(aggregate_Z, dtype=float)
        n = Z.shape[0]
        bin_kb = res / 1000
        coords_kb = make_centered_coords(n, bin_kb)
        extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
        im = ax.imshow(Z, origin='upper', extent=extent, aspect='equal', interpolation='nearest', vmin=background_vmin, vmax=background_vmax, cmap=background_cmap)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Observed / expected')
    else:
        ax.set_xlim(-plot_window_kb, plot_window_kb)
        ax.set_ylim(plot_window_kb, -plot_window_kb)
    rectangles = []
    for _, row in df_draw.iterrows():
        e0 = float(row['enhancer_rel_start_kb'])
        e1 = float(row['enhancer_rel_end_kb'])
        p0 = float(row['promoter_rel_start_kb'])
        p1 = float(row['promoter_rel_end_kb'])
        rectangles.append(Rectangle((min(e0, e1), min(p0, p1)), abs(e1 - e0), abs(p1 - p0)))
        if draw_symmetric:
            rectangles.append(Rectangle((min(p0, p1), min(e0, e1)), abs(p1 - p0), abs(e1 - e0)))
    patch_collection = PatchCollection(rectangles, facecolor='black', edgecolor='none', alpha=contact_alpha, linewidth=contact_lw)
    ax.add_collection(patch_collection)
    enh_cols = ['Fountain index', 'enhancer_feature_id', 'enhancer_rel_start_kb', 'enhancer_rel_end_kb']
    enh_unique = df[enh_cols].drop_duplicates().copy()
    enhancer_segments = []
    for _, row in enh_unique.iterrows():
        s = float(row['enhancer_rel_start_kb'])
        e = float(row['enhancer_rel_end_kb'])
        enhancer_segments.append([(s, s), (e, e)])
    enh_lc = LineCollection(enhancer_segments, linewidths=enhancer_lw, alpha=0.85, colors='red', label='Enhancers')
    ax.add_collection(enh_lc)
    prom_cols = ['Fountain index', 'promoter_feature_id', 'promoter_rel_start_kb', 'promoter_rel_end_kb']
    prom_unique = df[prom_cols].drop_duplicates().copy()
    promoter_segments = []
    for _, row in prom_unique.iterrows():
        s = float(row['promoter_rel_start_kb'])
        e = float(row['promoter_rel_end_kb'])
        promoter_segments.append([(s, s), (e, e)])
    prom_lc = LineCollection(promoter_segments, linewidths=promoter_lw, alpha=0.65, colors='blue', label='Promoters')
    ax.add_collection(prom_lc)
    ax.axline((0, 0), slope=1, linestyle='--', linewidth=1, color='black', alpha=0.6)
    ax.axline((0, 0), slope=-1, linestyle=':', linewidth=1, color='black', alpha=0.6)
    ax.axvline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)
    ax.axhline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)
    ax.set_xlim(-plot_window_kb, plot_window_kb)
    ax.set_ylim(plot_window_kb, -plot_window_kb)
    ax.set_xlabel('Distance from fountain base, kb')
    ax.set_ylabel('Distance from fountain base, kb')
    if title is None:
        n_fountains = df['Fountain index'].nunique()
        n_enh = enh_unique.shape[0]
        n_prom = prom_unique.shape[0]
        n_contacts = len(df)
        title = f'Candidate enhancer-promoter contacts near aggregate fountain\n{n_fountains} fountains, {n_enh} enhancer hits, {n_prom} promoter hits, {n_contacts} E-P pairs'
    ax.set_title(title)
    ax.plot([], [], color='red', linewidth=enhancer_lw, label='Enhancers on diagonal')
    ax.plot([], [], color='blue', linewidth=promoter_lw, label='Promoters on diagonal')
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor='black', alpha=0.25, label='E-P contact rectangles'))
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
    return (fig, ax)


def build_ep_contact_density_grid(ep_contacts, plot_window_kb=200, bin_width_kb=5, draw_symmetric=True):
    """
    Строит матрицу плотности candidate E-P контактов.

    H[y_bin, x_bin]:
        x = enhancer/promoter coordinate
        y = promoter/enhancer coordinate
    """
    df = ep_contacts.copy()
    edges = np.arange(-plot_window_kb, plot_window_kb + bin_width_kb, bin_width_kb)
    centers = 0.5 * (edges[:-1] + edges[1:])
    H = np.zeros((len(centers), len(centers)), dtype=float)

    def interval_to_bins(start_kb, end_kb):
        lo = min(start_kb, end_kb)
        hi = max(start_kb, end_kb)
        i0 = np.searchsorted(edges, lo, side='right') - 1
        i1 = np.searchsorted(edges, hi, side='left')
        i0 = max(i0, 0)
        i1 = min(i1, len(centers) - 1)
        if i1 < i0:
            return None
        return (i0, i1)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building E-P density grid'):
        e_bins = interval_to_bins(float(row['enhancer_rel_start_kb']), float(row['enhancer_rel_end_kb']))
        p_bins = interval_to_bins(float(row['promoter_rel_start_kb']), float(row['promoter_rel_end_kb']))
        if e_bins is None or p_bins is None:
            continue
        ex0, ex1 = e_bins
        py0, py1 = p_bins
        H[py0:py1 + 1, ex0:ex1 + 1] += 1
        if draw_symmetric:
            H[ex0:ex1 + 1, py0:py1 + 1] += 1
    return (H, edges, centers)


def plot_ep_contact_density_on_aggregate_fountain(ep_contacts, aggregate_Z=None, res=5000, plot_window_kb=200, bin_width_kb=5, background_vmin=0.5, background_vmax=1.6, background_cmap='coolwarm', density_cmap='Greys', density_alpha=0.45, density_quantile_vmax=0.995, draw_symmetric=True, title=None, output_png=None):
    H, edges, centers = build_ep_contact_density_grid(ep_contacts=ep_contacts, plot_window_kb=plot_window_kb, bin_width_kb=bin_width_kb, draw_symmetric=draw_symmetric)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    if aggregate_Z is not None:
        Z = np.asarray(aggregate_Z, dtype=float)
        n = Z.shape[0]
        coords_kb = make_centered_coords(n, res / 1000)
        extent_bg = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
        im = ax.imshow(Z, origin='upper', extent=extent_bg, aspect='equal', interpolation='nearest', vmin=background_vmin, vmax=background_vmax, cmap=background_cmap)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Observed / expected')
    H_plot = H.copy()
    H_plot[H_plot == 0] = np.nan
    vmax = np.nanquantile(H_plot, density_quantile_vmax)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(H_plot)
    extent_density = [edges[0], edges[-1], edges[-1], edges[0]]
    im2 = ax.imshow(H_plot, origin='upper', extent=extent_density, aspect='equal', interpolation='nearest', cmap=density_cmap, alpha=density_alpha, vmin=0, vmax=vmax)
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.08)
    cbar2.set_label('E-P contact density')
    ax.axline((0, 0), slope=1, linestyle='--', linewidth=1, color='black', alpha=0.6)
    ax.axline((0, 0), slope=-1, linestyle=':', linewidth=1, color='black', alpha=0.6)
    ax.axvline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)
    ax.axhline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)
    ax.set_xlim(-plot_window_kb, plot_window_kb)
    ax.set_ylim(plot_window_kb, -plot_window_kb)
    ax.set_xlabel('Distance from fountain base, kb')
    ax.set_ylabel('Distance from fountain base, kb')
    if title is None:
        title = 'Density of candidate enhancer-promoter contacts near aggregate fountain'
    ax.set_title(title)
    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
    return (fig, ax, H, edges, centers)
# -----------------------------------------------------------------------------
# Oriented promoter / enhancer-promoter / oriented aggregate utilities
# -----------------------------------------------------------------------------

def _first_existing_column(df, candidates, required=True, what='column'):
    """Return first matching column, case-insensitive."""
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        key = str(c).lower()
        if key in lower_to_col:
            return lower_to_col[key]
    if required:
        raise ValueError(
            f'Не нашла {what}. Кандидаты: {candidates}. '
            f'Доступные колонки: {list(df.columns)}'
        )
    return None


def _normalize_strand_value(x):
    """Normalize strand values to '+', '-' or np.nan."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ['+', 'plus', 'forward', 'fwd', 'sense', '1', '+1']:
        return '+'
    if s in ['-', 'minus', 'reverse', 'rev', 'antisense', '-1']:
        return '-'
    return np.nan


def interval_overlap_bp(start1, end1, start2, end2):
    """Overlap length for half-open intervals [start, end)."""
    return max(0, min(int(end1), int(end2)) - max(int(start1), int(start2)))


def interval_distance_bp(start1, end1, start2, end2):
    """Distance between half-open intervals [start, end). Zero if they overlap."""
    start1 = int(start1)
    end1 = int(end1)
    start2 = int(start2)
    end2 = int(end2)
    if end1 <= start2:
        return start2 - end1
    if end2 <= start1:
        return start1 - end2
    return 0


def load_baranasic_promoters_oriented(
    path,
    sheet_name=0,
    sep=None,
    chrom_col=None,
    start_col=None,
    end_col=None,
    strand_col=None,
    promoter_id_col=None,
    gene_name_col=None,
    gene_id_col=None,
    transcript_id_col=None,
):
    """
    Load Baranasic promoter table and standardize columns.

    The output always contains:
        baranasic_promoter_id
        baranasic_chrom
        baranasic_start
        baranasic_end
        baranasic_center
        baranasic_strand
        baranasic_tss
        baranasic_gene_name
        baranasic_gene_id
        baranasic_transcript_id
    """
    path = Path(path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        if sep is None:
            df = pd.read_csv(path, sep=None, engine='python')
        else:
            df = pd.read_csv(path, sep=sep)

    return standardize_baranasic_promoters(
        df,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        strand_col=strand_col,
        promoter_id_col=promoter_id_col,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        transcript_id_col=transcript_id_col,
    )


def standardize_baranasic_promoters(
    baranasic_df,
    chrom_col=None,
    start_col=None,
    end_col=None,
    strand_col=None,
    promoter_id_col=None,
    gene_name_col=None,
    gene_id_col=None,
    transcript_id_col=None,
):
    """
    Standardize a Baranasic promoter table with strand/orientation information.

    You can pass explicit column names, or let the function guess common names.
    """
    df = baranasic_df.copy()

    chrom_col = chrom_col or _first_existing_column(
        df,
        ['chrom', '#chrom', 'Chromosome', 'chromosome', 'chr', 'seqnames', 'seqname'],
        what='колонку хромосомы',
    )
    start_col = start_col or _first_existing_column(
        df,
        ['start', 'Start', 'chromStart', 'promoter_start', 'Promoter start', 'TSS_start'],
        what='колонку start',
    )
    end_col = end_col or _first_existing_column(
        df,
        ['end', 'End', 'chromEnd', 'promoter_end', 'Promoter end', 'TSS_end'],
        what='колонку end',
    )
    strand_col = strand_col or _first_existing_column(
        df,
        ['strand', 'Strand', 'gene_strand', 'Gene strand', 'orientation', 'Orientation', 'direction', 'Direction'],
        what='колонку strand/orientation',
    )

    promoter_id_col = promoter_id_col or _first_existing_column(
        df,
        ['promoter_id', 'Promoter ID', 'promoter', 'id', 'ID', 'cluster_id', 'Cluster ID', 'name', 'Name'],
        required=False,
        what='ID промотора',
    )
    gene_name_col = gene_name_col or _first_existing_column(
        df,
        ['gene_name', 'Gene name', 'gene', 'Gene', 'gene_symbol', 'Gene symbol', 'symbol', 'external_gene_name'],
        required=False,
        what='имя гена',
    )
    gene_id_col = gene_id_col or _first_existing_column(
        df,
        ['gene_id', 'Gene ID', 'geneId', 'ensembl_gene_id', 'Ensembl gene ID'],
        required=False,
        what='gene_id',
    )
    transcript_id_col = transcript_id_col or _first_existing_column(
        df,
        ['transcript_id', 'Transcript ID', 'transcriptId', 'ensembl_transcript_id'],
        required=False,
        what='transcript_id',
    )

    out = df.copy()
    out['baranasic_chrom'] = out[chrom_col].astype(str)
    out['baranasic_start'] = out[start_col].astype(int)
    out['baranasic_end'] = out[end_col].astype(int)
    out['baranasic_center'] = ((out['baranasic_start'] + out['baranasic_end']) // 2).astype(int)
    out['baranasic_strand'] = out[strand_col].apply(_normalize_strand_value)

    if promoter_id_col is not None:
        out['baranasic_promoter_id'] = out[promoter_id_col].astype(str)
    else:
        out['baranasic_promoter_id'] = np.arange(len(out)).astype(str)

    out['baranasic_gene_name'] = out[gene_name_col] if gene_name_col is not None else np.nan
    out['baranasic_gene_id'] = out[gene_id_col] if gene_id_col is not None else np.nan
    out['baranasic_transcript_id'] = out[transcript_id_col] if transcript_id_col is not None else np.nan

    # TSS coordinate from oriented promoter interval.
    out['baranasic_tss'] = np.where(
        out['baranasic_strand'] == '+',
        out['baranasic_start'],
        np.where(out['baranasic_strand'] == '-', out['baranasic_end'], np.nan),
    )

    out = out[np.isfinite(out['baranasic_tss'])].copy()
    out['baranasic_tss'] = out['baranasic_tss'].astype(int)

    return out


def _harmonize_baranasic_chrom_to_reference(baranasic_df, reference_df):
    """Harmonize baranasic_chrom to reference_df['chrom'] style."""
    tmp = baranasic_df.copy()
    tmp['chrom'] = tmp['baranasic_chrom'].astype(str)
    tmp = harmonize_chrom_style(tmp, reference_df, chrom_col='chrom')
    tmp['baranasic_chrom_original'] = baranasic_df['baranasic_chrom'].astype(str).values
    tmp['baranasic_chrom'] = tmp['chrom'].astype(str)
    tmp = tmp.drop(columns=['chrom'])
    return tmp


def match_padre_promoters_to_baranasic(
    padre_promoters,
    baranasic_promoters,
    max_match_distance_bp=0,
    keep_all_matches=False,
    drop_ambiguous_strand=False,
):
    """
    Match PADRE promoter-like intervals to oriented Baranasic promoters.

    Parameters
    ----------
    padre_promoters : DataFrame
        Usually select_chromhmm_features(reg, 'promoter', 'strict').
        Must contain chrom, start, end, state.

    baranasic_promoters : DataFrame
        Output of load_baranasic_promoters_oriented or standardize_baranasic_promoters.

    max_match_distance_bp : int
        0 means true interval overlap is required.
        >0 allows nearest Baranasic promoter within this distance from PADRE interval.

    keep_all_matches : bool
        If False, returns one best Baranasic match per PADRE interval.
        If True, returns all matches within max_match_distance_bp.

    drop_ambiguous_strand : bool
        If True, drops PADRE intervals that match both + and - Baranasic promoters.
        This is strict but useful for orientation-sensitive analyses.

    Returns
    -------
    oriented_padre_promoters : DataFrame
        PADRE promoters with gene_strand and Baranasic promoter metadata.
    """
    padre = ensure_feature_center(padre_promoters.copy())
    padre['chrom'] = padre['chrom'].astype(str)

    if 'baranasic_chrom' not in baranasic_promoters.columns:
        bar = standardize_baranasic_promoters(baranasic_promoters)
    else:
        bar = baranasic_promoters.copy()

    bar = _harmonize_baranasic_chrom_to_reference(bar, padre)

    bar_by_chrom = {
        chrom: g.sort_values('baranasic_start').reset_index(drop=True)
        for chrom, g in bar.groupby('baranasic_chrom', sort=False)
    }

    rows = []

    for _, p in tqdm(padre.iterrows(), total=len(padre), desc='Matching PADRE promoters to Baranasic'):
        chrom = str(p['chrom'])
        p_start = int(p['start'])
        p_end = int(p['end'])
        p_center = int(p['feature_center'])
        p_id = p.get('feature_id', p.name)

        if chrom not in bar_by_chrom:
            continue

        b = bar_by_chrom[chrom]
        candidates = b[
            (b['baranasic_end'] >= p_start - max_match_distance_bp)
            & (b['baranasic_start'] <= p_end + max_match_distance_bp)
        ].copy()

        if len(candidates) == 0:
            continue

        candidates['padre_baranasic_overlap_bp'] = [
            interval_overlap_bp(p_start, p_end, s, e)
            for s, e in zip(candidates['baranasic_start'], candidates['baranasic_end'])
        ]
        candidates['distance_to_padre_interval_bp'] = [
            interval_distance_bp(p_start, p_end, s, e)
            for s, e in zip(candidates['baranasic_start'], candidates['baranasic_end'])
        ]
        candidates['distance_to_padre_center_bp'] = (
            candidates['baranasic_center'].astype(int) - p_center
        ).abs()

        candidates = candidates[
            candidates['distance_to_padre_interval_bp'] <= max_match_distance_bp
        ].copy()

        if len(candidates) == 0:
            continue

        n_matches = len(candidates)
        n_strands = candidates['baranasic_strand'].nunique(dropna=True)

        if drop_ambiguous_strand and n_strands > 1:
            continue

        candidates = candidates.sort_values(
            ['distance_to_padre_interval_bp', 'padre_baranasic_overlap_bp', 'distance_to_padre_center_bp'],
            ascending=[True, False, True],
        ).copy()

        if not keep_all_matches:
            candidates = candidates.iloc[:1].copy()

        for _, c in candidates.iterrows():
            row = p.to_dict()
            row.update({
                'padre_promoter_id': p_id,
                'padre_promoter_start': p_start,
                'padre_promoter_end': p_end,
                'padre_promoter_center': p_center,
                'gene_strand': c['baranasic_strand'],
                'baranasic_promoter_id': c['baranasic_promoter_id'],
                'baranasic_chrom': c['baranasic_chrom'],
                'baranasic_chrom_original': c.get('baranasic_chrom_original', c['baranasic_chrom']),
                'baranasic_start': int(c['baranasic_start']),
                'baranasic_end': int(c['baranasic_end']),
                'baranasic_center': int(c['baranasic_center']),
                'baranasic_tss': int(c['baranasic_tss']),
                'baranasic_gene_name': c.get('baranasic_gene_name', np.nan),
                'baranasic_gene_id': c.get('baranasic_gene_id', np.nan),
                'baranasic_transcript_id': c.get('baranasic_transcript_id', np.nan),
                'distance_to_padre_interval_bp': int(c['distance_to_padre_interval_bp']),
                'distance_to_padre_center_bp': int(c['distance_to_padre_center_bp']),
                'padre_baranasic_overlap_bp': int(c['padre_baranasic_overlap_bp']),
                'n_baranasic_matches_for_padre': int(n_matches),
                'n_baranasic_strands_for_padre': int(n_strands),
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out[out['gene_strand'].isin(['+', '-'])].copy()
    return out


def build_enhancer_fountain_nearest_oriented_promoter_table(
    enhancer_fountain_hits,
    oriented_padre_promoters,
    max_ep_distance_bp=70000,
    fountain_id_col='Fountain index',
    promoter_anchor_col='baranasic_tss',
):
    """
    For every enhancer associated with a fountain, find the nearest oriented PADRE promoter.

    Input enhancer_fountain_hits is the output of find_enhancers_near_fountains.
    Output contains one row per enhancer-fountain hit with a nearest oriented promoter.
    Rows without a promoter within max_ep_distance_bp are dropped.
    """
    enh = enhancer_fountain_hits.copy()
    prom = oriented_padre_promoters.copy()

    if len(enh) == 0:
        return pd.DataFrame()
    if len(prom) == 0:
        return pd.DataFrame()

    prom = ensure_feature_center(prom)
    prom = harmonize_chrom_style(prom, enh, chrom_col='chrom')

    if promoter_anchor_col not in prom.columns:
        promoter_anchor_col = 'feature_center'

    prom_by_chrom = {
        chrom: g.sort_values(promoter_anchor_col).reset_index(drop=True)
        for chrom, g in prom.groupby('chrom', sort=False)
    }

    rows = []

    for _, e in tqdm(enh.iterrows(), total=len(enh), desc='Finding nearest oriented promoter for enhancers'):
        chrom = str(e['chrom'])
        if chrom not in prom_by_chrom:
            continue

        enhancer_center = int(e['enhancer_center'])
        base = int(e['base_bp'])
        fountain_id = e[fountain_id_col]

        p = prom_by_chrom[chrom]
        candidates = p[
            (p[promoter_anchor_col].astype(int) >= enhancer_center - max_ep_distance_bp)
            & (p[promoter_anchor_col].astype(int) <= enhancer_center + max_ep_distance_bp)
        ].copy()

        if len(candidates) == 0:
            continue

        candidates['ep_signed_distance_bp'] = candidates[promoter_anchor_col].astype(int) - enhancer_center
        candidates['ep_abs_distance_bp'] = candidates['ep_signed_distance_bp'].abs()
        candidates['promoter_signed_distance_to_base_bp'] = candidates[promoter_anchor_col].astype(int) - base
        candidates['promoter_abs_distance_to_base_bp'] = candidates['promoter_signed_distance_to_base_bp'].abs()

        candidates = candidates.sort_values(
            ['ep_abs_distance_bp', 'promoter_abs_distance_to_base_bp', 'start', 'end']
        ).copy()

        best = candidates.iloc[0]

        row = e.to_dict()
        row.update({
            'nearest_promoter_feature_id': best.get('feature_id', np.nan),
            'nearest_promoter_state': best.get('state', np.nan),
            'nearest_promoter_start': int(best['start']),
            'nearest_promoter_end': int(best['end']),
            'nearest_promoter_center': int(best['feature_center']),
            'nearest_promoter_anchor_bp': int(best[promoter_anchor_col]),
            'nearest_promoter_anchor_col': promoter_anchor_col,
            'nearest_promoter_rel_start_kb': (int(best['start']) - base) / 1000,
            'nearest_promoter_rel_end_kb': (int(best['end']) - base) / 1000,
            'nearest_promoter_rel_center_kb': (int(best['feature_center']) - base) / 1000,
            'nearest_promoter_rel_anchor_kb': (int(best[promoter_anchor_col]) - base) / 1000,
            'promoter_gene_strand': best['gene_strand'],
            'promoter_baranasic_promoter_id': best.get('baranasic_promoter_id', np.nan),
            'promoter_baranasic_gene_name': best.get('baranasic_gene_name', np.nan),
            'promoter_baranasic_gene_id': best.get('baranasic_gene_id', np.nan),
            'promoter_baranasic_transcript_id': best.get('baranasic_transcript_id', np.nan),
            'ep_signed_distance_bp': int(best['ep_signed_distance_bp']),
            'ep_abs_distance_bp': int(best['ep_abs_distance_bp']),
            'promoter_signed_distance_to_base_bp': int(best['promoter_signed_distance_to_base_bp']),
            'promoter_abs_distance_to_base_bp': int(best['promoter_abs_distance_to_base_bp']),
            'n_oriented_promoters_within_max_ep_distance': int(len(candidates)),
            'max_ep_distance_bp': int(max_ep_distance_bp),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out[out['promoter_gene_strand'].isin(['+', '-'])].copy()
    return out


def select_one_oriented_contact_per_fountain(
    oriented_ep_table,
    fountain_id_col='Fountain index',
    prefer='nearest_ep',
):
    """
    Reduce enhancer-fountain-promoter table to one orientation-defining contact per fountain.

    prefer='nearest_ep' sorts by enhancer-promoter distance first.
    prefer='nearest_enhancer_to_base' sorts by enhancer-base distance first.
    """
    df = oriented_ep_table.copy()
    if len(df) == 0:
        return df

    if prefer == 'nearest_ep':
        sort_cols = ['ep_abs_distance_bp', 'enhancer_distance_to_base_bp', 'promoter_abs_distance_to_base_bp']
    elif prefer == 'nearest_enhancer_to_base':
        sort_cols = ['enhancer_distance_to_base_bp', 'ep_abs_distance_bp', 'promoter_abs_distance_to_base_bp']
    else:
        raise ValueError("prefer должен быть 'nearest_ep' или 'nearest_enhancer_to_base'.")

    existing = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(existing).copy()
    return df.drop_duplicates(subset=[fountain_id_col], keep='first').copy()


def reflect_across_anti_diagonal(Z):
    """
    Reflect a square matrix across the anti-diagonal x + y = 0
    in centered Hi-C coordinates.

    For a matrix Z[row_y, col_x], this maps (x, y) -> (-y, -x).
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f'Z должна быть квадратной 2D-матрицей, получено {Z.shape}')
    return np.flipud(np.fliplr(Z)).T


def orient_fountain_matrix_by_promoter_strand(Z, strand, minus_reflect=True):
    """
    Orient one fountain matrix by promoter strand.

    By default, '-' strand fountains are reflected across the anti-diagonal,
    '+' strand fountains are left as-is.
    """
    if strand == '+':
        return Z.copy()
    if strand == '-':
        return reflect_across_anti_diagonal(Z) if minus_reflect else Z.copy()
    raise ValueError(f'Unknown strand: {strand}')


def build_oriented_aggregate_fountain(
    cool_path,
    fountains,
    oriented_ep_table,
    flank=300000,
    res=5000,
    weight_col='Fountain Score',
    z_transform='oe',
    expected_nproc=1,
    min_weight=0,
    require_full_window=True,
    base_shift_bp=-6000,
    fountain_id_col='Fountain index',
    minus_reflect=True,
    contact_prefer='nearest_ep',
):
    """
    Build an aggregate fountain after orienting matrices by nearest promoter strand.

    Steps:
        1. select one enhancer-promoter orientation-defining contact per fountain;
        2. extract each fountain matrix;
        3. reflect '-' strand fountains across the anti-diagonal;
        4. weighted-average matrices using weight_col.

    Parameters
    ----------
    fountains : str/path or DataFrame
        Fountain CSV path or already loaded fountain table.

    oriented_ep_table : DataFrame
        Output of build_enhancer_fountain_nearest_oriented_promoter_table.
    """
    clr = cooler.Cooler(f'{cool_path}::resolutions/{res}')

    if isinstance(fountains, (str, Path)):
        fountains_df = load_fountains_csv(fountains)
    else:
        fountains_df = prepare_fountains_for_annotation(fountains)

    orientation_contacts = select_one_oriented_contact_per_fountain(
        oriented_ep_table,
        fountain_id_col=fountain_id_col,
        prefer=contact_prefer,
    )

    if len(orientation_contacts) == 0:
        raise ValueError('oriented_ep_table пустая после выбора одного контакта на фонтан.')

    keep_cols = [
        fountain_id_col,
        'promoter_gene_strand',
        'enhancer_feature_id',
        'enhancer_center',
        'enhancer_rel_center_kb',
        'nearest_promoter_feature_id',
        'nearest_promoter_anchor_bp',
        'nearest_promoter_rel_anchor_kb',
        'ep_abs_distance_bp',
        'ep_signed_distance_bp',
        'promoter_baranasic_gene_name',
        'promoter_baranasic_gene_id',
        'promoter_baranasic_promoter_id',
    ]
    keep_cols = [c for c in keep_cols if c in orientation_contacts.columns]
    orientation_contacts = orientation_contacts[keep_cols].copy()

    fountains_df = fountains_df.merge(
        orientation_contacts,
        on=fountain_id_col,
        how='inner',
        validate='one_to_one',
    )

    fountains_df = harmonize_fountain_chroms_to_cooler(fountains_df, clr)

    chroms = sorted(fountains_df['chrom'].unique())
    expected_by_chrom, expected_raw, exp_col = compute_expected_by_chrom(
        clr,
        chroms=chroms,
        smooth=False,
        aggregate_smoothed=False,
        nproc=expected_nproc,
        chunksize=1000000,
    )

    exp_cache = ExpectedMatrixCache(expected_by_chrom)
    matrix_selector = clr.matrix(balance=True)

    expected_n = int(2 * flank / res) + 1

    sum_weighted_Z = None
    sum_weights = None
    n_contributors = None
    used_rows = []
    failed_rows = []

    for _, row in tqdm(fountains_df.iterrows(), total=len(fountains_df), desc='Building oriented aggregate fountain'):
        fountain_index = row[fountain_id_col]
        try:
            strand = row['promoter_gene_strand']
            if strand not in ['+', '-']:
                raise ValueError(f'Unknown promoter strand: {strand}')

            weight = float(row[weight_col]) if weight_col in row.index else 1.0
            if not np.isfinite(weight):
                raise ValueError(f'Non-finite weight: {weight}')
            if weight <= min_weight:
                raise ValueError(f'Weight <= min_weight: {weight}')

            Z, meta = extract_fountain_Z(
                row=row,
                clr=clr,
                matrix_selector=matrix_selector,
                exp_cache=exp_cache,
                flank=flank,
                res=res,
                z_transform=z_transform,
                require_full_window=require_full_window,
                base_shift_bp=base_shift_bp,
            )

            if Z.shape != (expected_n, expected_n):
                raise ValueError(f'Unexpected Z shape: {Z.shape}, expected {(expected_n, expected_n)}')

            Z_oriented = orient_fountain_matrix_by_promoter_strand(
                Z,
                strand=strand,
                minus_reflect=minus_reflect,
            )

            if sum_weighted_Z is None:
                sum_weighted_Z = np.zeros_like(Z_oriented, dtype=float)
                sum_weights = np.zeros_like(Z_oriented, dtype=float)
                n_contributors = np.zeros_like(Z_oriented, dtype=int)

            valid = np.isfinite(Z_oriented)
            if valid.sum() == 0:
                raise ValueError('No finite pixels in oriented Z')

            sum_weighted_Z[valid] += weight * Z_oriented[valid]
            sum_weights[valid] += weight
            n_contributors[valid] += 1

            used = {
                fountain_id_col: fountain_index,
                'chrom': row['chrom'],
                'chrom_original': row.get('chrom_original', row['chrom']),
                'start': int(row['start']),
                'end': int(row['end']),
                'base_bp': int(row['base_bp']),
                'region': meta['region'],
                'weight': weight,
                weight_col: row.get(weight_col, np.nan),
                'promoter_gene_strand': strand,
                'reflected_across_anti_diagonal': bool(strand == '-' and minus_reflect),
                'ep_abs_distance_bp': row.get('ep_abs_distance_bp', np.nan),
                'ep_signed_distance_bp': row.get('ep_signed_distance_bp', np.nan),
                'enhancer_rel_center_kb': row.get('enhancer_rel_center_kb', np.nan),
                'nearest_promoter_rel_anchor_kb': row.get('nearest_promoter_rel_anchor_kb', np.nan),
                'promoter_baranasic_gene_name': row.get('promoter_baranasic_gene_name', np.nan),
            }
            used_rows.append(used)

        except Exception as e:
            failed_rows.append({
                fountain_id_col: fountain_index,
                'chrom': row.get('chrom', np.nan),
                'start': row.get('start', np.nan),
                'end': row.get('end', np.nan),
                'base_bp': row.get('base_bp', np.nan),
                'promoter_gene_strand': row.get('promoter_gene_strand', np.nan),
                'error': repr(e),
            })
            continue

    if sum_weighted_Z is None:
        raise RuntimeError('Не удалось добавить ни одного фонтана в oriented aggregate.')

    aggregate_Z = np.full_like(sum_weighted_Z, np.nan, dtype=float)
    valid_agg = sum_weights > 0
    aggregate_Z[valid_agg] = sum_weighted_Z[valid_agg] / sum_weights[valid_agg]
    np.fill_diagonal(aggregate_Z, np.nan)

    used_df = pd.DataFrame(used_rows)
    failed_df = pd.DataFrame(failed_rows)

    aggregate_info = {
        'n_used': len(used_df),
        'n_failed': len(failed_df),
        'n_plus': int((used_df['promoter_gene_strand'] == '+').sum()) if len(used_df) else 0,
        'n_minus': int((used_df['promoter_gene_strand'] == '-').sum()) if len(used_df) else 0,
        'z_transform': z_transform,
        'weight_col': weight_col,
        'expected_column': exp_col,
        'flank': flank,
        'res': res,
        'n_pixels': expected_n,
        'base_shift_bp': base_shift_bp,
        'minus_reflect': minus_reflect,
        'n_contributors': n_contributors,
        'sum_weights': sum_weights,
    }

    return aggregate_Z, aggregate_info, used_df, failed_df, expected_raw


def plot_oriented_aggregate_fountain(
    aggregate_Z,
    res=5000,
    vmin=0.5,
    vmax=1.6,
    cmap='coolwarm',
    title=None,
    output_png=None,
):
    """Plot oriented aggregate fountain."""
    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]
    coords_kb = make_centered_coords(n, res / 1000)
    extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        Z,
        origin='upper',
        extent=extent,
        aspect='equal',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Observed / expected')

    ax.axline((0, 0), slope=1, linestyle='--', linewidth=1, color='black', alpha=0.6)
    ax.axline((0, 0), slope=-1, linestyle=':', linewidth=1, color='black', alpha=0.6)
    ax.axvline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)
    ax.axhline(0, linestyle='-', linewidth=0.7, color='black', alpha=0.25)

    ax.set_xlabel('Distance from fountain base, kb')
    ax.set_ylabel('Distance from fountain base, kb')

    if title is None:
        title = 'Oriented aggregate fountain'
    ax.set_title(title)

    plt.tight_layout()
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches='tight')

    return fig, ax
    
    
#построение фонтана, выровненного относительно энхансера
def select_nearest_enhancer_per_fountain(
    enhancer_fountain_hits,
    fountain_id_col="Fountain index",
):
    """
    Из long-table fountain-enhancer оставляет один ближайший enhancer
    для каждого фонтана.
    """

    required = [
        fountain_id_col,
        "chrom",
        "base_bp",
        "enhancer_feature_id",
        "enhancer_start",
        "enhancer_end",
        "enhancer_center",
        "enhancer_distance_to_base_bp",
    ]

    missing = [c for c in required if c not in enhancer_fountain_hits.columns]
    if missing:
        raise ValueError(f"В enhancer_fountain_hits не хватает колонок: {missing}")

    sort_cols = [
        fountain_id_col,
        "enhancer_distance_to_base_bp",
        "enhancer_feature_id",
    ]

    nearest = (
        enhancer_fountain_hits
        .drop_duplicates()
        .sort_values(sort_cols)
        .drop_duplicates(fountain_id_col, keep="first")
        .reset_index(drop=True)
    )

    return nearest
    
def extract_Z_around_anchor(
    chrom,
    anchor_bp,
    clr,
    matrix_selector,
    exp_cache,
    flank=300_000,
    res=5_000,
    z_transform="oe",
    require_full_window=True,
):
    """
    Достаёт окно Hi-C вокруг anchor_bp.

    anchor_bp — координата энхансера, по которой выравниваем.
    Центральным делаем bin, в который попал anchor_bp.

    Возвращает:
        Z, meta
    """

    chrom = str(chrom)
    anchor_bp = int(anchor_bp)

    chrom_len = int(clr.chromsizes[chrom])

    # bin, в который попал enhancer center
    anchor_bin_start = (anchor_bp // res) * res

    # делаем окно с нечётным числом бинов:
    # [anchor_bin_start - flank, anchor_bin_start + flank + res)
    start = anchor_bin_start - flank
    end = anchor_bin_start + flank + res

    if start < 0 or end > chrom_len:
        if require_full_window:
            raise ValueError(
                f"Anchor too close to chromosome edge: "
                f"{chrom}:{anchor_bp}, window {start}-{end}, chrom_len={chrom_len}"
            )

        start = max(0, start)
        end = min(chrom_len, end)

        # дополнительно выравниваем на res
        start = (start // res) * res
        end = ((end + res - 1) // res) * res
        end = min(chrom_len, end)

    region = f"{chrom}:{start}-{end}"

    mat = matrix_selector.fetch(region)
    mat = np.asarray(mat, dtype=float)

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Non-square matrix for {region}: {mat.shape}")

    n = mat.shape[0]
    expected_n = int(2 * flank / res) + 1

    if require_full_window and n != expected_n:
        raise ValueError(
            f"Window has unexpected size: n={n}, expected={expected_n}, region={region}"
        )

    exp_mat = exp_cache.get_exp_mat(chrom, n)

    oe = mat / exp_mat
    oe[~np.isfinite(oe)] = np.nan
    np.fill_diagonal(oe, np.nan)

    if z_transform == "oe":
        Z = oe

    elif z_transform == "log_oe":
        Z = np.log2(oe)
        Z[~np.isfinite(Z)] = np.nan
        np.fill_diagonal(Z, np.nan)

    else:
        raise ValueError("z_transform должен быть 'oe' или 'log_oe'.")

    meta = {
        "chrom": chrom,
        "anchor_bp": anchor_bp,
        "anchor_bin_start": anchor_bin_start,
        "anchor_offset_within_bin_bp": anchor_bp - anchor_bin_start,
        "region": region,
        "region_start": start,
        "region_end": end,
        "n": n,
    }

    return Z, meta


def build_enhancer_aligned_aggregate_fountain(
    cool_path,
    nearest_enhancer_per_fountain,
    flank=300_000,
    res=5_000,
    weight_col="Fountain Score",
    z_transform="oe",
    expected_nproc=1,
    min_weight=0,
    require_full_window=True,
    fountain_id_col="Fountain index",
):
    """
    Строит агрегированный Hi-C вокруг энхансеров,
    ассоциированных с фонтанами.

    Одна строка nearest_enhancer_per_fountain = один фонтан + его ближайший enhancer.

    Центр окна = enhancer_center.
    """

    clr = cooler.Cooler(f"{cool_path}::resolutions/{res}")

    anchors = nearest_enhancer_per_fountain.copy()

    if "enhancer_center" not in anchors.columns:
        anchors["enhancer_center"] = (
            anchors["enhancer_start"].astype(int)
            + anchors["enhancer_end"].astype(int)
        ) // 2

    # Приводим chr1 / Chr1 / 1 к стилю mcool
    if True:
        anchors = harmonize_fountain_chroms_to_cooler(
            anchors,
            clr,
            chrom_col="chrom",
        )

    chroms = sorted(anchors["chrom"].astype(str).unique())

    expected_by_chrom, expected_raw, exp_col = compute_expected_by_chrom(
        clr,
        chroms=chroms,
        smooth=False,
        aggregate_smoothed=False,
        nproc=expected_nproc,
        chunksize=1_000_000,
    )

    exp_cache = ExpectedMatrixCache(expected_by_chrom)
    matrix_selector = clr.matrix(balance=True)

    expected_n = int(2 * flank / res) + 1

    sum_weighted_Z = None
    sum_weights = None
    n_contributors = None

    used_rows = []
    failed_rows = []

    for _, row in tqdm(
        anchors.iterrows(),
        total=len(anchors),
        desc="Building enhancer-aligned aggregate",
    ):
        fountain_id = row[fountain_id_col]

        try:
            chrom = str(row["chrom"])
            enhancer_center = int(row["enhancer_center"])

            if weight_col in row.index:
                weight = float(row[weight_col])
            else:
                weight = 1.0

            if not np.isfinite(weight):
                raise ValueError(f"Non-finite weight: {weight}")

            if weight <= min_weight:
                raise ValueError(f"Weight <= min_weight: {weight}")

            Z, meta = extract_Z_around_anchor(
                chrom=chrom,
                anchor_bp=enhancer_center,
                clr=clr,
                matrix_selector=matrix_selector,
                exp_cache=exp_cache,
                flank=flank,
                res=res,
                z_transform=z_transform,
                require_full_window=require_full_window,
            )

            if Z.shape != (expected_n, expected_n):
                raise ValueError(f"Unexpected Z shape: {Z.shape}")

            if sum_weighted_Z is None:
                sum_weighted_Z = np.zeros_like(Z, dtype=float)
                sum_weights = np.zeros_like(Z, dtype=float)
                n_contributors = np.zeros_like(Z, dtype=int)

            valid = np.isfinite(Z)

            if valid.sum() == 0:
                raise ValueError("No finite pixels in Z")

            sum_weighted_Z[valid] += weight * Z[valid]
            sum_weights[valid] += weight
            n_contributors[valid] += 1

            used_rows.append({
                fountain_id_col: fountain_id,
                "chrom": chrom,
                "fountain_base_bp": int(row["base_bp"]) if "base_bp" in row.index else np.nan,
                "enhancer_feature_id": row.get("enhancer_feature_id", np.nan),
                "enhancer_start": int(row["enhancer_start"]),
                "enhancer_end": int(row["enhancer_end"]),
                "enhancer_center": enhancer_center,
                "enhancer_distance_to_base_bp": row.get("enhancer_distance_to_base_bp", np.nan),
                "enhancer_rel_to_fountain_base_bp": (
                    enhancer_center - int(row["base_bp"])
                    if "base_bp" in row.index else np.nan
                ),
                "weight": weight,
                "region": meta["region"],
                "anchor_offset_within_bin_bp": meta["anchor_offset_within_bin_bp"],
            })

        except Exception as e:
            failed_rows.append({
                fountain_id_col: fountain_id,
                "chrom": row.get("chrom", np.nan),
                "base_bp": row.get("base_bp", np.nan),
                "enhancer_feature_id": row.get("enhancer_feature_id", np.nan),
                "enhancer_center": row.get("enhancer_center", np.nan),
                "Fountain Score": row.get(weight_col, np.nan),
                "error": repr(e),
            })

            continue

    if sum_weighted_Z is None:
        raise RuntimeError("Не удалось добавить ни одного enhancer-aligned окна.")

    aggregate_Z = np.full_like(sum_weighted_Z, np.nan, dtype=float)

    valid_agg = sum_weights > 0
    aggregate_Z[valid_agg] = sum_weighted_Z[valid_agg] / sum_weights[valid_agg]

    np.fill_diagonal(aggregate_Z, np.nan)

    used_df = pd.DataFrame(used_rows)
    failed_df = pd.DataFrame(failed_rows)

    aggregate_info = {
        "n_used": len(used_df),
        "n_failed": len(failed_df),
        "z_transform": z_transform,
        "weight_col": weight_col,
        "expected_column": exp_col,
        "flank": flank,
        "res": res,
        "n_pixels": expected_n,
        "alignment": "enhancer_center",
        "sum_weights_total": float(np.nansum(used_df["weight"])) if len(used_df) else 0,
        "n_contributors": n_contributors,
        "sum_weights": sum_weights,
    }

    return aggregate_Z, aggregate_info, used_df, failed_df, expected_raw


def plot_enhancer_aligned_aggregate(
    aggregate_Z,
    res=5_000,
    vmin=0.5,
    vmax=1.6,
    cmap="coolwarm",
    title=None,
    output_png=None,
    dpi=600,
):
    """
    Рисует агрегат, центрированный по энхансеру.
    Ноль на осях = enhancer-centered bin.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    n = Z.shape[0]
    coords_kb = make_centered_coords(n, res / 1000)

    extent = [
        coords_kb.min(),
        coords_kb.max(),
        coords_kb.max(),
        coords_kb.min(),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        Z,
        origin="upper",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.axline((0, 0), slope=1, linestyle="--", linewidth=1, color="black", alpha=0.5)
    ax.axline((0, 0), slope=-1, linestyle=":", linewidth=1, color="black", alpha=0.5)

    ax.axvline(0, linewidth=0.8, color="black", alpha=0.35)
    ax.axhline(0, linewidth=0.8, color="black", alpha=0.35)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)

    ax.tick_params(axis="both", which="major", labelsize=16)

    ax.set_xlabel("Distance from enhancer, kb", fontsize=18)
    ax.set_ylabel("Distance from enhancer, kb", fontsize=18)

    if title is not None:
        ax.set_title(title, fontsize=18)

    plt.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, bbox_inches="tight", dpi=dpi)

    return fig, ax
 
#численное решение

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def gaussian_loading_density(f, mu=0.0, sigma=0.01):
    """
    Нормированная гауссова плотность посадки когезинов rho(f).

    Parameters
    ----------
    f : array-like
        Координата посадки.
    mu : float
        Центр распределения посадки.
    sigma : float
        Стандартное отклонение распределения посадки.

    Returns
    -------
    rho : np.ndarray
        Значения rho(f).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    f = np.asarray(f, dtype=float)

    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-0.5 * ((f - mu) / sigma) ** 2)
    )


def build_kernel_interpolator(l_grid, r_grid, A_kernel, fill_value=0.0):
    """
    Строит интерполятор для численного решения A(l, r).

    Parameters
    ----------
    l_grid, r_grid : array-like
        Сетки по l и r.
    A_kernel : 2D array
        Численное решение A(l, r), например Nfin или A11.
    fill_value : float
        Значение вне области расчёта.

    Returns
    -------
    A_interp : scipy.interpolate.RegularGridInterpolator
    """
    return RegularGridInterpolator(
        (np.asarray(l_grid, dtype=float), np.asarray(r_grid, dtype=float)),
        np.asarray(A_kernel, dtype=float),
        bounds_error=False,
        fill_value=fill_value,
    )


def compute_contact_map_from_kernel(
    A_interp,
    left_grid,
    right_grid,
    rho_mu=0.0,
    rho_sigma=0.01,
    n_int=301,
    normalize=True,
    chunk_size=5000,
):
    """
    Вычисляет контактную карту:

        P(x1, x2) = ∫ A(f - x1, x2 - f) rho(f) df

    для x1 < 0, x2 > 0.

    На выходе карта в координатах:

        left = -x1 >= 0
        right = x2 >= 0

    То есть итоговая картинка — P(-x1, x2).

    Parameters
    ----------
    A_interp : RegularGridInterpolator
        Интерполятор для A(l, r).
    left_grid : array-like
        Сетка по left = -x1.
    right_grid : array-like
        Сетка по right = x2.
    rho_mu : float
        Центр распределения посадки rho(f).
    rho_sigma : float
        Ширина распределения посадки rho(f).
    n_int : int
        Число точек для численного интегрирования по f.
    normalize : bool
        Если True, нормирует P на максимум.
    chunk_size : int
        Размер чанка для векторизации.

    Returns
    -------
    P : 2D np.ndarray
        Контактная карта P(-x1, x2).
    LEFT, RIGHT : 2D np.ndarray
        Координатные сетки.
    """

    left_grid = np.asarray(left_grid, dtype=float)
    right_grid = np.asarray(right_grid, dtype=float)

    RIGHT, LEFT = np.meshgrid(right_grid, left_grid)

    # x1 < 0, x2 > 0
    x1 = -LEFT
    x2 = RIGHT

    span = x2 - x1  # = LEFT + RIGHT

    P = np.zeros_like(span, dtype=float)

    valid = span > 0
    ii, jj = np.where(valid)

    t = np.linspace(0.0, 1.0, n_int)

    for start in range(0, len(ii), chunk_size):
        stop = min(start + chunk_size, len(ii))

        i_chunk = ii[start:stop]
        j_chunk = jj[start:stop]

        left_v = LEFT[i_chunk, j_chunk]
        span_v = span[i_chunk, j_chunk]

        # f runs from x1=-left to x2=right
        f = -left_v[:, None] + span_v[:, None] * t[None, :]

        # l = f - x1 = f + left
        l_vals = span_v[:, None] * t[None, :]

        # r = x2 - f
        r_vals = span_v[:, None] * (1.0 - t[None, :])

        pts = np.column_stack(
            [
                l_vals.ravel(),
                r_vals.ravel(),
            ]
        )

        A_vals = A_interp(pts).reshape(len(i_chunk), n_int)

        rho_vals = gaussian_loading_density(
            f,
            mu=rho_mu,
            sigma=rho_sigma,
        )

        integrand = A_vals * rho_vals

        # df = span * dt
        P_vals = span_v * np.trapz(integrand, t, axis=1)

        P[i_chunk, j_chunk] = P_vals

    if normalize:
        pmax = np.nanmax(P)
        if np.isfinite(pmax) and pmax > 0:
            P = P / pmax

    return P, LEFT, RIGHT


def contact_map_from_kernel_array(
    l_grid,
    r_grid,
    A_kernel,
    left_grid=None,
    right_grid=None,
    xmax=0.25,
    n_map=250,
    rho_mu=0.0,
    rho_sigma=0.01,
    n_int=301,
    normalize=True,
    fill_value=0.0,
):
    """
    Удобная обёртка: из готового A_kernel сразу строит P(-x1, x2).

    Parameters
    ----------
    l_grid, r_grid : array-like
        Сетки численного решения.
    A_kernel : 2D array
        Ядро A(l,r), например Nfin.
    left_grid, right_grid : array-like or None
        Если None, строятся автоматически от 0 до xmax.
    xmax : float
        Максимум по -x1 и x2, если сетки не заданы.
    n_map : int
        Размер сетки контактной карты.
    rho_mu, rho_sigma : float
        Параметры распределения посадки rho(f).
    n_int : int
        Число точек интегрирования.
    normalize : bool
        Нормировать ли P на максимум.

    Returns
    -------
    P, LEFT, RIGHT, A_interp
    """

    if left_grid is None:
        left_grid = np.linspace(0.0, xmax, n_map)

    if right_grid is None:
        right_grid = np.linspace(0.0, xmax, n_map)

    A_interp = build_kernel_interpolator(
        l_grid=l_grid,
        r_grid=r_grid,
        A_kernel=A_kernel,
        fill_value=fill_value,
    )

    P, LEFT, RIGHT = compute_contact_map_from_kernel(
        A_interp=A_interp,
        left_grid=left_grid,
        right_grid=right_grid,
        rho_mu=rho_mu,
        rho_sigma=rho_sigma,
        n_int=n_int,
        normalize=normalize,
    )

    return P, LEFT, RIGHT, A_interp


def plot_contact_map_P(
    P,
    LEFT,
    RIGHT,
    levels=60,
    contour_levels=(0.2, 0.4, 0.6, 0.8),
    contour_smooth_sigma=1.2,
    fill_smooth_sigma=None,
    cmap="coolwarm",
    title=None,
    colorbar_label=r"$P(-x_1, x_2)$",
    figsize=(6, 5.5),
    show_equal_arms=True,
    output_png=None,
    dpi=300,
):
    """
    Рисует контактную карту P(-x1, x2).

    Parameters
    ----------
    P : 2D array
        Контактная карта.
    LEFT, RIGHT : 2D array
        Координатные сетки: LEFT=-x1, RIGHT=x2.
    contour_smooth_sigma : float or None
        Сглаживание только для линий уровня.
    fill_smooth_sigma : float or None
        Сглаживание для заливки contourf.
    """

    P = np.asarray(P, dtype=float)

    if fill_smooth_sigma is not None:
        P_fill = gaussian_filter(P, sigma=fill_smooth_sigma)
    else:
        P_fill = P

    if contour_smooth_sigma is not None:
        P_contour = gaussian_filter(P, sigma=contour_smooth_sigma)
    else:
        P_contour = P

    fig, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(
        RIGHT,
        LEFT,
        P_fill,
        levels=levels,
        cmap=cmap,
    )

    plt.colorbar(cf, ax=ax, label=colorbar_label)

    if contour_levels is not None:
        cs = ax.contour(
            RIGHT,
            LEFT,
            P_contour,
            levels=contour_levels,
            colors="black",
            linewidths=1.2,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    if show_equal_arms:
        max_val = min(np.nanmax(LEFT), np.nanmax(RIGHT))
        ax.plot(
            [0, max_val],
            [0, max_val],
            linestyle="--",
            linewidth=1.4,
            color="black",
            alpha=0.7,
            label=r"$-x_1 = x_2$",
        )
        ax.legend(frameon=False)

    ax.set_xlabel(r"$x_2$")
    ax.set_ylabel(r"$-x_1$")

    if title is None:
        title = r"Contact map $P(-x_1, x_2)$"

    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    return fig, ax
    
    
from scipy.ndimage import gaussian_filter

def build_numeric_fountain_kernel(
    Lmax=1.0,
    N=600,
    alpha=0.2,
    gamma_c0=10.0,
    gamma_c=0.0,
    gamma_b=0.0,
    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",
    kernel="Nfin",
):
    """
    Считает численное решение PDE и возвращает ядро A(l,r),
    которое потом можно использовать для построения контактной карты:

        P(x1,x2) = ∫ A(f-x1, x2-f) rho(f) df

    Parameters
    ----------
    kernel : {"Nfin", "A11", "A11+A1c+Ac1"}
        Какую величину использовать как A_kernel.

    Returns
    -------
    result : dict
        Содержит l_grid, r_grid, A, gamma_field, Nfin, A_kernel.
    """

    l_grid, r_grid, A, gamma_field = solve_first_quadrant_with_gamma_field(
        Lmax=Lmax,
        N=N,
        alpha=alpha,
        gamma_c0=gamma_c0,
        gamma_c=gamma_c,
        gamma_b=gamma_b,
        sigma=sigma,
        sigma2=sigma2,
        bottom_mode=bottom_mode,
    )

    A11 = A[0]
    A1c = A[1]
    Ac1 = A[2]
    A1b = A[3]
    Ab1 = A[4]

    Acc = gamma_field / 6.0 * (A1c + Ac1)

    Abb = gamma_b / (2.0 * (1.0 + alpha)) * (A1b + Ab1)

    Abc = 1.0 / (2.0 * (2.0 + alpha)) * (
        A1c * gamma_b + Ab1 * gamma_field
    )

    Acb = 1.0 / (2.0 * (2.0 + alpha)) * (
        Ac1 * gamma_b + A1b * gamma_field
    )

    Nfin = A1c + Ac1 + A1b + Ab1 + Abb + Acc + Abc + Acb + A11

    if kernel == "Nfin":
        A_kernel = Nfin

    elif kernel == "A11":
        A_kernel = A11

    elif kernel == "A11+A1c+Ac1":
        A_kernel = A11 + A1c + Ac1

    else:
        raise ValueError("kernel must be 'Nfin', 'A11', or 'A11+A1c+Ac1'")

    return {
        "l_grid": l_grid,
        "r_grid": r_grid,
        "A": A,
        "gamma_field": gamma_field,
        "Nfin": Nfin,
        "A_kernel": A_kernel,
        "A11": A11,
        "A1c": A1c,
        "Ac1": Ac1,
        "A1b": A1b,
        "Ab1": Ab1,
        "params": {
            "Lmax": Lmax,
            "N": N,
            "alpha": alpha,
            "gamma_c0": gamma_c0,
            "gamma_c": gamma_c,
            "gamma_b": gamma_b,
            "sigma": sigma,
            "sigma2": sigma2,
            "bottom_mode": bottom_mode,
            "kernel": kernel,
        },
    }
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def _get_fit_param(fit, name):
    """
    Достаёт параметр из fit-словаря.
    Поддерживает варианты:
        a_kb / a
        b_kb / b
        p_kb / p
    """
    if f"{name}_kb" in fit:
        return float(fit[f"{name}_kb"])
    return float(fit[name])


def gaussian_fountain_from_fit_on_grid(gaussian_fit, X, Y):
    """
    Строит 2D Gaussian fountain model на сетке X,Y в kb.

    Используется форма:
        C + A * exp(-((X+Y)^2/a^2 + (X-Y-2p)^2/b^2))

    Максимум верхнего фонтана находится около:
        X = p, Y = -p
    """

    a = _get_fit_param(gaussian_fit, "a")
    b = _get_fit_param(gaussian_fit, "b")
    p = _get_fit_param(gaussian_fit, "p")

    A = float(gaussian_fit.get("A", 1.0))
    C = float(gaussian_fit.get("C", 0.0))

    G = C + A * np.exp(
        -(
            ((X + Y) ** 2) / (a ** 2)
            + ((X - Y - 2.0 * p) ** 2) / (b ** 2)
        )
    )

    return G




def _smooth_array_with_nans(A, sigma=1.0, fill="median"):
    """
    Аккуратное сглаживание массива с NaN.
    NaN временно заменяются, после сглаживания исходная NaN-маска возвращается.
    """
    A = np.asarray(A, dtype=float)
    out = A.copy()

    finite = np.isfinite(out)

    if sigma is None or sigma == 0:
        return out

    if not np.any(finite):
        return out

    if fill == "median":
        fill_value = np.nanmedian(out[finite])
    elif fill == "zero":
        fill_value = 0.0
    else:
        fill_value = float(fill)

    A_fill = out.copy()
    A_fill[~finite] = fill_value

    A_smooth = gaussian_filter(A_fill, sigma=sigma)
    A_smooth[~finite] = np.nan

    return A_smooth


def _filter_contour_levels(Z, levels):
    """
    Оставляет только те уровни, которые реально попадают в диапазон данных.
    """
    if levels is None:
        return None

    levels = np.asarray(levels, dtype=float)

    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    levels = np.sort(levels[(levels > zmin) & (levels < zmax)])
    return levels

def _smooth_model_for_contours(
    P,
    sigma=1.2,
    fill_value=0.0,
    renormalize=True,
):
    """
    Сглаживает модельную карту для contour.
    Важно: NaN не возвращаем обратно, иначе contour будет рваться.
    """
    P = np.asarray(P, dtype=float)

    P_fill = np.nan_to_num(
        P,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )

    if sigma is not None and sigma > 0:
        P_smooth = gaussian_filter(P_fill, sigma=sigma)
    else:
        P_smooth = P_fill.copy()

    if renormalize:
        pmax = np.nanmax(P_smooth)
        if np.isfinite(pmax) and pmax > 0:
            P_smooth = P_smooth / pmax

    return P_smooth


def _pad_rectilinear_grid_for_contours(X, Y, Z, pad=3, value=0.0):
    """
    Добавляет вокруг модельной карты рамку из нулей.
    Это помогает контурам замыкаться, если они доходят до края сетки.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if pad is None or pad <= 0:
        return X, Y, Z

    x_axis = X[0, :]
    y_axis = Y[:, 0]

    dx = np.nanmedian(np.diff(x_axis))
    dy = np.nanmedian(np.diff(y_axis))

    x_left = x_axis[0] - dx * np.arange(pad, 0, -1)
    x_right = x_axis[-1] + dx * np.arange(1, pad + 1)

    y_top = y_axis[0] - dy * np.arange(pad, 0, -1)
    y_bottom = y_axis[-1] + dy * np.arange(1, pad + 1)

    x_pad = np.concatenate([x_left, x_axis, x_right])
    y_pad = np.concatenate([y_top, y_axis, y_bottom])

    X_pad, Y_pad = np.meshgrid(x_pad, y_pad)

    Z_pad = np.pad(
        Z,
        pad_width=pad,
        mode="constant",
        constant_values=value,
    )

    return X_pad, Y_pad, Z_pad


def normalize_for_contours(F, mask=None, q_low=0.05, q_high=0.995, clip=True):
    """
    Приводит поле F к шкале ~[0, 1] по робастным квантилям.
    """
    F = np.asarray(F, dtype=float)

    if mask is None:
        vals = F[np.isfinite(F)]
    else:
        vals = F[np.isfinite(F) & mask]

    if len(vals) == 0:
        return np.full_like(F, np.nan), np.nan, np.nan

    lo = np.nanquantile(vals, q_low)
    hi = np.nanquantile(vals, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(F, np.nan), lo, hi

    Fn = (F - lo) / (hi - lo)

    if clip:
        Fn = np.clip(Fn, 0.0, 1.0)

    return Fn, lo, hi

# ============================================================
# Small utilities
# ============================================================

def make_hic_grid(aggregate_Z, res):
    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]
    bin_kb = res / 1000.0
    coords_kb = (np.arange(n) - (n - 1) / 2.0) * bin_kb

    X_grid, Y_grid = np.meshgrid(coords_kb, coords_kb)

    extent = [
        coords_kb.min(),
        coords_kb.max(),
        coords_kb.max(),
        coords_kb.min(),
    ]

    return X_grid, Y_grid, coords_kb, extent


def smooth_array(A, sigma=1.0, fill_nan="median"):
    A = np.asarray(A, dtype=float)
    finite = np.isfinite(A)

    if fill_nan == "median":
        fill_value = np.nanmedian(A[finite]) if np.any(finite) else 0.0
    elif fill_nan == "zero":
        fill_value = 0.0
    else:
        fill_value = float(fill_nan)

    A_fill = A.copy()
    A_fill[~finite] = fill_value

    if sigma is not None and sigma > 0:
        return gaussian_filter(A_fill, sigma=sigma)

    return A_fill


def normalize_for_contours(F, mask=None, q_low=0.05, q_high=0.995, clip=True):
    F = np.asarray(F, dtype=float)

    if mask is None:
        vals = F[np.isfinite(F)]
    else:
        vals = F[np.isfinite(F) & mask]

    if len(vals) == 0:
        return np.full_like(F, np.nan), np.nan, np.nan

    lo = np.nanquantile(vals, q_low)
    hi = np.nanquantile(vals, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(F, np.nan), lo, hi

    Fn = (F - lo) / (hi - lo)

    if clip:
        Fn = np.clip(Fn, 0.0, 1.0)

    return Fn, lo, hi


def filter_levels_to_data(F, levels):
    levels = np.asarray(levels, dtype=float)

    fmin = np.nanmin(F)
    fmax = np.nanmax(F)

    return np.sort(levels[(levels > fmin) & (levels < fmax)])


def make_norm_mask(X_grid, Y_grid, orientation="upper", norm_window_kb=None):
    if orientation == "upper":
        mask = (X_grid > 0) & (Y_grid < 0)

    elif orientation == "lower":
        mask = (X_grid < 0) & (Y_grid > 0)

    elif orientation == "both":
        mask = ((X_grid > 0) & (Y_grid < 0)) | ((X_grid < 0) & (Y_grid > 0))

    else:
        raise ValueError("orientation должен быть 'upper', 'lower' или 'both'.")

    if norm_window_kb is not None:
        mask = mask & (np.abs(X_grid) <= norm_window_kb) & (np.abs(Y_grid) <= norm_window_kb)

    return mask
	
# ============================================================
# Data preparation
# ============================================================

def prepare_gaussian_contour_model(
    gaussian_fit,
    gaussian_fit_cache,
    n,
    X_grid,
    Y_grid,
):
    if gaussian_fit is None:
        return None

    try:
        if gaussian_fit_cache is not None and hasattr(gaussian_fit_cache, "get"):
            ctx = gaussian_fit_cache.get(n)
            X_gauss = ctx.get("X", X_grid)
            Y_gauss = ctx.get("Y", Y_grid)
        else:
            X_gauss = X_grid
            Y_gauss = Y_grid
    except Exception:
        X_gauss = X_grid
        Y_gauss = Y_grid

    gaussian_model = gaussian_fountain_from_fit_on_grid(
        gaussian_fit=gaussian_fit,
        X=X_gauss,
        Y=Y_gauss,
    )

    return gaussian_model

def prepare_aggregate_contour_data(
    aggregate_Z,
    P,
    LEFT,
    RIGHT,
    res,
    lp_kb=1.0,
    orientation="upper",

    # smoothing
    hic_contour_smooth_sigma=1.0,
    numeric_contour_smooth_sigma=1.2,

    # contour levels
    common_contour_levels=(0.35, 0.50, 0.70, 0.85),
    hic_contour_levels=None,
    numeric_contour_levels=None,
    gaussian_contour_levels=None,
    normalize_contours=True,

    # normalization
    norm_window_kb=None,
    norm_q_low=0.05,
    norm_q_high=0.995,

    # gaussian
    gaussian_fit=None,
    gaussian_fit_cache=None,
):
    """
    Готовит все поля и координаты для отрисовки.
    Здесь нет matplotlib-отрисовки.

    Levels:
    -------
    common_contour_levels:
        Общий fallback для всех типов контуров.

    hic_contour_levels:
        Уровни для Hi-C contours.

    numeric_contour_levels:
        Уровни для numerical model contours.

    gaussian_contour_levels:
        Уровни для Gaussian fit contours.

    Если individual levels не заданы, используется common_contour_levels.
    """

    Z = np.asarray(aggregate_Z, dtype=float)
    P = np.asarray(P, dtype=float)

    X_grid, Y_grid, coords_kb, extent = make_hic_grid(Z, res)
    n = Z.shape[0]

    # ------------------------------------------------------------
    # Resolve levels
    # ------------------------------------------------------------

    if hic_contour_levels is None:
        hic_contour_levels = common_contour_levels

    if numeric_contour_levels is None:
        numeric_contour_levels = common_contour_levels

    if gaussian_contour_levels is None:
        gaussian_contour_levels = common_contour_levels

    hic_contour_levels = np.asarray(hic_contour_levels, dtype=float)
    numeric_contour_levels = np.asarray(numeric_contour_levels, dtype=float)
    gaussian_contour_levels = np.asarray(gaussian_contour_levels, dtype=float)

    # ----------------------------
    # Hi-C contours
    # ----------------------------

    Z_smooth = smooth_array(
        Z,
        sigma=hic_contour_smooth_sigma,
        fill_nan="median",
    )

    norm_mask = make_norm_mask(
        X_grid,
        Y_grid,
        orientation=orientation,
        norm_window_kb=norm_window_kb,
    )

    if normalize_contours:
        Z_contour, z_lo, z_hi = normalize_for_contours(
            Z_smooth,
            mask=norm_mask,
            q_low=norm_q_low,
            q_high=norm_q_high,
        )
        hic_levels = filter_levels_to_data(
            Z_contour,
            hic_contour_levels,
        )
    else:
        Z_contour = Z_smooth
        z_lo, z_hi = np.nan, np.nan
        hic_levels = filter_levels_to_data(
            Z_contour,
            hic_contour_levels,
        )

    # ----------------------------
    # Numerical contours
    # ----------------------------

    P_smooth = smooth_array(
        P,
        sigma=numeric_contour_smooth_sigma,
        fill_nan="zero",
    )

    if normalize_contours:
        P_contour, p_lo, p_hi = normalize_for_contours(
            P_smooth,
            mask=None,
            q_low=norm_q_low,
            q_high=norm_q_high,
        )
        numeric_levels = filter_levels_to_data(
            P_contour,
            numeric_contour_levels,
        )
    else:
        P_contour = P_smooth
        p_lo, p_hi = np.nan, np.nan
        numeric_levels = filter_levels_to_data(
            P_contour,
            numeric_contour_levels,
        )

    numeric_contour_sets = []

    if orientation in ("upper", "both"):
        numeric_contour_sets.append({
            "X": RIGHT * lp_kb,
            "Y": -LEFT * lp_kb,
            "Z": P_contour,
            "levels": numeric_levels,
            "name": "numeric_upper",
        })

    if orientation in ("lower", "both"):
        numeric_contour_sets.append({
            "X": -RIGHT * lp_kb,
            "Y": LEFT * lp_kb,
            "Z": P_contour,
            "levels": numeric_levels,
            "name": "numeric_lower",
        })

    # ----------------------------
    # Gaussian contours
    # ----------------------------

    gaussian_model = prepare_gaussian_contour_model(
        gaussian_fit=gaussian_fit,
        gaussian_fit_cache=gaussian_fit_cache,
        n=n,
        X_grid=X_grid,
        Y_grid=Y_grid,
    )

    if gaussian_model is not None:
        G_smooth = gaussian_model

        if normalize_contours:
            G_contour, g_lo, g_hi = normalize_for_contours(
                G_smooth,
                mask=norm_mask,
                q_low=norm_q_low,
                q_high=norm_q_high,
            )
            gaussian_levels = filter_levels_to_data(
                G_contour,
                gaussian_contour_levels,
            )
        else:
            G_contour = G_smooth
            g_lo, g_hi = np.nan, np.nan
            gaussian_levels = filter_levels_to_data(
                G_contour,
                gaussian_contour_levels,
            )
    else:
        G_contour = None
        gaussian_levels = np.array([])
        g_lo, g_hi = np.nan, np.nan

    return {
        "Z": Z,
        "X_grid": X_grid,
        "Y_grid": Y_grid,
        "extent": extent,

        "Z_smooth": Z_smooth,
        "Z_contour": Z_contour,
        "hic_levels": hic_levels,

        "P_smooth": P_smooth,
        "P_contour": P_contour,
        "numeric_contour_sets": numeric_contour_sets,

        "gaussian_model": gaussian_model,
        "G_contour": G_contour,
        "gaussian_levels": gaussian_levels,

        "requested_levels": {
            "hic_contour_levels": hic_contour_levels,
            "numeric_contour_levels": numeric_contour_levels,
            "gaussian_contour_levels": gaussian_contour_levels,
        },

        "normalization": {
            "z_lo": z_lo,
            "z_hi": z_hi,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "g_lo": g_lo,
            "g_hi": g_hi,
        },
    }
    
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt


def plot_aggregate_with_numeric_contact_contours(
    plot_data,

    image_vmin=0.5,
    image_vmax=1.6,
    image_cmap="coolwarm",

    # optional level overrides at plotting stage
    hic_contour_levels=None,
    numeric_contour_levels=None,
    gaussian_contour_levels=None,

    # title params
    species_name=None,
    gamma_c=None,
    gamma_b=1.8,
    title=None,

    # Hi-C contours
    show_hic_contours=True,
    linecolor_hic="black",
    linewidth_hic=1.3,
    hic_contour_linestyle="solid",
    label_hic_contours=False,

    # numerical contours
    show_numeric_contours=True,
    show_numeric_mirror=True,
    contour_color="white",
    contour_linewidth=1.6,
    contour_linestyle="solid",
    label_numeric_contours=False,

    # Gaussian contours
    show_gaussian_contours=True,
    show_gaussian_mirror=True,
    gaussian_color="yellow",
    gaussian_linewidth=1.5,
    gaussian_linestyle="--",
    label_gaussian_contours=False,

    # legend
    show_legend=True,
    legend_loc="upper right",
    legend_frameon=True,

    show_diagonals=True,
    show_colorbar=True,
    figsize=(6.5, 5.8),
    output_png=None,
    dpi=300,
):
    """
    Только отрисовка.
    Все сглаживания, нормировки и подготовка координат должны быть сделаны заранее.

    Уровни можно задать либо заранее в prepare_aggregate_contour_data,
    либо переопределить здесь через:
        hic_contour_levels
        numeric_contour_levels
        gaussian_contour_levels
    """

    def _valid_levels_for_field(F, levels):
        if levels is None:
            return None

        F = np.asarray(F, dtype=float)
        levels = np.asarray(levels, dtype=float)

        if levels.size == 0:
            return np.array([])

        fmin = np.nanmin(F)
        fmax = np.nanmax(F)

        return np.sort(levels[(levels > fmin) & (levels < fmax)])

    Z = plot_data["Z"]
    X_grid = plot_data["X_grid"]
    Y_grid = plot_data["Y_grid"]
    extent = plot_data["extent"]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        Z,
        origin="upper",
        extent=extent,
        cmap=image_cmap,
        vmin=image_vmin,
        vmax=image_vmax,
        interpolation="nearest",
        aspect="equal",
        zorder=0,
    )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Hi-C intensity", fontsize=11)

    legend_handles = []

    # ----------------------------
    # Hi-C contours
    # ----------------------------

    hic_cs = None

    if hic_contour_levels is None:
        hic_levels_use = plot_data["hic_levels"]
    else:
        hic_levels_use = _valid_levels_for_field(
            plot_data["Z_contour"],
            hic_contour_levels,
        )

    if show_hic_contours and len(hic_levels_use) > 0:
        hic_cs = ax.contour(
            X_grid,
            Y_grid,
            plot_data["Z_contour"],
            levels=hic_levels_use,
            colors=linecolor_hic,
            linewidths=linewidth_hic,
            linestyles=hic_contour_linestyle,
            zorder=10,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=linecolor_hic,
                linewidth=linewidth_hic,
                linestyle=hic_contour_linestyle,
                label="Hi-C contour",
            )
        )

        if label_hic_contours:
            ax.clabel(hic_cs, inline=True, fontsize=8, fmt="%.2f")

    # ----------------------------
    # Numerical contours
    # ----------------------------

    numeric_cs = []
    numeric_cs_mirror = []

    if show_numeric_contours:
        numerical_was_drawn = False

        for item in plot_data["numeric_contour_sets"]:

            X_num = np.asarray(item["X"])
            Y_num = np.asarray(item["Y"])
            Z_num = np.asarray(item["Z"])

            if numeric_contour_levels is None:
                levels_use = item["levels"]
            else:
                levels_use = _valid_levels_for_field(
                    Z_num,
                    numeric_contour_levels,
                )

            if len(levels_use) == 0:
                continue

            cs = ax.contour(
                X_num,
                Y_num,
                Z_num,
                levels=levels_use,
                colors=contour_color,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle,
                zorder=20,
            )

            numeric_cs.append(cs)
            numerical_was_drawn = True

            if label_numeric_contours:
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

            if show_numeric_mirror:
                cs_mirror = ax.contour(
                    Y_num.T,
                    X_num.T,
                    Z_num.T,
                    levels=levels_use,
                    colors=contour_color,
                    linewidths=contour_linewidth,
                    linestyles=contour_linestyle,
                    zorder=20,
                )

                numeric_cs_mirror.append(cs_mirror)

                if label_numeric_contours:
                    ax.clabel(cs_mirror, inline=True, fontsize=8, fmt="%.2f")

        if numerical_was_drawn:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=contour_color,
                    linewidth=contour_linewidth,
                    linestyle=contour_linestyle,
                    label="numerical model fit",
                )
            )

    # ----------------------------
    # Gaussian contours
    # ----------------------------

    gaussian_cs = None
    gaussian_cs_mirror = None

    if (
        show_gaussian_contours
        and plot_data.get("G_contour") is not None
    ):
        G = np.asarray(plot_data["G_contour"])

        if gaussian_contour_levels is None:
            gaussian_levels_use = plot_data["gaussian_levels"]
        else:
            gaussian_levels_use = _valid_levels_for_field(
                G,
                gaussian_contour_levels,
            )

        if len(gaussian_levels_use) > 0:
            gaussian_cs = ax.contour(
                X_grid,
                Y_grid,
                G,
                levels=gaussian_levels_use,
                colors=gaussian_color,
                linewidths=gaussian_linewidth,
                linestyles=gaussian_linestyle,
                zorder=30,
            )

            if show_gaussian_mirror:
                gaussian_cs_mirror = ax.contour(
                    X_grid,
                    Y_grid,
                    G.T,
                    levels=gaussian_levels_use,
                    colors=gaussian_color,
                    linewidths=gaussian_linewidth,
                    linestyles=gaussian_linestyle,
                    zorder=30,
                )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=gaussian_color,
                    linewidth=gaussian_linewidth,
                    linestyle=gaussian_linestyle,
                    label="gaussian fit",
                )
            )

            if label_gaussian_contours:
                ax.clabel(gaussian_cs, inline=True, fontsize=8, fmt="%.2f")
                if gaussian_cs_mirror is not None:
                    ax.clabel(gaussian_cs_mirror, inline=True, fontsize=8, fmt="%.2f")

    # ----------------------------
    # Guides
    # ----------------------------

    if show_diagonals:
        ax.axline(
            (0, 0),
            slope=1,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.45,
            zorder=40,
        )
        ax.axline(
            (0, 0),
            slope=-1,
            linestyle=":",
            linewidth=1,
            color="gray",
            alpha=0.55,
            zorder=40,
        )

    # ----------------------------
    # Title
    # ----------------------------

    if title is None:
        title_parts = []

        if species_name is not None:
            title_parts.append(str(species_name))

        if gamma_c is not None:
            gamma_c = float(gamma_c)
            gamma_b = float(gamma_b)

            rho = (
                ((4.0 + gamma_c) * (12.0 + gamma_c) + 12.0 * gamma_b)
                / (
                    4.0
                    * (4.0 + gamma_c + gamma_b)
                    * (3.0 + gamma_c + 3.0 * gamma_b)
                )
            )

            title_parts.append(rf"$\gamma_c$ = {gamma_c:.2f}")
            title_parts.append(rf"$\rho$ = {rho:.3f}")

        if len(title_parts) > 0:
            title = ", ".join(title_parts)
        else:
            title = "Aggregated fountain with normalized contours"

    ax.set_title(title, fontsize=11)

    # ----------------------------
    # Legend
    # ----------------------------

    if show_legend and len(legend_handles) > 0:
        order = {
            "gaussian fit": 0,
            "numerical model fit": 1,
            "Hi-C contour": 2,
        }

        legend_handles = sorted(
            legend_handles,
            key=lambda h: order.get(h.get_label(), 99),
        )

        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            frameon=legend_frameon,
            fontsize=9,
        )

    ax.set_xlabel("Distance from fountain base, kb", fontsize=12)
    ax.set_ylabel("Distance from fountain base, kb", fontsize=12)

    plt.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    return {
        "fig": fig,
        "ax": ax,
        "hic_contours": hic_cs,
        "numeric_contours": numeric_cs,
        "numeric_contours_mirror": numeric_cs_mirror,
        "gaussian_contours": gaussian_cs,
        "gaussian_contours_mirror": gaussian_cs_mirror,
    }
    
#решатель

def gamma_obstacle_field(l, r, gamma_c0=2.6, gamma_c=2.6, sigma2=0.05):
    """
    gamma(l,r) = gamma_c0 + gamma_c * exp(-(l^2 + r^2) / sigma2^2)

    sigma2 здесь — характерный радиус поля препятствий,
    не квадрат sigma.
    """
    return gamma_c0 + gamma_c * np.exp(-(l**2 + r**2) / sigma2**2)
def gaussian_source(x, sigma):
    """
    exp(-x^2 / sigma^2)

    sigma=0.1 соответствует exp(-100 x^2) = exp(-x^2 / 0.01)
    """
    return np.exp(-(x / sigma) ** 2)

def make_rhs_matrix(alpha=0.2, gamma=2.6, gamma_b=2.0):
    """
    RHS = M @ A

    Порядок переменных:
        0: A11
        1: A1c
        2: Ac1
        3: A1b
        4: Ab1
    """

    a = alpha
    g = gamma
    gb = gamma_b

    M = np.zeros((5, 5), dtype=float)

    # A11
    M[0, 0] = -(1.0 + g + gb)
    M[0, 1] = 1.0
    M[0, 2] = 1.0
    M[0, 3] = a
    M[0, 4] = a

    c_decay = -2.0 * (2.0 + 0.5 * (g + gb))
    b_decay = -2.0 * (1.0 + a + 0.5 * (g + gb))

    # A1c
    M[1, 0] = g
    M[1, 1] = c_decay + g / 3.0 + a * gb / (2.0 + a)
    M[1, 2] = g / 3.0
    M[1, 3] = 0.0
    M[1, 4] = a * g / (2.0 + a)

    # Ac1
    M[2, 0] = g
    M[2, 1] = g / 3.0
    M[2, 2] = c_decay + g / 3.0 + a * gb / (2.0 + a)
    M[2, 3] = a * g / (2.0 + a)
    M[2, 4] = 0.0

    # A1b
    M[3, 0] = gb
    M[3, 1] = 0.0
    M[3, 2] = gb / (2.0 + a)
    M[3, 3] = b_decay + g / (2.0 + a) + a * gb / (1.0 + 2.0 * a)
    M[3, 4] = a * gb / (1.0 + 2.0 * a)

    # Ab1
    M[4, 0] = gb
    M[4, 1] = gb / (2.0 + a)
    M[4, 2] = 0.0
    M[4, 3] = a * gb / (1.0 + 2.0 * a)
    M[4, 4] = b_decay + g / (2.0 + a) + a * gb / (1.0 + 2.0 * a)

    return M


def initialize_boundaries(
    A,
    l_grid,
    r_grid,
    h,
    M,
    sigma=0.1,
    bottom_mode="gamma0_compatible",
):
    """
    Заполняет граничные условия на l=0 и r=0.

    bottom_mode:
        "gamma0_compatible" :
            A11(l,0) = exp(-2l) exp(-l^2 / sigma^2)

            Это даёт аналитически правильную полную первую четверть
            в тесте gamma = gamma_b = 0.

        "same_gaussian" :
            A11(l,0) = exp(-l^2 / sigma^2)

            Симметричный гауссов источник на обеих осях.

        "zero" :
            A11(l,0) = 0 для l > 0.
            Тогда область r < l не имеет физического входа.
    """

    N = len(l_grid) - 1

    # ----------------------------
    # A11 on l = 0
    # ----------------------------

    A[0, 0, :] = gaussian_source(r_grid, sigma=sigma)

    # ----------------------------
    # A11 on r = 0
    # ----------------------------

    if bottom_mode == "gamma0_compatible":
        A[0, :, 0] = np.exp(-2.0 * l_grid) * gaussian_source(l_grid, sigma=sigma)

    elif bottom_mode == "same_gaussian":
        A[0, :, 0] = gaussian_source(l_grid, sigma=sigma)

    elif bottom_mode == "zero":
        A[0, 1:, 0] = 0.0

    else:
        raise ValueError("Unknown bottom_mode")

    A[0, 0, 0] = 1.0

    # ----------------------------
    # Incoming non-A11 components
    # ----------------------------

    # A1c and A1b need l=0 boundary.
    A[1, 0, :] = 0.0
    A[3, 0, :] = 0.0

    # Ac1 and Ab1 need r=0 boundary.
    A[2, :, 0] = 0.0
    A[4, :, 0] = 0.0

    return A

def solve_first_quadrant_with_gamma_field(
    Lmax=3.0,
    N=600,
    alpha=0.2,
    gamma_c0=2.6,
    gamma_c=2.6,
    gamma_b=2.0,
    sigma=0.05,
    sigma2=0.05,
    bottom_mode="gamma0_compatible",
):
    """
    Решает систему на квадрате:
        0 <= l <= Lmax
        0 <= r <= Lmax

    Теперь gamma зависит от координат:

        gamma(l,r) = gamma_c0 + gamma_c * exp(-(l^2+r^2)/sigma2^2)

    gamma_b пока остаётся константой.
    """

    l_grid = np.linspace(0.0, Lmax, N + 1)
    r_grid = np.linspace(0.0, Lmax, N + 1)

    h = l_grid[1] - l_grid[0]

    A = np.zeros((5, N + 1, N + 1), dtype=float)

    # Для граничных условий M фактически не используется,
    # но оставим совместимость со старой функцией.
    M0 = make_rhs_matrix(
        alpha=alpha,
        gamma=gamma_obstacle_field(
            0.0,
            0.0,
            gamma_c0=gamma_c0,
            gamma_c=gamma_c,
            sigma2=sigma2,
        ),
        gamma_b=gamma_b,
    )

    A = initialize_boundaries(
        A=A,
        l_grid=l_grid,
        r_grid=r_grid,
        h=h,
        M=M0,
        sigma=sigma,
        bottom_mode=bottom_mode,
    )

    D = np.diag([
        1.0 / (2.0 * h),  # A11, diagonal characteristic
        1.0 / h,          # A1c, l direction
        1.0 / h,          # Ac1, r direction
        1.0 / h,          # A1b, l direction
        1.0 / h,          # Ab1, r direction
    ])

    gamma_field = np.zeros((N + 1, N + 1), dtype=float)

    for i in range(N + 1):
        for j in range(N + 1):
            gamma_field[i, j] = gamma_obstacle_field(
                l_grid[i],
                r_grid[j],
                gamma_c0=gamma_c0,
                gamma_c=gamma_c,
                sigma2=sigma2,
            )

    for i in range(1, N + 1):
        for j in range(1, N + 1):

            l = l_grid[i]
            r = r_grid[j]

            gamma_ij = gamma_field[i, j]

            M = make_rhs_matrix(
                alpha=alpha,
                gamma=gamma_ij,
                gamma_b=gamma_b,
            )

            K = D - M

            b = np.array([
                A[0, i - 1, j - 1] / (2.0 * h),
                A[1, i - 1, j] / h,
                A[2, i, j - 1] / h,
                A[3, i - 1, j] / h,
                A[4, i, j - 1] / h,
            ])

            A[:, i, j] = np.linalg.solve(K, b)

    return l_grid, r_grid, A, gamma_field


def make_hic_grid_kb(aggregate_Z, res):
    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]
    bin_kb = res / 1000.0
    coords_kb = (np.arange(n) - (n - 1) / 2.0) * bin_kb
    X, Y = np.meshgrid(coords_kb, coords_kb)
    extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
    return X, Y, coords_kb, extent


def robust01(A, mask=None, q_low=0.05, q_high=0.995):
    """
    Робастно переводит карту в 0..1.
    """
    A = np.asarray(A, dtype=float)

    if mask is None:
        vals = A[np.isfinite(A)]
    else:
        vals = A[np.isfinite(A) & mask]

    if len(vals) == 0:
        return np.full_like(A, np.nan)

    lo = np.nanquantile(vals, q_low)
    hi = np.nanquantile(vals, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(A, np.nan)

    out = (A - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out


def trimmed_rmse(a, b, trim_q=0.90):
    """
    RMSE после выбрасывания самых больших residuals.
    Это делает фит устойчивее к шумным пикселям.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    r = a[mask] - b[mask]

    if len(r) == 0:
        return np.inf

    cutoff = np.nanquantile(np.abs(r), trim_q)
    r = r[np.abs(r) <= cutoff]

    if len(r) == 0:
        return np.inf

    return float(np.sqrt(np.mean(r**2)))
	
	
def interpolate_P_to_hic_grid(
    P,
    LEFT,
    RIGHT,
    X,
    Y,
    lp_kb,
    orientation="upper",
):
    """
    Переносит P(LEFT, RIGHT) на координатную сетку Hi-C карты.

    upper:
        left arm  = -Y
        right arm = X

    lower:
        left arm  = Y
        right arm = -X
    """

    P = np.asarray(P, dtype=float)

    left_axis_kb = LEFT[:, 0] * lp_kb
    right_axis_kb = RIGHT[0, :] * lp_kb

    P_use = P.copy()

    if np.any(np.diff(left_axis_kb) < 0):
        left_axis_kb = left_axis_kb[::-1]
        P_use = P_use[::-1, :]

    if np.any(np.diff(right_axis_kb) < 0):
        right_axis_kb = right_axis_kb[::-1]
        P_use = P_use[:, ::-1]

    P_use = np.nan_to_num(P_use, nan=0.0, posinf=0.0, neginf=0.0)

    interp = RegularGridInterpolator(
        (left_axis_kb, right_axis_kb),
        P_use,
        bounds_error=False,
        fill_value=np.nan,
    )

    P_hic = np.full_like(X, np.nan, dtype=float)

    if orientation == "upper":
        mask = (X >= 0) & (Y <= 0)
        left = -Y
        right = X

    elif orientation == "lower":
        mask = (X <= 0) & (Y >= 0)
        left = Y
        right = -X

    else:
        raise ValueError("orientation пока лучше использовать 'upper' или 'lower'.")

    pts = np.column_stack([left[mask], right[mask]])
    P_hic[mask] = interp(pts)

    return P_hic

def compute_model_P_hic_for_params(
    aggregate_Z,
    res,
    lp_kb,
    gamma_c,
    rho_sigma,

    flank_kb=None,
    orientation="upper",

    # fixed PDE params
    alpha=0.0,
    gamma_c0=15.0,
    gamma_b=1.8,
    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",

    # numerical params
    N=300,
    n_map=180,
    n_int=201,
):
    """
    Считает численную модель и возвращает её уже на Hi-C сетке.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    X, Y, coords_kb, extent = make_hic_grid_kb(Z, res)

    if flank_kb is None:
        flank_kb = float(np.max(np.abs(coords_kb)))

    xmax = flank_kb / lp_kb
    Lmax = 2.2 * xmax

    kernel_res = ft.build_numeric_fountain_kernel(
        Lmax=Lmax,
        N=N,

        alpha=alpha,

        gamma_c0=gamma_c0,
        gamma_c=gamma_c,
        gamma_b=gamma_b,

        sigma=sigma,
        sigma2=sigma2,

        bottom_mode=bottom_mode,
        kernel="Nfin",
    )

    A_kernel = kernel_res["Nfin"]

    P, LEFT, RIGHT, A_interp = ft.contact_map_from_kernel_array(
        l_grid=kernel_res["l_grid"],
        r_grid=kernel_res["r_grid"],
        A_kernel=A_kernel,

        xmax=xmax,
        n_map=n_map,

        rho_mu=0.0,
        rho_sigma=rho_sigma,

        n_int=n_int,
        normalize=True,
    )

    P_hic = interpolate_P_to_hic_grid(
        P=P,
        LEFT=LEFT,
        RIGHT=RIGHT,
        X=X,
        Y=Y,
        lp_kb=lp_kb,
        orientation=orientation,
    )

    return {
        "P_hic": P_hic,
        "P": P,
        "LEFT": LEFT,
        "RIGHT": RIGHT,
        "X": X,
        "Y": Y,
        "extent": extent,
        "kernel_res": kernel_res,
    }
	
	
def fit_gamma_c_and_rho_sigma_grid(
    aggregate_Z,
    res,
    lp_kb,

    gamma_c_grid,
    rho_sigma_grid,

    flank_kb=None,
    orientation="upper",

    gaussian_fit=None,
    peak_xy_kb=None,
    peak_half_window_kb=50,
    peak_mask_shape="square",

    # fixed PDE params
    alpha=0.0,
    gamma_c0=15.0,
    gamma_b=1.8,
    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",

    # numerical params for fit
    N=300,
    n_map=180,
    n_int=201,

    # normalization / loss
    q_low=0.05,
    q_high=0.995,
    trim_q=0.90,

    verbose=True,
  ):
    """
    Максимально простой и робастный фит:
    Hi-C -> 0..1,
    model -> 0..1,
    score = trimmed RMSE.

    Фитятся только:
        gamma_c
        rho_sigma
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    X, Y, coords_kb, extent = make_hic_grid_kb(Z, res)
    if flank_kb is None:
      flank_kb = float(np.max(np.abs(coords_kb)))

    peak_x_kb, peak_y_kb = get_peak_xy_from_gaussian_or_data(
      aggregate_Z=Z,
      X=X,
      Y=Y,
      orientation=orientation,
      gaussian_fit=gaussian_fit,
      peak_xy_kb=peak_xy_kb,
      smooth_sigma=1.0,
    )

    fit_mask = make_peak_local_fit_mask(
      aggregate_Z=Z,
      X=X,
      Y=Y,
      peak_x_kb=peak_x_kb,
      peak_y_kb=peak_y_kb,
      orientation=orientation,
      peak_half_window_kb=peak_half_window_kb,
      mask_shape=peak_mask_shape,
    )

    print(
      f"Fitting local peak region: "
      f"peak=({peak_x_kb:.1f}, {peak_y_kb:.1f}) kb, "
      f"half-window={peak_half_window_kb} kb, "
      f"n_pixels={fit_mask.sum()}"
    )

    if fit_mask.sum() < 10:
      raise ValueError(
        f"Слишком мало пикселей в локальной области фита: {fit_mask.sum()}. "
        f"Увеличь peak_half_window_kb."
      )

    Z01 = robust01(
      Z,
      mask=fit_mask,
      q_low=q_low,
      q_high=q_high,
    )

    results = []
    best = None
    best_pack = None

    total = len(gamma_c_grid) * len(rho_sigma_grid)
    counter = 0

    for gamma_c in gamma_c_grid:
        for rho_sigma in rho_sigma_grid:
            counter += 1

            try:
                pack = compute_model_P_hic_for_params(
                    aggregate_Z=Z,
                    res=res,
                    lp_kb=lp_kb,
                    gamma_c=float(gamma_c),
                    rho_sigma=float(rho_sigma),

                    flank_kb=flank_kb,
                    orientation=orientation,

                    alpha=alpha,
                    gamma_c0=gamma_c0,
                    gamma_b=gamma_b,
                    sigma=sigma,
                    sigma2=sigma2,
                    bottom_mode=bottom_mode,

                    N=N,
                    n_map=n_map,
                    n_int=n_int,
                )

                P_hic = pack["P_hic"]

                P01 = robust01(
                    P_hic,
                    mask=fit_mask,
                    q_low=q_low,
                    q_high=q_high,
                )

                score = trimmed_rmse(
                    Z01[fit_mask],
                    P01[fit_mask],
                    trim_q=trim_q,
                )

                row = {
                    "gamma_c": float(gamma_c),
                    "rho_sigma": float(rho_sigma),
                    "rho_sigma_kb": float(rho_sigma * lp_kb),
                    "score": float(score),
                    "ok": True,
                }

                if best is None or score < best["score"]:
                    best = row.copy()
                    best_pack = {
                        "Z01": Z01,
                        "P01": P01,
                        "P_hic": P_hic,
                        "fit_mask": fit_mask,
                        **pack,
                    }

            except Exception as e:
                row = {
                    "gamma_c": float(gamma_c),
                    "rho_sigma": float(rho_sigma),
                    "rho_sigma_kb": float(rho_sigma * lp_kb),
                    "score": np.inf,
                    "ok": False,
                    "error": str(e),
                }

            results.append(row)

            if verbose:
                print(
                    f"[{counter:03d}/{total}] "
                    f"gamma_c={gamma_c:.4g}, "
                    f"rho_sigma={rho_sigma:.4g}, "
                    f"score={row['score']:.4g}"
                )

    results_df = pd.DataFrame(results).sort_values("score").reset_index(drop=True)

    return best, results_df, best_pack
	
def get_peak_xy_from_gaussian_or_data(
    aggregate_Z,
    X,
    Y,
    orientation="upper",
    gaussian_fit=None,
    peak_xy_kb=None,
    smooth_sigma=1.0,
):
    """
    Возвращает координаты пика в kb.

    Приоритет:
    1. peak_xy_kb, если передан руками;
    2. максимум gaussian_fit;
    3. максимум сглаженной Hi-C карты.
    """

    if peak_xy_kb is not None:
        return float(peak_xy_kb[0]), float(peak_xy_kb[1])

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("Для локального фита лучше использовать orientation='upper' или 'lower'.")

    if gaussian_fit is not None:
        peak_field = gaussian_fountain_from_fit_on_grid(
            gaussian_fit=gaussian_fit,
            X=X,
            Y=Y,
        )
    else:
        Z = np.asarray(aggregate_Z, dtype=float)
        Z_fill = Z.copy()
        finite = np.isfinite(Z_fill)
        Z_fill[~finite] = np.nanmedian(Z_fill[finite])
        peak_field = gaussian_filter(Z_fill, sigma=smooth_sigma)

    field = np.where(orient_mask & np.isfinite(peak_field), peak_field, np.nan)

    if not np.any(np.isfinite(field)):
        raise ValueError("Не удалось найти пик: нет finite значений в выбранной области.")

    i, j = np.unravel_index(np.nanargmax(field), field.shape)

    return float(X[i, j]), float(Y[i, j])
	
def make_peak_local_fit_mask(
    aggregate_Z,
    X,
    Y,
    peak_x_kb,
    peak_y_kb,
    orientation="upper",
    peak_half_window_kb=50,
    mask_shape="square",
):
    """
    Маска области фита вокруг пика.

    mask_shape:
        "square" — квадрат ±peak_half_window_kb;
        "circle" — круг радиуса peak_half_window_kb.
    """

    Z = np.asarray(aggregate_Z, dtype=float)

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("Для локального фита лучше использовать orientation='upper' или 'lower'.")

    if mask_shape == "square":
        local_mask = (
            (np.abs(X - peak_x_kb) <= peak_half_window_kb) &
            (np.abs(Y - peak_y_kb) <= peak_half_window_kb)
        )

    elif mask_shape == "circle":
        local_mask = (
            (X - peak_x_kb) ** 2 +
            (Y - peak_y_kb) ** 2
        ) <= peak_half_window_kb ** 2

    else:
        raise ValueError("mask_shape должен быть 'square' или 'circle'.")

    fit_mask = (
        orient_mask &
        local_mask &
        np.isfinite(Z)
    )

    return fit_mask

def make_hic_grid_kb(aggregate_Z, res):
    n = aggregate_Z.shape[0]
    bin_kb = res / 1000.0
    coords_kb = (np.arange(n) - (n - 1) / 2.0) * bin_kb
    X, Y = np.meshgrid(coords_kb, coords_kb)
    extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
    return X, Y, coords_kb, extent


def robust01_values(v, q_low=0.05, q_high=0.995):
    v = np.asarray(v, dtype=float)
    vals = v[np.isfinite(v)]

    if len(vals) == 0:
        return np.full_like(v, np.nan)

    lo = np.nanquantile(vals, q_low)
    hi = np.nanquantile(vals, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(v, np.nan)

    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def trimmed_rmse(a, b, trim_q=0.90):
    mask = np.isfinite(a) & np.isfinite(b)
    r = np.asarray(a)[mask] - np.asarray(b)[mask]

    if len(r) == 0:
        return np.inf

    cutoff = np.nanquantile(np.abs(r), trim_q)
    r = r[np.abs(r) <= cutoff]

    if len(r) == 0:
        return np.inf

    return float(np.sqrt(np.mean(r**2)))
	
def get_peak_xy_from_gaussian_or_data(
    aggregate_Z,
    X,
    Y,
    orientation="upper",
    gaussian_fit=None,
    peak_xy_kb=None,
    smooth_sigma=1.0,
):
    if peak_xy_kb is not None:
        return float(peak_xy_kb[0]), float(peak_xy_kb[1])

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("Для быстрого локального фита используй orientation='upper' или 'lower'.")

    if gaussian_fit is not None:
        field = gaussian_fountain_from_fit_on_grid(
            gaussian_fit=gaussian_fit,
            X=X,
            Y=Y,
        )
    else:
        Z = np.asarray(aggregate_Z, dtype=float)
        Z_fill = Z.copy()
        finite = np.isfinite(Z_fill)
        Z_fill[~finite] = np.nanmedian(Z_fill[finite])
        field = gaussian_filter(Z_fill, sigma=smooth_sigma)

    field = np.where(orient_mask & np.isfinite(field), field, np.nan)

    i, j = np.unravel_index(np.nanargmax(field), field.shape)
    return float(X[i, j]), float(Y[i, j])


def make_local_peak_mask(
    aggregate_Z,
    X,
    Y,
    peak_x_kb,
    peak_y_kb,
    orientation="upper",
    peak_half_window_kb=50,
):
    Z = np.asarray(aggregate_Z, dtype=float)

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("orientation должен быть 'upper' или 'lower'.")

    local_mask = (
        (np.abs(X - peak_x_kb) <= peak_half_window_kb) &
        (np.abs(Y - peak_y_kb) <= peak_half_window_kb)
    )

    return orient_mask & local_mask & np.isfinite(Z)
	
def local_contact_values_from_kernel(
    l_grid,
    r_grid,
    A_kernel,
    Xv_kb,
    Yv_kb,
    lp_kb,
    rho_sigma,
    orientation="upper",
    n_int=101,
):
    """
    Считает модель только в выбранных Hi-C пикселях.

    rho_sigma — в безразмерных координатах PDE.
    """

    A_kernel = np.asarray(A_kernel, dtype=float)
    A_kernel = np.nan_to_num(A_kernel, nan=0.0, posinf=0.0, neginf=0.0)

    amax = np.nanmax(A_kernel)
    if np.isfinite(amax) and amax > 0:
        A_kernel = A_kernel / amax

    A_interp = RegularGridInterpolator(
        (l_grid, r_grid),
        A_kernel,
        bounds_error=False,
        fill_value=0.0,
    )

    if orientation == "upper":
        x1 = Yv_kb / lp_kb
        x2 = Xv_kb / lp_kb
    elif orientation == "lower":
        x1 = Xv_kb / lp_kb
        x2 = Yv_kb / lp_kb
    else:
        raise ValueError("orientation должен быть 'upper' или 'lower'.")

    span = x2 - x1

    t = np.linspace(0.0, 1.0, n_int)

    l = span[:, None] * t[None, :]
    r = span[:, None] * (1.0 - t[None, :])
    f = x1[:, None] + span[:, None] * t[None, :]

    pts = np.column_stack([l.ravel(), r.ravel()])
    Aval = A_interp(pts).reshape(len(span), n_int)

    rho = (
        np.exp(-0.5 * (f / rho_sigma) ** 2)
        / (np.sqrt(2.0 * np.pi) * rho_sigma)
    )

    integrand = Aval * rho

    Pvals = span * np.trapz(integrand, t, axis=1)

    return Pvals

from scipy.optimize import minimize
def fit_gamma_c_rho_sigma_fast(
    aggregate_Z,
    res,
    lp_kb,

    gamma_c0=15.0,
    gamma_b=1.8,
    alpha=0.0,

    gaussian_fit=None,
    peak_xy_kb=None,
    peak_half_window_kb=50,
    orientation="upper",

    gamma_c_init=5.0,
    rho_sigma_init=0.01,

    gamma_c_bounds=(0.0, 50.0),
    rho_sigma_bounds=(0.003, 0.06),

    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",

    N=220,
    n_int=101,

    q_low=0.05,
    q_high=0.995,
    trim_q=0.90,

    maxfev=60,
    verbose=True,
):
    """
    Быстрый локальный фит:
    - фитим только область вокруг пика;
    - не строим полную contact map на каждом шаге;
    - оптимизируем gamma_c и log(rho_sigma).
    """

    Z = np.asarray(aggregate_Z, dtype=float)
    X, Y, coords_kb, extent = make_hic_grid_kb(Z, res)

    peak_x_kb, peak_y_kb = get_peak_xy_from_gaussian_or_data(
        aggregate_Z=Z,
        X=X,
        Y=Y,
        orientation=orientation,
        gaussian_fit=gaussian_fit,
        peak_xy_kb=peak_xy_kb,
    )

    fit_mask = make_local_peak_mask(
        aggregate_Z=Z,
        X=X,
        Y=Y,
        peak_x_kb=peak_x_kb,
        peak_y_kb=peak_y_kb,
        orientation=orientation,
        peak_half_window_kb=peak_half_window_kb,
    )

    if fit_mask.sum() < 10:
        raise ValueError(
            f"Слишком мало пикселей в области фита: {fit_mask.sum()}. "
            f"Увеличь peak_half_window_kb."
        )

    Xv = X[fit_mask]
    Yv = Y[fit_mask]
    Zv = Z[fit_mask]

    Z01 = robust01_values(Zv, q_low=q_low, q_high=q_high)

    # Lmax нужен только для локальной области, а не для всего окна
    if orientation == "upper":
        x1_dim = Yv / lp_kb
        x2_dim = Xv / lp_kb
    else:
        x1_dim = Xv / lp_kb
        x2_dim = Yv / lp_kb

    max_span_dim = np.nanmax(x2_dim - x1_dim)
    Lmax = max(1.15 * max_span_dim, 0.02)

    history = []

    def objective(theta):
        gamma_c = float(theta[0])
        rho_sigma = float(np.exp(theta[1]))

        if (
            gamma_c < gamma_c_bounds[0]
            or gamma_c > gamma_c_bounds[1]
            or rho_sigma < rho_sigma_bounds[0]
            or rho_sigma > rho_sigma_bounds[1]
        ):
            return 1e6

        try:
            kernel_res = ft.build_numeric_fountain_kernel(
                Lmax=Lmax,
                N=N,
                alpha=alpha,

                gamma_c0=gamma_c0,
                gamma_c=gamma_c,
                gamma_b=gamma_b,

                sigma=sigma,
                sigma2=sigma2,

                bottom_mode=bottom_mode,
                kernel="Nfin",
            )

            Pvals = local_contact_values_from_kernel(
                l_grid=kernel_res["l_grid"],
                r_grid=kernel_res["r_grid"],
                A_kernel=kernel_res["Nfin"],
                Xv_kb=Xv,
                Yv_kb=Yv,
                lp_kb=lp_kb,
                rho_sigma=rho_sigma,
                orientation=orientation,
                n_int=n_int,
            )

            P01 = robust01_values(Pvals, q_low=q_low, q_high=q_high)

            score = trimmed_rmse(Z01, P01, trim_q=trim_q)

        except Exception:
            score = 1e6

        history.append({
            "gamma_c": gamma_c,
            "rho_sigma": rho_sigma,
            "rho_sigma_kb": rho_sigma * lp_kb,
            "score": score,
        })

        if verbose:
            print(
                f"gamma_c={gamma_c:.4g}, "
                f"rho_sigma={rho_sigma:.5g} "
                f"({rho_sigma * lp_kb:.1f} kb), "
                f"score={score:.4g}"
            )

        return score

    x0 = np.array([
        gamma_c_init,
        np.log(rho_sigma_init),
    ])

    opt = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=[
            gamma_c_bounds,
            (np.log(rho_sigma_bounds[0]), np.log(rho_sigma_bounds[1])),
        ],
        options={
            "maxfev": maxfev,
            "xtol": 1e-2,
            "ftol": 1e-3,
            "disp": verbose,
        },
    )

    gamma_c_best = float(opt.x[0])
    rho_sigma_best = float(np.exp(opt.x[1]))

    history_df = pd.DataFrame(history).sort_values("score").reset_index(drop=True)

    best = {
        "gamma_c": gamma_c_best,
        "rho_sigma": rho_sigma_best,
        "rho_sigma_kb": rho_sigma_best * lp_kb,
        "score": float(opt.fun),
        "peak_x_kb": peak_x_kb,
        "peak_y_kb": peak_y_kb,
        "peak_half_window_kb": peak_half_window_kb,
        "n_fit_pixels": int(fit_mask.sum()),
        "Lmax": Lmax,
        "success": bool(opt.success),
        "message": opt.message,
    }

    pack = {
        "X": X,
        "Y": Y,
        "extent": extent,
        "fit_mask": fit_mask,
        "Z01_local": Z01,
        "history": history_df,
        "opt": opt,
    }

    return best, history_df, pack



def make_hic_grid_kb(aggregate_Z, res):
    n = aggregate_Z.shape[0]
    bin_kb = res / 1000.0
    coords_kb = (np.arange(n) - (n - 1) / 2.0) * bin_kb
    X, Y = np.meshgrid(coords_kb, coords_kb)
    extent = [coords_kb.min(), coords_kb.max(), coords_kb.max(), coords_kb.min()]
    return X, Y, coords_kb, extent


def robust01_values(v, q_low=0.05, q_high=0.995):
    v = np.asarray(v, dtype=float)
    vals = v[np.isfinite(v)]

    if len(vals) < 5:
        return np.full_like(v, np.nan)

    lo = np.nanquantile(vals, q_low)
    hi = np.nanquantile(vals, q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(v, np.nan)

    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def get_peak_xy_from_gaussian_or_data(
    aggregate_Z,
    X,
    Y,
    orientation="upper",
    gaussian_fit=None,
    peak_xy_kb=None,
    smooth_sigma=1.0,
):
    if peak_xy_kb is not None:
        return float(peak_xy_kb[0]), float(peak_xy_kb[1])

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("orientation должен быть 'upper' или 'lower'.")

    if gaussian_fit is not None:
        field = gaussian_fountain_from_fit_on_grid(
            gaussian_fit=gaussian_fit,
            X=X,
            Y=Y,
        )
    else:
        Z = np.asarray(aggregate_Z, dtype=float)
        Z_fill = Z.copy()
        finite = np.isfinite(Z_fill)
        Z_fill[~finite] = np.nanmedian(Z_fill[finite])
        field = gaussian_filter(Z_fill, sigma=smooth_sigma)

    field = np.where(orient_mask & np.isfinite(field), field, np.nan)
    i, j = np.unravel_index(np.nanargmax(field), field.shape)

    return float(X[i, j]), float(Y[i, j])


def make_local_peak_mask(
    aggregate_Z,
    X,
    Y,
    peak_x_kb,
    peak_y_kb,
    orientation="upper",
    peak_half_window_kb=50,
):
    Z = np.asarray(aggregate_Z, dtype=float)

    if orientation == "upper":
        orient_mask = (X >= 0) & (Y <= 0)
    elif orientation == "lower":
        orient_mask = (X <= 0) & (Y >= 0)
    else:
        raise ValueError("orientation должен быть 'upper' или 'lower'.")

    local_mask = (
        (np.abs(X - peak_x_kb) <= peak_half_window_kb) &
        (np.abs(Y - peak_y_kb) <= peak_half_window_kb)
    )

    return orient_mask & local_mask & np.isfinite(Z)
	
	
def local_contact_values_from_kernel(
    l_grid,
    r_grid,
    A_kernel,
    Xv_kb,
    Yv_kb,
    lp_kb,
    rho_sigma,
    orientation="upper",
    n_int=101,
):
    """
    Считает P только в выбранных пикселях Hi-C.

    rho_sigma — безразмерная ширина rho в координатах PDE.
    """

    A_kernel = np.asarray(A_kernel, dtype=float)
    A_kernel = np.nan_to_num(A_kernel, nan=0.0, posinf=0.0, neginf=0.0)

    amax = np.nanmax(A_kernel)
    if np.isfinite(amax) and amax > 0:
        A_kernel = A_kernel / amax

    A_interp = RegularGridInterpolator(
        (l_grid, r_grid),
        A_kernel,
        bounds_error=False,
        fill_value=0.0,
    )

    if orientation == "upper":
        x1 = Yv_kb / lp_kb
        x2 = Xv_kb / lp_kb
    elif orientation == "lower":
        x1 = Xv_kb / lp_kb
        x2 = Yv_kb / lp_kb
    else:
        raise ValueError("orientation должен быть 'upper' или 'lower'.")

    span = x2 - x1

    t = np.linspace(0.0, 1.0, n_int)

    l = span[:, None] * t[None, :]
    r = span[:, None] * (1.0 - t[None, :])
    f = x1[:, None] + span[:, None] * t[None, :]

    pts = np.column_stack([l.ravel(), r.ravel()])
    Aval = A_interp(pts).reshape(len(span), n_int)

    rho = (
        np.exp(-0.5 * (f / rho_sigma) ** 2)
        / (np.sqrt(2.0 * np.pi) * rho_sigma)
    )

    integrand = Aval * rho
    Pvals = span * np.trapz(integrand, t, axis=1)

    return Pvals
	
	
def fit_ac_and_score(
    Z01,
    Pvals,
    trim_q=0.90,
    require_positive_A=True,
):
    """
    Фитит:
        Z01 = C + A * Pvals

    и возвращает robust RMSE.
    """

    Z01 = np.asarray(Z01, dtype=float)
    Pvals = np.asarray(Pvals, dtype=float)

    mask = np.isfinite(Z01) & np.isfinite(Pvals)

    if mask.sum() < 5:
        return np.inf, np.nan, np.nan

    z = Z01[mask]
    p = Pvals[mask]

    if np.nanstd(p) < 1e-12:
        return np.inf, np.nan, np.nan

    M = np.column_stack([p, np.ones_like(p)])
    A, C = np.linalg.lstsq(M, z, rcond=None)[0]

    if require_positive_A and A <= 0:
        return np.inf, A, C

    z_pred = C + A * p
    resid = z - z_pred

    cutoff = np.nanquantile(np.abs(resid), trim_q)
    resid_trim = resid[np.abs(resid) <= cutoff]

    if len(resid_trim) < 5:
        return np.inf, A, C

    score = float(np.sqrt(np.mean(resid_trim ** 2)))

    return score, float(A), float(C)
	
def fit_gamma_c_rho_sigma_progressive_grid(
    aggregate_Z,
    res,
    lp_kb,

    gamma_c0=15.0,
    gamma_b=1.8,
    alpha=0.0,

    gaussian_fit=None,
    peak_xy_kb=None,
    peak_half_window_kb=50,
    orientation="upper",

    gamma_c_range=(0.0, 50.0),
    rho_sigma_range=(0.003, 0.06),

    n_gamma=7,
    n_rho=9,
    n_rounds=4,

    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",

    N=220,
    n_int=101,

    q_low=0.05,
    q_high=0.995,
    trim_q=0.90,

    verbose=True,
):
    """
    Быстрый и устойчивый фит по gamma_c и rho_sigma.

    Вместо огромной сетки:
        round 1: грубая сетка
        round 2-4: сетка сужается вокруг лучшего результата

    PDE считается один раз на каждый gamma_c,
    а затем переиспользуется для всех rho_sigma.
    """

    Z = np.asarray(aggregate_Z, dtype=float)
    X, Y, coords_kb, extent = make_hic_grid_kb(Z, res)

    peak_x_kb, peak_y_kb = get_peak_xy_from_gaussian_or_data(
        aggregate_Z=Z,
        X=X,
        Y=Y,
        orientation=orientation,
        gaussian_fit=gaussian_fit,
        peak_xy_kb=peak_xy_kb,
    )

    fit_mask = make_local_peak_mask(
        aggregate_Z=Z,
        X=X,
        Y=Y,
        peak_x_kb=peak_x_kb,
        peak_y_kb=peak_y_kb,
        orientation=orientation,
        peak_half_window_kb=peak_half_window_kb,
    )

    if fit_mask.sum() < 10:
        raise ValueError(
            f"Слишком мало пикселей в области фита: {fit_mask.sum()}. "
            f"Увеличь peak_half_window_kb."
        )

    Xv = X[fit_mask]
    Yv = Y[fit_mask]
    Zv = Z[fit_mask]

    Z01 = robust01_values(
        Zv,
        q_low=q_low,
        q_high=q_high,
    )

    # Lmax нужен только на локальную область
    if orientation == "upper":
        x1_dim = Yv / lp_kb
        x2_dim = Xv / lp_kb
    else:
        x1_dim = Xv / lp_kb
        x2_dim = Yv / lp_kb

    max_span_dim = np.nanmax(x2_dim - x1_dim)
    Lmax = max(1.15 * max_span_dim, 0.02)

    if verbose:
        print(
            f"Local fit region: peak=({peak_x_kb:.1f}, {peak_y_kb:.1f}) kb, "
            f"half-window={peak_half_window_kb} kb, "
            f"n_pixels={fit_mask.sum()}, Lmax={Lmax:.4g}"
        )

    history = []
    best = None

    g_min, g_max = gamma_c_range
    r_min, r_max = rho_sigma_range

    for round_idx in range(n_rounds):

        gamma_grid = np.linspace(g_min, g_max, n_gamma)
        rho_grid = np.geomspace(r_min, r_max, n_rho)

        if verbose:
            print(
                f"\nRound {round_idx + 1}/{n_rounds}: "
                f"gamma=[{g_min:.4g}, {g_max:.4g}], "
                f"rho=[{r_min:.4g}, {r_max:.4g}]"
            )

        round_best = None

        for gamma_c in gamma_grid:

            try:
                kernel_res = ft.build_numeric_fountain_kernel(
                    Lmax=Lmax,
                    N=N,
                    alpha=alpha,

                    gamma_c0=gamma_c0,
                    gamma_c=float(gamma_c),
                    gamma_b=gamma_b,

                    sigma=sigma,
                    sigma2=sigma2,

                    bottom_mode=bottom_mode,
                    kernel="Nfin",
                )

                l_grid = kernel_res["l_grid"]
                r_grid = kernel_res["r_grid"]
                A_kernel = kernel_res["Nfin"]

            except Exception as e:
                if verbose:
                    print(f"kernel failed: gamma_c={gamma_c:.4g}, error={e}")
                continue

            for rho_sigma in rho_grid:

                try:
                    Pvals = local_contact_values_from_kernel(
                        l_grid=l_grid,
                        r_grid=r_grid,
                        A_kernel=A_kernel,
                        Xv_kb=Xv,
                        Yv_kb=Yv,
                        lp_kb=lp_kb,
                        rho_sigma=float(rho_sigma),
                        orientation=orientation,
                        n_int=n_int,
                    )

                    score, A_fit, C_fit = fit_ac_and_score(
                        Z01=Z01,
                        Pvals=Pvals,
                        trim_q=trim_q,
                        require_positive_A=True,
                    )

                except Exception as e:
                    score = np.inf
                    A_fit = np.nan
                    C_fit = np.nan

                row = {
                    "round": round_idx + 1,
                    "gamma_c": float(gamma_c),
                    "rho_sigma": float(rho_sigma),
                    "rho_sigma_kb": float(rho_sigma * lp_kb),
                    "A": float(A_fit) if np.isfinite(A_fit) else np.nan,
                    "C": float(C_fit) if np.isfinite(C_fit) else np.nan,
                    "score": float(score),
                }

                history.append(row)

                if np.isfinite(score):
                    if round_best is None or score < round_best["score"]:
                        round_best = row.copy()

                    if best is None or score < best["score"]:
                        best = row.copy()

        if round_best is None:
            raise RuntimeError("На этом раунде не получилось ни одного валидного результата.")

        if verbose:
            print(
                f"Best round {round_idx + 1}: "
                f"gamma_c={round_best['gamma_c']:.4g}, "
                f"rho_sigma={round_best['rho_sigma']:.5g} "
                f"({round_best['rho_sigma_kb']:.1f} kb), "
                f"score={round_best['score']:.4g}"
            )

        # Сужаем диапазон вокруг лучшего результата
        g_center = round_best["gamma_c"]
        r_center = round_best["rho_sigma"]

        g_width = (g_max - g_min) / 3.0
        log_r_width = (np.log(r_max) - np.log(r_min)) / 3.0

        g_min = max(gamma_c_range[0], g_center - g_width / 2.0)
        g_max = min(gamma_c_range[1], g_center + g_width / 2.0)

        log_r_min = max(np.log(rho_sigma_range[0]), np.log(r_center) - log_r_width / 2.0)
        log_r_max = min(np.log(rho_sigma_range[1]), np.log(r_center) + log_r_width / 2.0)

        r_min = float(np.exp(log_r_min))
        r_max = float(np.exp(log_r_max))

    history_df = pd.DataFrame(history).sort_values("score").reset_index(drop=True)

    best = best.copy()
    best.update({
        "peak_x_kb": peak_x_kb,
        "peak_y_kb": peak_y_kb,
        "peak_half_window_kb": peak_half_window_kb,
        "n_fit_pixels": int(fit_mask.sum()),
        "Lmax": float(Lmax),
    })

    pack = {
        "X": X,
        "Y": Y,
        "extent": extent,
        "fit_mask": fit_mask,
        "Z01_local": Z01,
        "history": history_df,
    }

    return best, history_df, pack



def compute_norm_rmse_from_plot_data(
    plot_data,
    orientation="upper",
    window_kb=None,
    center_xy_kb=None,
    half_window_kb=None,
):
    """
    Считает RMSE между нормированным агрегатом Z_norm
    и нормированной численной моделью P_norm.

    Ожидает plot_data после prepare_aggregate_contour_data(...).

    Parameters
    ----------
    window_kb : float or None
        Если задано, сравниваем только область |x| <= window_kb, |y| <= window_kb.

    center_xy_kb : tuple or None
        Например (50, -50). Если задано вместе с half_window_kb,
        сравниваем локальную область вокруг этого центра.

    half_window_kb : float or None
        Полуразмер локального окна вокруг center_xy_kb.
    """

    Z_norm = np.asarray(plot_data["Z_contour"], dtype=float)

    X_hic = np.asarray(plot_data["X_grid"], dtype=float)
    Y_hic = np.asarray(plot_data["Y_grid"], dtype=float)

    # Берём первую численную карту из prepared contour sets
    num = plot_data["numeric_contour_sets"][0]

    X_model = np.asarray(num["X"], dtype=float)
    Y_model = np.asarray(num["Y"], dtype=float)
    P_norm = np.asarray(num["Z"], dtype=float)

    # Оси модельной карты
    x_axis = X_model[0, :]
    y_axis = Y_model[:, 0]

    P_use = P_norm.copy()

    # RegularGridInterpolator хочет возрастающие оси
    if np.any(np.diff(x_axis) < 0):
        x_axis = x_axis[::-1]
        P_use = P_use[:, ::-1]

    if np.any(np.diff(y_axis) < 0):
        y_axis = y_axis[::-1]
        P_use = P_use[::-1, :]

    interp = RegularGridInterpolator(
        (y_axis, x_axis),
        P_use,
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([
        Y_hic.ravel(),
        X_hic.ravel(),
    ])

    P_on_hic = interp(pts).reshape(Z_norm.shape)

    # Маска области сравнения
    if orientation == "upper":
        mask = (X_hic >= 0) & (Y_hic <= 0)
    elif orientation == "lower":
        mask = (X_hic <= 0) & (Y_hic >= 0)
    elif orientation == "both":
        mask = ((X_hic >= 0) & (Y_hic <= 0)) | ((X_hic <= 0) & (Y_hic >= 0))
    else:
        raise ValueError("orientation должен быть 'upper', 'lower' или 'both'.")

    if window_kb is not None:
        mask = mask & (np.abs(X_hic) <= window_kb) & (np.abs(Y_hic) <= window_kb)

    if center_xy_kb is not None and half_window_kb is not None:
        cx, cy = center_xy_kb
        mask = mask & (np.abs(X_hic - cx) <= half_window_kb)
        mask = mask & (np.abs(Y_hic - cy) <= half_window_kb)

    mask = mask & np.isfinite(Z_norm) & np.isfinite(P_on_hic)

    diff = Z_norm[mask] - P_on_hic[mask]

    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {
        "rmse": rmse,
        "n_pixels": int(mask.sum()),
        "Z_norm": Z_norm,
        "P_norm_on_hic": P_on_hic,
        "mask": mask,
    }
    
    
    
def compute_rmse_for_numeric_fit_params(
    aggregate_Z,
    res_agg,
    flank_kb,

    lp_kb,
    rho_sigma,
    gamma_c0,

    # fixed model params
    alpha=0.0,
    gamma_c=0.0,
    gamma_b=1.8,
    sigma=0.001,
    sigma2=0.001,

    # numerical params
    N=500,
    n_map=250,
    n_int=251,

    # RMSE region
    orientation="upper",
    norm_window_kb=150,
    center_xy_kb=(40, -40),
    half_window_kb=40,

    # contour preparation
    hic_contour_smooth_sigma=1.0,
    numeric_contour_smooth_sigma=1.0,

    verbose=False,
):
    """
    Считает численную модель для заданных lp_kb, rho_sigma, gamma_c0
    и возвращает RMSE между Z_norm и P_norm.
    """

    lp_kb = float(lp_kb)
    rho_sigma = float(rho_sigma)
    gamma_c0 = float(gamma_c0)

    xmax = flank_kb / lp_kb
    Lmax = 2.2 * xmax

    kernel_res = ft.build_numeric_fountain_kernel(
        Lmax=Lmax,
        N=N,
        alpha=alpha,

        gamma_c0=gamma_c0,
        gamma_c=gamma_c,
        gamma_b=gamma_b,

        sigma=sigma,
        sigma2=sigma2,

        bottom_mode="gamma0_compatible",
        kernel="Nfin",
    )

    P, LEFT, RIGHT, A_interp = ft.contact_map_from_kernel_array(
        l_grid=kernel_res["l_grid"],
        r_grid=kernel_res["r_grid"],
        A_kernel=kernel_res["Nfin"],

        xmax=xmax,
        n_map=n_map,

        rho_mu=0.0,
        rho_sigma=rho_sigma,

        n_int=n_int,
        normalize=True,
    )

    plot_data = ft.prepare_aggregate_contour_data(
        aggregate_Z=aggregate_Z,
        P=P,
        LEFT=LEFT,
        RIGHT=RIGHT,
        res=res_agg,
        lp_kb=lp_kb,
        orientation=orientation,

        hic_contour_smooth_sigma=hic_contour_smooth_sigma,
        numeric_contour_smooth_sigma=numeric_contour_smooth_sigma,

        common_contour_levels=[0.5, 0.7, 0.9],
        normalize_contours=True,
        norm_window_kb=norm_window_kb,

        gaussian_fit=None,
        gaussian_fit_cache=None,
    )

    rmse_res = ft.compute_norm_rmse_from_plot_data(
        plot_data,
        orientation=orientation,
        center_xy_kb=center_xy_kb,
        half_window_kb=half_window_kb,
    )

    rmse = float(rmse_res["rmse"])

    if verbose:
        print(
            f"lp={lp_kb:.1f}, "
            f"rho_sigma={rho_sigma:.5f}, "
            f"gamma_c0={gamma_c0:.3f}, "
            f"rmse={rmse:.5f}"
        )

    return {
        "rmse": rmse,
        "lp_kb": lp_kb,
        "rho_sigma": rho_sigma,
        "rho_sigma_kb": rho_sigma * lp_kb,
        "gamma_c0": gamma_c0,
        "P": P,
        "LEFT": LEFT,
        "RIGHT": RIGHT,
        "plot_data": plot_data,
        "rmse_res": rmse_res,
        "kernel_res": kernel_res,
    }
    
    
def fit_lp_rhosigma_gammac0_simple(
    aggregate_Z,
    res_agg=5_000,
    flank_kb=200,

    # initial guess
    lp_kb_init=1900.0,
    rho_sigma_init=0.021,
    gamma_c0_init=8.0,

    # bounds
    lp_kb_bounds=(1200.0, 2800.0),
    rho_sigma_bounds=(0.005, 0.05),
    gamma_c0_bounds=(0.0, 30.0),

    # initial steps
    lp_kb_step=150.0,
    rho_sigma_step=0.002,
    gamma_c0_step=1.0,

    # search settings
    max_iter=20,
    shrink=0.5,
    min_lp_kb_step=10.0,
    min_rho_sigma_step=0.0002,
    min_gamma_c0_step=0.1,

    # model params
    alpha=0.0,
    gamma_c=0.0,
    gamma_b=1.8,
    sigma=0.001,
    sigma2=0.001,

    # numerical params
    N=400,
    n_map=220,
    n_int=201,

    # RMSE region
    orientation="upper",
    norm_window_kb=150,
    center_xy_kb=(40, -40),
    half_window_kb=40,

    verbose=True,
):
    """
    Простой coordinate-search:
    на каждом шаге пробуем +/- шаг по каждому параметру.
    Если стало лучше — переходим туда.
    Если нет — уменьшаем шаги.
    """

    def clip_params(lp_kb, rho_sigma, gamma_c0):
        lp_kb = np.clip(lp_kb, *lp_kb_bounds)
        rho_sigma = np.clip(rho_sigma, *rho_sigma_bounds)
        gamma_c0 = np.clip(gamma_c0, *gamma_c0_bounds)
        return float(lp_kb), float(rho_sigma), float(gamma_c0)

    cache = {}

    def eval_params(lp_kb, rho_sigma, gamma_c0):
        lp_kb, rho_sigma, gamma_c0 = clip_params(lp_kb, rho_sigma, gamma_c0)

        key = (
            round(lp_kb, 6),
            round(rho_sigma, 8),
            round(gamma_c0, 6),
        )

        if key in cache:
            return cache[key]

        try:
            out = compute_rmse_for_numeric_fit_params(
                aggregate_Z=aggregate_Z,
                res_agg=res_agg,
                flank_kb=flank_kb,

                lp_kb=lp_kb,
                rho_sigma=rho_sigma,
                gamma_c0=gamma_c0,

                alpha=alpha,
                gamma_c=gamma_c,
                gamma_b=gamma_b,
                sigma=sigma,
                sigma2=sigma2,

                N=N,
                n_map=n_map,
                n_int=n_int,

                orientation=orientation,
                norm_window_kb=norm_window_kb,
                center_xy_kb=center_xy_kb,
                half_window_kb=half_window_kb,

                verbose=False,
            )

            row = {
                "lp_kb": out["lp_kb"],
                "rho_sigma": out["rho_sigma"],
                "rho_sigma_kb": out["rho_sigma_kb"],
                "gamma_c0": out["gamma_c0"],
                "rmse": out["rmse"],
                "ok": True,
                "error": None,
                "full": out,
            }

        except Exception as e:
            row = {
                "lp_kb": lp_kb,
                "rho_sigma": rho_sigma,
                "rho_sigma_kb": rho_sigma * lp_kb,
                "gamma_c0": gamma_c0,
                "rmse": np.inf,
                "ok": False,
                "error": str(e),
                "full": None,
            }

        cache[key] = row
        return row

    lp_kb, rho_sigma, gamma_c0 = clip_params(
        lp_kb_init,
        rho_sigma_init,
        gamma_c0_init,
    )

    steps = {
        "lp_kb": float(lp_kb_step),
        "rho_sigma": float(rho_sigma_step),
        "gamma_c0": float(gamma_c0_step),
    }

    history = []

    current = eval_params(lp_kb, rho_sigma, gamma_c0)

    if verbose:
        print(
            "Initial:",
            f"lp={current['lp_kb']:.1f},",
            f"rho={current['rho_sigma']:.5f},",
            f"gamma_c0={current['gamma_c0']:.3f},",
            f"rmse={current['rmse']:.5f}",
        )

    for it in range(max_iter):

        candidates = []

        lp = current["lp_kb"]
        rho = current["rho_sigma"]
        gam = current["gamma_c0"]

        # текущая точка
        candidates.append(current)

        # +/- по lp
        candidates.append(eval_params(lp + steps["lp_kb"], rho, gam))
        candidates.append(eval_params(lp - steps["lp_kb"], rho, gam))

        # +/- по rho_sigma
        candidates.append(eval_params(lp, rho + steps["rho_sigma"], gam))
        candidates.append(eval_params(lp, rho - steps["rho_sigma"], gam))

        # +/- по gamma_c0
        candidates.append(eval_params(lp, rho, gam + steps["gamma_c0"]))
        candidates.append(eval_params(lp, rho, gam - steps["gamma_c0"]))

        # выбираем лучший вариант
        best_candidate = min(candidates, key=lambda x: x["rmse"])

        improved = best_candidate["rmse"] < current["rmse"] - 1e-8

        for c in candidates:
            history.append({
                "iter": it,
                "lp_kb": c["lp_kb"],
                "rho_sigma": c["rho_sigma"],
                "rho_sigma_kb": c["rho_sigma_kb"],
                "gamma_c0": c["gamma_c0"],
                "rmse": c["rmse"],
                "ok": c["ok"],
                "error": c["error"],
                "step_lp_kb": steps["lp_kb"],
                "step_rho_sigma": steps["rho_sigma"],
                "step_gamma_c0": steps["gamma_c0"],
                "accepted": False,
            })

        if improved:
            current = best_candidate

            if verbose:
                print(
                    f"iter {it:02d}: improved -> "
                    f"lp={current['lp_kb']:.1f}, "
                    f"rho={current['rho_sigma']:.5f} "
                    f"({current['rho_sigma_kb']:.1f} kb), "
                    f"gamma_c0={current['gamma_c0']:.3f}, "
                    f"rmse={current['rmse']:.5f}"
                )

        else:
            steps["lp_kb"] *= shrink
            steps["rho_sigma"] *= shrink
            steps["gamma_c0"] *= shrink

            if verbose:
                print(
                    f"iter {it:02d}: no improvement, shrink steps -> "
                    f"d_lp={steps['lp_kb']:.3g}, "
                    f"d_rho={steps['rho_sigma']:.3g}, "
                    f"d_gamma={steps['gamma_c0']:.3g}"
                )

        if (
            steps["lp_kb"] <= min_lp_kb_step
            and steps["rho_sigma"] <= min_rho_sigma_step
            and steps["gamma_c0"] <= min_gamma_c0_step
        ):
            if verbose:
                print("Stop: steps are small enough.")
            break

    history_df = pd.DataFrame(history)

    # достаём лучший полный результат
    best_full = current["full"]

    best = {
        "lp_kb": current["lp_kb"],
        "rho_sigma": current["rho_sigma"],
        "rho_sigma_kb": current["rho_sigma_kb"],
        "gamma_c0": current["gamma_c0"],
        "rmse": current["rmse"],
        "steps_final": steps,
        "n_evals": len(cache),
    }

    return best, history_df, best_full
    
import numpy as np
from scipy.ndimage import gaussian_filter


def nan_gaussian_filter(A, sigma):
    """
    Gaussian smoothing with NaN handling.
    """

    A = np.asarray(A, dtype=float)
    finite = np.isfinite(A)

    if sigma is None or sigma <= 0:
        return A.copy()

    numerator = gaussian_filter(
        np.where(finite, A, 0.0),
        sigma=sigma,
    )

    denominator = gaussian_filter(
        finite.astype(float),
        sigma=sigma,
    )

    out = numerator / np.maximum(denominator, 1e-12)
    out[denominator < 1e-12] = np.nan

    return out


def compute_smoothing_rmse_floor(
    aggregate_Z,
    fit_cache,
    fit=None,
    p_center_kb=None,
    near_peak_radius_kb=70,
    diag_exclusion_kb=10,
    smoothing_sigma_bins=1.0,
    normalize=False,
    norm_q_low=0.05,
    norm_q_high=0.995,
):
    """
    Считает RMSE между исходной Hi-C картой и сглаженной Hi-C картой
    в том же near-peak окне, где считается качество фита.

    Это можно использовать как эмпирический error floor для profile-RMSE.

    Parameters
    ----------
    aggregate_Z : 2D array
        Агрегированная Hi-C карта.

    fit_cache : FountainFitCache
        Тот же fit_cache, что используется в fit.

    fit : dict or None
        Можно передать fit_agg. Тогда p_center_kb берется из:
            fit["fit_peak_center_kb"]
        если оно есть.

    p_center_kb : float or None
        Центр near-peak окна. Если None, берется из fit.

    near_peak_radius_kb : float
        Радиус окна в координатах U, V.

    smoothing_sigma_bins : float
        Sigma Gaussian smoothing в bins.
        Например, sigma=1.0 при res=5 kb означает 5 kb.

    normalize : bool
        Если True, исходная и сглаженная карта сравниваются
        после одной и той же robust-нормировки по исходной Hi-C карте.

    Returns
    -------
    dict
    """

    Z = np.asarray(aggregate_Z, dtype=float)
    n = Z.shape[0]

    ctx = fit_cache.get(n)
    X = ctx["X"]
    Y = ctx["Y"]

    if p_center_kb is None:
        if fit is not None:
            p_center_kb = fit.get(
                "fit_peak_center_kb",
                fit.get("p_kb", None),
            )

        if p_center_kb is None:
            raise ValueError(
                "Нужно передать p_center_kb или fit с полем "
                "'fit_peak_center_kb' / 'p_kb'."
            )

    p_center_kb = float(p_center_kb)

    U = (X + Y) / np.sqrt(2)
    V = (X - Y - 2 * p_center_kb) / np.sqrt(2)

    mask = np.isfinite(Z)
    mask &= Y < X - diag_exclusion_kb
    mask &= (U**2 + V**2) <= near_peak_radius_kb**2

    Z_smooth = nan_gaussian_filter(
        Z,
        sigma=smoothing_sigma_bins,
    )

    mask &= np.isfinite(Z_smooth)

    H = Z[mask].astype(float)
    Hs = Z_smooth[mask].astype(float)

    if len(H) < 5:
        return {
            "smoothing_rmse": np.nan,
            "smoothing_nrmse_percent": np.nan,
            "n_pixels": int(len(H)),
        }

    if normalize:
        lo, hi = np.nanquantile(H, [norm_q_low, norm_q_high])
        scale = max(hi - lo, 1e-12)

        H = np.clip((H - lo) / scale, 0.0, 1.0)
        Hs = np.clip((Hs - lo) / scale, 0.0, 1.0)

    diff = H - Hs

    smoothing_rmse = float(np.sqrt(np.mean(diff**2)))

    robust_range = float(
        max(
            np.nanquantile(H, norm_q_high)
            - np.nanquantile(H, norm_q_low),
            1e-12,
        )
    )

    smoothing_nrmse = smoothing_rmse / robust_range

    return {
        "smoothing_rmse": smoothing_rmse,
        "smoothing_nrmse_percent": 100.0 * smoothing_nrmse,
        "robust_range": robust_range,
        "n_pixels": int(len(H)),
        "near_peak_radius_kb": near_peak_radius_kb,
        "smoothing_sigma_bins": smoothing_sigma_bins,
        "p_center_kb": p_center_kb,
        "normalize": normalize,
    }


import time

def profile_rmse_threshold_from_smoothing(
    rmse_min,
    smoothing_rmse,
):
    """
    Два варианта порога для profile RMSE.

    additive:
        RMSE <= RMSE_min + smoothing_rmse

    quadrature:
        RMSE^2 <= RMSE_min^2 + smoothing_rmse^2

    Второй вариант лучше обосновывается статистически.
    """

    rmse_min = float(rmse_min)
    smoothing_rmse = float(smoothing_rmse)

    threshold_additive = rmse_min + smoothing_rmse

    threshold_quadrature = np.sqrt(
        rmse_min**2 + smoothing_rmse**2
    )

    return {
        "rmse_min": rmse_min,
        "smoothing_rmse": smoothing_rmse,

        "threshold_additive": float(threshold_additive),
        "threshold_quadrature": float(threshold_quadrature),

        "delta_rmse_additive": float(
            threshold_additive - rmse_min
        ),

        "delta_rmse_quadrature": float(
            threshold_quadrature - rmse_min
        ),

        "relative_delta_additive_percent": float(
            100.0 * (threshold_additive / rmse_min - 1.0)
        ),

        "relative_delta_quadrature_percent": float(
            100.0 * (threshold_quadrature / rmse_min - 1.0)
        ),
    }


import numpy as np
import pandas as pd


def fit_numeric_fountain_coordinate_descent(
    aggregate_Z,
    flank_kb,
    res_agg,

    # начальная точка
    lp_kb=2300.0,
    rho_sigma=0.017,
    gamma_c0=10.0,

    # фиксированные параметры
    fixed_params=None,

    # начальные шаги поиска
    step_lp=100.0,
    step_rho=0.001,
    step_gamma=1.0,

    # границы параметров
    lp_bounds=(1200.0, 3500.0),
    rho_bounds=(0.003, 0.06),
    gamma_bounds=(0.0, 40.0),

    # model params
    alpha=0.0,
    gamma_c=0.0,
    gamma_b=1.8,
    sigma=0.001,
    sigma2=0.001,
    bottom_mode="gamma0_compatible",
    kernel="Nfin",

    # numerical params
    N=600,
    n_map=300,
    n_int=301,
    Lmax_factor=2.2,

    # contour / normalization params
    hic_contour_smooth_sigma=1.0,
    numeric_contour_smooth_sigma=1.0,
    common_contour_levels=(0.5, 0.7, 0.9),
    normalize_contours=True,
    norm_window_kb=150,

    # RMSE window
    rmse_orientation="upper",
    rmse_center_xy_kb=(40, -40),
    rmse_half_window_kb=40,

    # peak-aware score params
    peak_weight=3.0,
    peak_search_window_kb=50,
    peak_scale_kb=20.0,
    peak_quantile=0.80,

    # search params
    max_iter=25,
    shrink=0.5,
    tol_improve=1e-5,
    min_step_lp=10.0,
    min_step_rho=0.0001,
    min_step_gamma=0.1,

    # diagnostics
    verbose=True,
    verbose_level=2,
    return_cache=False,
):
    """
    Coordinate descent для численного фита фонтана.

    verbose_level:
        0 — ничего не печатать
        1 — только итерации, кандидаты, ACCEPT / shrink
        2 — дополнительно печатать дорогие стадии:
            kernel, contact map, contour data, RMSE, peak score
    """

    if fixed_params is None:
        fixed_params = {}

    if "gamma_ci" in fixed_params:
        fixed_params["gamma_c0"] = fixed_params.pop("gamma_ci")

    allowed_fixed = {"lp_kb", "rho_sigma", "gamma_c0"}
    unknown = set(fixed_params) - allowed_fixed
    if unknown:
        raise ValueError(f"Unknown fixed parameter(s): {unknown}")

    lp_min, lp_max = lp_bounds
    rho_min, rho_max = rho_bounds
    gamma_min, gamma_max = gamma_bounds

    if "lp_kb" in fixed_params:
        lp_kb = float(fixed_params["lp_kb"])
    if "rho_sigma" in fixed_params:
        rho_sigma = float(fixed_params["rho_sigma"])
    if "gamma_c0" in fixed_params:
        gamma_c0 = float(fixed_params["gamma_c0"])

    lp_kb = float(np.clip(lp_kb, lp_min, lp_max))
    rho_sigma = float(np.clip(rho_sigma, rho_min, rho_max))
    gamma_c0 = float(np.clip(gamma_c0, gamma_min, gamma_max))

    history = []
    cache = {}

    current_rmse = np.inf
    current_score = np.inf
    current_plot_data = None
    current_P = None
    current_LEFT = None
    current_RIGHT = None
    current_kernel_res = None

    t_global = time.perf_counter()

    def log(msg, level=1):
        if verbose and verbose_level >= level:
            print(msg, flush=True)

    def fmt_time(t0):
        return f"{time.perf_counter() - t0:.2f}s"

    log("\nSTART numeric coordinate descent")
    log(
        f"initial: lp={lp_kb:.1f}, "
        f"rho={rho_sigma:.5f} ({rho_sigma * lp_kb:.1f} kb), "
        f"gamma={gamma_c0:.2f}"
    )
    log(f"fixed_params = {fixed_params}")
    log(
        f"steps: step_lp={step_lp}, "
        f"step_rho={step_rho}, "
        f"step_gamma={step_gamma}\n"
    )

    def make_candidates(lp, rho, gamma):
        candidates = [(lp, rho, gamma, "current")]

        if "lp_kb" not in fixed_params:
            candidates.extend([
                (lp + step_lp, rho, gamma, "+lp"),
                (lp - step_lp, rho, gamma, "-lp"),
            ])

        if "rho_sigma" not in fixed_params:
            candidates.extend([
                (lp, rho + step_rho, gamma, "+rho"),
                (lp, rho - step_rho, gamma, "-rho"),
            ])

        if "gamma_c0" not in fixed_params:
            candidates.extend([
                (lp, rho, gamma + step_gamma, "+gamma"),
                (lp, rho, gamma - step_gamma, "-gamma"),
            ])

        out = []
        seen = set()

        for lp_try, rho_try, gamma_try, move_name in candidates:
            lp_try = float(np.clip(lp_try, lp_min, lp_max))
            rho_try = float(np.clip(rho_try, rho_min, rho_max))
            gamma_try = float(np.clip(gamma_try, gamma_min, gamma_max))

            if "lp_kb" in fixed_params:
                lp_try = float(fixed_params["lp_kb"])
            if "rho_sigma" in fixed_params:
                rho_try = float(fixed_params["rho_sigma"])
            if "gamma_c0" in fixed_params:
                gamma_try = float(fixed_params["gamma_c0"])

            key = (
                round(lp_try, 6),
                round(rho_try, 8),
                round(gamma_try, 6),
            )

            if key not in seen:
                seen.add(key)
                out.append((lp_try, rho_try, gamma_try, move_name))

        return out

    def evaluate_candidate(lp_try, rho_try, gamma_try, move_name="", iter_id=None):
        t_candidate = time.perf_counter()

        key = (
            round(lp_try, 6),
            round(rho_try, 8),
            round(gamma_try, 6),
        )

        prefix = f"[iter {iter_id:02d} | {move_name:>8s}]" if iter_id is not None else "[candidate]"

        log(
            f"{prefix} evaluate: "
            f"lp={lp_try:.1f}, "
            f"rho={rho_try:.5f} ({rho_try * lp_try:.1f} kb), "
            f"gamma={gamma_try:.2f}",
            level=1,
        )

        if key in cache:
            log(f"{prefix} cache hit", level=2)
            cached = cache[key]
            log(
                f"{prefix} done from cache | "
                f"rmse={cached['rmse']:.5f}, "
                f"peak_dist={cached['peak_dist_kb']:.1f} kb, "
                f"score={cached['score']:.5f}",
                level=1,
            )
            return cached

        log(f"{prefix} cache miss", level=2)

        try:
            xmax = flank_kb / lp_try
            Lmax = Lmax_factor * xmax

            log(
                f"{prefix} build kernel: "
                f"xmax={xmax:.4f}, Lmax={Lmax:.4f}, N={N}",
                level=2,
            )
            t0 = time.perf_counter()

            kernel_res = build_numeric_fountain_kernel(
                Lmax=Lmax,
                N=N,
                alpha=alpha,

                gamma_c0=gamma_try,
                gamma_c=gamma_c,
                gamma_b=gamma_b,

                sigma=sigma,
                sigma2=sigma2,

                bottom_mode=bottom_mode,
                kernel=kernel,
            )

            log(f"{prefix} kernel built in {fmt_time(t0)}", level=2)

            A_kernel = kernel_res["Nfin"].copy()

            log(
                f"{prefix} build contact map: "
                f"n_map={n_map}, n_int={n_int}, rho_sigma={rho_try:.5f}",
                level=2,
            )
            t0 = time.perf_counter()

            P, LEFT, RIGHT, A_interp = contact_map_from_kernel_array(
                l_grid=kernel_res["l_grid"],
                r_grid=kernel_res["r_grid"],
                A_kernel=A_kernel,

                xmax=xmax,
                n_map=n_map,

                rho_mu=0.0,
                rho_sigma=rho_try,

                n_int=n_int,
                normalize=True,
            )

            log(f"{prefix} contact map built in {fmt_time(t0)}", level=2)

            log(f"{prefix} prepare contour data", level=2)
            t0 = time.perf_counter()

            plot_data = prepare_aggregate_contour_data(
                aggregate_Z=aggregate_Z,
                P=P,
                LEFT=LEFT,
                RIGHT=RIGHT,
                res=res_agg,
                lp_kb=lp_try,
                orientation="upper",

                hic_contour_smooth_sigma=hic_contour_smooth_sigma,
                numeric_contour_smooth_sigma=numeric_contour_smooth_sigma,

                common_contour_levels=list(common_contour_levels),
                normalize_contours=normalize_contours,
                norm_window_kb=norm_window_kb,

                gaussian_fit=None,
                gaussian_fit_cache=None,
            )

            log(f"{prefix} contour data ready in {fmt_time(t0)}", level=2)

            log(
                f"{prefix} compute RMSE: "
                f"center={rmse_center_xy_kb}, half_window={rmse_half_window_kb} kb",
                level=2,
            )
            t0 = time.perf_counter()

            rmse_res = compute_norm_rmse_from_plot_data(
                plot_data,
                orientation=rmse_orientation,
                center_xy_kb=rmse_center_xy_kb,
                half_window_kb=rmse_half_window_kb,
            )

            log(f"{prefix} RMSE computed in {fmt_time(t0)}", level=2)

            Z_norm = rmse_res["Z_norm"]
            P_norm = rmse_res["P_norm_on_hic"]

            X_grid = plot_data["X_grid"]
            Y_grid = plot_data["Y_grid"]

            cx, cy = rmse_center_xy_kb

            log(
                f"{prefix} compute peak-aware score: "
                f"peak_window={peak_search_window_kb} kb, "
                f"peak_quantile={peak_quantile}",
                level=2,
            )

            peak_mask = (
                (np.abs(X_grid - cx) <= peak_search_window_kb) &
                (np.abs(Y_grid - cy) <= peak_search_window_kb) &
                np.isfinite(Z_norm) &
                np.isfinite(P_norm)
            )

            log(
                f"{prefix} peak mask pixels: {int(peak_mask.sum())}",
                level=2,
            )

            if peak_mask.sum() < 5:
                rmse = np.inf
                peak_dist_kb = np.inf
                peak_penalty = np.inf
                score = np.inf

                z_peak_x = np.nan
                z_peak_y = np.nan
                p_peak_x = np.nan
                p_peak_y = np.nan

                log(f"{prefix} too few pixels in peak mask -> inf score", level=2)

            else:
                rmse = float(rmse_res["rmse"])

                Z_local = Z_norm[peak_mask]
                P_local = P_norm[peak_mask]
                X_local = X_grid[peak_mask]
                Y_local = Y_grid[peak_mask]

                z_thr = np.nanquantile(Z_local, peak_quantile)
                p_thr = np.nanquantile(P_local, peak_quantile)

                z_top = Z_local >= z_thr
                p_top = P_local >= p_thr

                log(
                    f"{prefix} top pixels: "
                    f"Hi-C={int(z_top.sum())}, model={int(p_top.sum())}; "
                    f"thresholds: Hi-C={z_thr:.4g}, model={p_thr:.4g}",
                    level=2,
                )

                if z_top.sum() < 3 or p_top.sum() < 3:
                    peak_dist_kb = np.inf
                    peak_penalty = np.inf
                    score = np.inf

                    z_peak_x = np.nan
                    z_peak_y = np.nan
                    p_peak_x = np.nan
                    p_peak_y = np.nan

                    log(f"{prefix} too few top pixels -> inf score", level=2)

                else:
                    wz = Z_local[z_top] - z_thr
                    wp = P_local[p_top] - p_thr

                    if np.sum(wz) <= 0:
                        wz = np.ones_like(wz)
                    if np.sum(wp) <= 0:
                        wp = np.ones_like(wp)

                    z_peak_x = float(np.average(X_local[z_top], weights=wz))
                    z_peak_y = float(np.average(Y_local[z_top], weights=wz))

                    p_peak_x = float(np.average(X_local[p_top], weights=wp))
                    p_peak_y = float(np.average(Y_local[p_top], weights=wp))

                    peak_dist_kb = float(
                        np.sqrt(
                            (p_peak_x - z_peak_x) ** 2
                            + (p_peak_y - z_peak_y) ** 2
                        )
                    )

                    peak_penalty = float(
                        peak_weight * (peak_dist_kb / peak_scale_kb) ** 2
                    )

                    score = float(rmse + peak_penalty)

                    log(
                        f"{prefix} peaks: "
                        f"Hi-C=({z_peak_x:.1f}, {z_peak_y:.1f}), "
                        f"model=({p_peak_x:.1f}, {p_peak_y:.1f}), "
                        f"dist={peak_dist_kb:.1f} kb",
                        level=2,
                    )

            result = {
                "rmse": rmse,
                "score": score,
                "peak_dist_kb": peak_dist_kb,
                "peak_penalty": peak_penalty,
                "plot_data": plot_data,

                "z_peak_x": z_peak_x,
                "z_peak_y": z_peak_y,
                "p_peak_x": p_peak_x,
                "p_peak_y": p_peak_y,

                "P": P,
                "LEFT": LEFT,
                "RIGHT": RIGHT,
                "kernel_res": kernel_res,
            }

            cache[key] = result

            log(
                f"{prefix} DONE in {fmt_time(t_candidate)} | "
                f"rmse={rmse:.5f}, "
                f"peak_dist={peak_dist_kb:.1f} kb, "
                f"peak_penalty={peak_penalty:.5f}, "
                f"score={score:.5f}",
                level=1,
            )

            return result

        except Exception as e:
            log(f"{prefix} FAILED: {repr(e)}", level=1)

            result = {
                "rmse": np.inf,
                "score": np.inf,
                "peak_dist_kb": np.inf,
                "peak_penalty": np.inf,
                "plot_data": None,

                "z_peak_x": np.nan,
                "z_peak_y": np.nan,
                "p_peak_x": np.nan,
                "p_peak_y": np.nan,

                "P": None,
                "LEFT": None,
                "RIGHT": None,
                "kernel_res": None,
                "error": repr(e),
            }

            cache[key] = result
            return result

    for it in range(max_iter):

        log("\n" + "=" * 80)
        log(
            f"ITERATION {it:02d} | current: "
            f"lp={lp_kb:.1f}, "
            f"rho={rho_sigma:.5f} ({rho_sigma * lp_kb:.1f} kb), "
            f"gamma={gamma_c0:.2f}, "
            f"current_rmse={current_rmse:.5f}, "
            f"current_score={current_score:.5f}"
        )
        log(
            f"steps: lp={step_lp:.4g}, "
            f"rho={step_rho:.4g}, "
            f"gamma={step_gamma:.4g}"
        )

        t_iter = time.perf_counter()

        candidates = make_candidates(
            lp_kb,
            rho_sigma,
            gamma_c0,
        )

        log(f"candidates: {len(candidates)}", level=1)

        evaluated = []

        for cand_id, (lp_try, rho_try, gamma_try, move_name) in enumerate(candidates, start=1):
            log(f"\n--- candidate {cand_id}/{len(candidates)} ---", level=1)

            res = evaluate_candidate(
                lp_try,
                rho_try,
                gamma_try,
                move_name=move_name,
                iter_id=it,
            )

            rmse = res["rmse"]
            score = res["score"]
            peak_dist_kb = res["peak_dist_kb"]
            peak_penalty = res["peak_penalty"]

            row = {
                "iter": it,
                "move": move_name,
                "lp_kb": lp_try,
                "rho_sigma": rho_try,
                "rho_sigma_kb": rho_try * lp_try,
                "gamma_c0": gamma_try,
                "rmse": rmse,
                "score": score,
                "peak_dist_kb": peak_dist_kb,
                "peak_penalty": peak_penalty,
                "z_peak_x": res["z_peak_x"],
                "z_peak_y": res["z_peak_y"],
                "p_peak_x": res["p_peak_x"],
                "p_peak_y": res["p_peak_y"],
                "step_lp": step_lp,
                "step_rho": step_rho,
                "step_gamma": step_gamma,
            }

            if "error" in res:
                row["error"] = res["error"]

            history.append(row)

            evaluated.append({
                **row,
                "plot_data": res["plot_data"],
                "P": res["P"],
                "LEFT": res["LEFT"],
                "RIGHT": res["RIGHT"],
                "kernel_res": res["kernel_res"],
            })

            log(
                f"SUMMARY iter={it:02d} {move_name:>8s} | "
                f"lp={lp_try:.1f}, "
                f"rho={rho_try:.5f} ({rho_try * lp_try:.1f} kb), "
                f"gamma={gamma_try:.2f}, "
                f"rmse={rmse:.5f}, "
                f"peak_dist={peak_dist_kb:.1f} kb, "
                f"peak_penalty={peak_penalty:.5f}, "
                f"score={score:.5f}",
                level=1,
            )

        best_row = min(evaluated, key=lambda x: x["score"])

        log("\nITERATION RESULT:")
        log(
            f"best move={best_row['move']}, "
            f"lp={best_row['lp_kb']:.1f}, "
            f"rho={best_row['rho_sigma']:.5f}, "
            f"gamma={best_row['gamma_c0']:.2f}, "
            f"rmse={best_row['rmse']:.5f}, "
            f"score={best_row['score']:.5f}"
        )

        if best_row["score"] < current_score - tol_improve:
            lp_kb = best_row["lp_kb"]
            rho_sigma = best_row["rho_sigma"]
            gamma_c0 = best_row["gamma_c0"]

            current_rmse = best_row["rmse"]
            current_score = best_row["score"]
            current_plot_data = best_row["plot_data"]
            current_P = best_row["P"]
            current_LEFT = best_row["LEFT"]
            current_RIGHT = best_row["RIGHT"]
            current_kernel_res = best_row["kernel_res"]

            log(
                "\nACCEPT: "
                f"move={best_row['move']}, "
                f"lp={lp_kb:.1f}, "
                f"rho={rho_sigma:.5f}, "
                f"gamma={gamma_c0:.2f}, "
                f"rmse={current_rmse:.5f}, "
                f"peak_dist={best_row['peak_dist_kb']:.1f} kb, "
                f"peak_penalty={best_row['peak_penalty']:.5f}, "
                f"score={current_score:.5f}"
            )

        else:
            step_lp *= shrink
            step_rho *= shrink
            step_gamma *= shrink

            log(
                "\nNO IMPROVEMENT -> shrink steps: "
                f"step_lp={step_lp:.4g}, "
                f"step_rho={step_rho:.4g}, "
                f"step_gamma={step_gamma:.4g}"
            )

        log(f"iteration time: {fmt_time(t_iter)}")
        log(f"elapsed total: {fmt_time(t_global)}")

        if (
            step_lp <= min_step_lp
            and step_rho <= min_step_rho
            and step_gamma <= min_step_gamma
        ):
            log("\nSTOP: steps are small enough.")
            break

    history_df = pd.DataFrame(history)

    best_params = {
        "lp_kb": float(lp_kb),
        "rho_sigma": float(rho_sigma),
        "rho_sigma_kb": float(rho_sigma * lp_kb),
        "gamma_c0": float(gamma_c0),
    }

    result = {
        "best_params": best_params,
        "best_rmse": float(current_rmse),
        "best_score": float(current_score),
        "best_plot_data": current_plot_data,

        "best_P": current_P,
        "best_LEFT": current_LEFT,
        "best_RIGHT": current_RIGHT,
        "best_kernel_res": current_kernel_res,

        "history_df": history_df,
        "fixed_params": dict(fixed_params),
    }

    if return_cache:
        result["cache"] = cache

    log("\n" + "=" * 80)
    log("BEST:")
    log(
        f"lp_kb = {lp_kb:.3f}\n"
        f"rho_sigma = {rho_sigma:.6f}\n"
        f"rho_sigma_kb = {rho_sigma * lp_kb:.3f}\n"
        f"gamma_c0 = {gamma_c0:.3f}\n"
        f"rmse = {current_rmse:.6f}\n"
        f"score = {current_score:.6f}\n"
        f"total time = {fmt_time(t_global)}"
    )

    return result