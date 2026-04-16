from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import RankerConfig
from .macro_states import get_macro_factor_weights
from .utils import robust_zscore

try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

CORE_FACTORS = ["shareholder_yield", "gross_profitability", "adjusted_book_to_market"]
OPTIONAL_CORE_FACTORS = ["residual_value_signal", "compounder_persistence_signal"]
BASE_BENEISH_HARD_FILTER_THRESHOLD = -1.20
LARGE_UNIVERSE_BENEISH_HARD_FILTER_THRESHOLD = -1.40
LARGE_UNIVERSE_MIN_SIZE = 1000
LIFE_CYCLE_STAGES = ("growth", "mature", "recovery")


# ---------------------------------------------------------------------------
# Pure utility helpers (no business logic)
# ---------------------------------------------------------------------------

def _empty_series_for_dtype(dtype: object, index: pd.Index) -> pd.Series:
    if pd.api.types.is_bool_dtype(dtype):
        return pd.Series(pd.NA, index=index, dtype="boolean")
    if pd.api.types.is_numeric_dtype(dtype):
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.Series(pd.NA, index=index, dtype="object")


def _concat_missing_columns(df: pd.DataFrame, columns: dict[str, pd.Series | float]) -> pd.DataFrame:
    missing = {col: value for col, value in columns.items() if col not in df.columns}
    if not missing:
        return df
    return pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1).copy()


def _binary_biotech_flag(df: pd.DataFrame, min_revenue: float) -> pd.Series:
    industry = (
        df["industry"]
        if "industry" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    revenue = (
        pd.to_numeric(df["total_revenue"], errors="coerce")
        if "total_revenue" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )

    biotech_mask = industry.fillna("").astype(str).str.contains("biotechnology", case=False, regex=False)
    low_revenue_mask = revenue.isna() | (revenue < float(min_revenue))
    return (biotech_mask & low_revenue_mask).astype(float)


def _groupwise_robust_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    min_group_size: int,
) -> pd.Series:
    global_z = robust_zscore(df[value_col])
    result = global_z.copy()

    if group_col not in df.columns:
        return result

    def _apply_group_zscore(chunk: pd.Series) -> pd.Series:
        if chunk.notna().sum() >= min_group_size:
            return robust_zscore(chunk)
        return pd.Series(np.nan, index=chunk.index, dtype=float)

    group_z = df.groupby(group_col, dropna=False)[value_col].transform(_apply_group_zscore)
    has_group_z = group_z.notna()
    result.loc[has_group_z] = group_z.loc[has_group_z]
    return result


def add_factor_zscores(df: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = df.copy()
    group_col = None if config.neutralize_by == "none" else config.neutralize_by

    factor_list = CORE_FACTORS + ["earnings_momentum_signal"]
    if getattr(config, "use_residual_valuation", False):
        factor_list += ["residual_value_signal"]
    if getattr(config, "use_compounder_persistence", False):
        factor_list += ["compounder_persistence_signal"]
    if getattr(config, "use_revision_impulse", False):
        factor_list += ["revision_impulse_signal"]
    if getattr(config, "use_revision_jerk", False):
        factor_list += ["revision_jerk_signal"]
    if getattr(config, "use_estimate_term_structure", False):
        factor_list += ["estimate_term_structure_signal"]
    if getattr(config, "use_growth_acceleration", False):
        factor_list += ["revenue_growth_yoy", "revenue_acceleration"]
    if getattr(config, "use_quality_acceleration", False):
        factor_list += ["quality_acceleration_signal"]
    if getattr(config, "use_price_momentum", False):
        factor_list += ["price_momentum_effective_signal"]
    if getattr(config, "use_news_events", False):
        factor_list += ["news_event_effective_signal"]
    if getattr(config, "use_news_shock", False):
        factor_list += ["news_shock_signal"]
    if getattr(config, "use_peer_relative_anomalies", False):
        factor_list += ["peer_relative_anomaly_signal"]
    if getattr(config, "use_capital_allocation_quality", False):
        factor_list += ["capital_allocation_quality_signal"]
    if getattr(config, "use_investment_restraint", False):
        factor_list += ["investment_restraint_signal"]
    if getattr(config, "use_accrual_quality", False):
        factor_list += ["accrual_quality_signal"]
    if getattr(config, "use_insider_conviction", False):
        factor_list += ["insider_conviction_signal"]
    if getattr(config, "use_news_theme_drift", False):
        factor_list += ["news_theme_drift_signal"]
    if getattr(config, "use_employee_efficiency", False):
        factor_list += ["revenue_per_employee", "gross_profit_per_employee"]
    if getattr(config, "use_technical_momentum", False):
        factor_list += ["technical_momentum_signal"]

    for factor in factor_list:
        z_col = f"z_{factor}"
        if factor not in out.columns:
            out[z_col] = np.nan
            continue

        if group_col and group_col in out.columns:
            out[z_col] = _groupwise_robust_zscore(out, factor, group_col, config.min_group_size)
        else:
            out[z_col] = robust_zscore(out[factor])

    return out


def _add_median_rows(rows: list[dict], frame: pd.DataFrame, prefix: str, columns: list[str]) -> None:
    for col in columns:
        if col in frame.columns:
            rows.append(
                {
                    "metric": f"{prefix}{col}",
                    "value": float(pd.to_numeric(frame[col], errors="coerce").median()),
                }
            )


def _resolved_beneish_hard_filter_threshold(config: RankerConfig, observed_universe_size: int) -> tuple[float, bool]:
    universe_size = int(getattr(config, "universe_size", 0) or 0)
    if universe_size <= 0:
        universe_size = int(observed_universe_size)

    is_large_universe = universe_size >= LARGE_UNIVERSE_MIN_SIZE
    threshold = (
        LARGE_UNIVERSE_BENEISH_HARD_FILTER_THRESHOLD if is_large_universe else BASE_BENEISH_HARD_FILTER_THRESHOLD
    )
    return float(threshold), bool(is_large_universe)


def _series_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _coverage_scale(series: pd.Series, floor: float = 0.0) -> pd.Series:
    clipped = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if floor <= 0.0:
        return clipped
    return clipped.where(clipped <= 0.0, floor + (1.0 - floor) * clipped)


def _depth_scale(series: pd.Series, full_count: float, floor: float = 0.0) -> pd.Series:
    if full_count <= 0:
        return pd.Series(1.0, index=series.index, dtype=float)
    clipped = (pd.to_numeric(series, errors="coerce").fillna(0.0) / float(full_count)).clip(0.0, 1.0)
    if floor <= 0.0:
        return clipped
    return clipped.where(clipped <= 0.0, floor + (1.0 - floor) * clipped)


def _normalize_weight_block(
    relative_weights: dict[str, pd.Series],
    total_share: pd.Series,
) -> dict[str, pd.Series]:
    if not relative_weights:
        return {}

    frame = pd.DataFrame(
        {
            name: pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
            for name, series in relative_weights.items()
        },
        index=total_share.index,
    )
    weight_sum = frame.sum(axis=1).replace(0.0, np.nan)
    normalized = frame.div(weight_sum, axis=0).fillna(0.0)
    total = pd.to_numeric(total_share, errors="coerce").fillna(0.0).clip(lower=0.0)
    return {name: normalized[name] * total for name in frame.columns}


def _peer_level_scale(series: pd.Series, mapping: dict[str, float], default: float = 0.0) -> pd.Series:
    labels = series.fillna("").astype(str).str.lower()
    scaled = labels.map({str(key).lower(): float(value) for key, value in mapping.items()})
    return pd.to_numeric(scaled, errors="coerce").fillna(float(default)).clip(lower=0.0)


def _risk_excess_scale(series: pd.Series, trigger: float, ceiling: float = 2.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    denom = max(1e-9, float(ceiling) - float(trigger))
    return ((numeric - float(trigger)) / denom).clip(0.0, 1.0)


def _winsorize_numeric(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lo = float(valid.quantile(lower))
    hi = float(valid.quantile(upper))
    return numeric.clip(lower=lo, upper=hi)


def _ols_residuals(
    frame: pd.DataFrame,
    response_col: str,
    predictor_cols: list[str],
    *,
    winsorize: bool = False,
) -> pd.Series:
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    usable_predictors = [col for col in predictor_cols if col in frame.columns and frame[col].notna().any()]
    if response_col not in frame.columns or not usable_predictors:
        return residuals

    design = frame[[response_col] + usable_predictors].apply(pd.to_numeric, errors="coerce").dropna()
    min_required = max(12, len(usable_predictors) + 4)
    if len(design) < min_required:
        return residuals

    if winsorize:
        for col in design.columns:
            design[col] = _winsorize_numeric(design[col])

    x = np.column_stack([np.ones(len(design), dtype=float)] + [design[col].to_numpy(dtype=float) for col in usable_predictors])
    y = design[response_col].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return residuals

    fitted = x @ beta
    residuals.loc[design.index] = y - fitted
    if winsorize and residuals.notna().any():
        residuals.loc[residuals.notna()] = _winsorize_numeric(residuals.dropna()).reindex(residuals.dropna().index)
    return residuals


def _huber_residuals(
    frame: pd.DataFrame,
    response_col: str,
    predictor_cols: list[str],
    *,
    min_samples: int = 16,
) -> tuple[pd.Series, bool]:
    """
    Compute residuals using Huber regression with standardized predictors.
    Returns (residuals, used_fallback) tuple.
    Falls back to winsorized OLS on fit failure or insufficient samples.
    """
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    used_fallback = False

    usable_predictors = [col for col in predictor_cols if col in frame.columns and frame[col].notna().any()]
    if response_col not in frame.columns or not usable_predictors:
        return residuals, used_fallback

    design = frame[[response_col] + usable_predictors].apply(pd.to_numeric, errors="coerce").dropna()
    min_required = max(min_samples, len(usable_predictors) + 5)
    if len(design) < min_required:
        return residuals, used_fallback

    if not SKLEARN_AVAILABLE:
        # Fall back to winsorized OLS if sklearn not available
        fallback_residuals = _ols_residuals(frame, response_col, predictor_cols, winsorize=True)
        return fallback_residuals, True

    try:
        # Standardize predictors and response
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_raw = design[usable_predictors].to_numpy(dtype=float)
        y_raw = design[response_col].to_numpy(dtype=float).reshape(-1, 1)

        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw).ravel()

        # Winsorize scaled values for additional robustness
        for i in range(X_scaled.shape[1]):
            col_data = X_scaled[:, i]
            q05, q95 = np.percentile(col_data, [5, 95])
            X_scaled[:, i] = np.clip(col_data, q05, q95)

        y_q05, y_q95 = np.percentile(y_scaled, [5, 95])
        y_scaled = np.clip(y_scaled, y_q05, y_q95)

        # Fit Huber regression
        huber = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)
        huber.fit(X_scaled, y_scaled)

        # Compute residuals in original scale
        y_pred_scaled = huber.predict(X_scaled)
        residuals_scaled = y_scaled - y_pred_scaled
        residuals_std = scaler_y.scale_[0] if hasattr(scaler_y, 'scale_') and len(scaler_y.scale_) > 0 else 1.0
        residuals_original = residuals_scaled * residuals_std

        residuals.loc[design.index] = residuals_original

    except Exception:
        # Fall back to winsorized OLS on any error
        fallback_residuals = _ols_residuals(frame, response_col, predictor_cols, winsorize=True)
        residuals = fallback_residuals
        used_fallback = True

    return residuals, used_fallback


def _residualize_against_predictors(
    response: pd.Series,
    predictors: dict[str, pd.Series],
    *,
    min_samples: int = 12,
) -> tuple[pd.Series, float]:
    response_numeric = pd.to_numeric(response, errors="coerce")
    frame = pd.DataFrame({"response": response_numeric}, index=response.index)
    predictor_cols: list[str] = []
    overlap_values: list[float] = []

    for name, series in predictors.items():
        predictor_numeric = pd.to_numeric(series, errors="coerce")
        if predictor_numeric.notna().sum() < 2:
            continue
        frame[name] = predictor_numeric
        predictor_cols.append(name)

        paired = pd.concat([response_numeric, predictor_numeric], axis=1).dropna()
        if len(paired) >= 3 and paired.iloc[:, 0].nunique(dropna=True) > 1 and paired.iloc[:, 1].nunique(dropna=True) > 1:
            corr = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
            if pd.notna(corr):
                overlap_values.append(abs(float(corr)))

    if not predictor_cols:
        return response_numeric, 0.0

    residuals = _ols_residuals(frame, "response", predictor_cols)
    if residuals.notna().sum() < max(min_samples, len(predictor_cols) + 4):
        return response_numeric, max(overlap_values, default=0.0)

    residualized = robust_zscore(residuals).clip(-2.0, 2.0)
    out = response_numeric.copy()
    out.loc[residualized.dropna().index] = residualized.dropna()
    return out, max(overlap_values, default=0.0)


def _build_residual_value_signal(eligible: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = eligible.copy()
    if "residual_value_signal" in out.columns:
        out["residual_value_signal"] = pd.to_numeric(out["residual_value_signal"], errors="coerce")
    else:
        out["residual_value_signal"] = np.nan
    if "residual_value_has_coverage" in out.columns:
        out["residual_value_has_coverage"] = pd.to_numeric(out["residual_value_has_coverage"], errors="coerce").fillna(0.0)
    else:
        out["residual_value_has_coverage"] = 0.0
    if "residual_value_peer_level" not in out.columns:
        out["residual_value_peer_level"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        out["residual_value_peer_level"] = out["residual_value_peer_level"].astype("object")

    # Track residual regression backend and fallback usage
    out["residual_regression_backend"] = "huber" if SKLEARN_AVAILABLE else "ols"
    out["residual_regression_fallback"] = 0.0

    use_v2 = str(getattr(config, "alpha_factor_spec", "legacy") or "legacy").lower() == "v2"

    predictors = [
        col
        for col in ["gross_profitability", "revenue_growth_yoy", "revenue_acceleration", "revision_impulse_signal"]
        if col in out.columns and out[col].notna().any()
    ]
    if use_v2 and "estimate_term_structure_signal" in out.columns and out["estimate_term_structure_signal"].notna().any():
        predictors.append("estimate_term_structure_signal")
    if not predictors or "adjusted_book_to_market" not in out.columns:
        out["residual_regression_backend"] = "none"
        return out

    # Use Huber regression with fallback to winsorized OLS
    global_residuals, global_fallback = _huber_residuals(out, "adjusted_book_to_market", predictors, min_samples=16)

    industry_residuals: dict[object, pd.Series] = {}
    industry_fallback: dict[object, bool] = {}
    if "industry" in out.columns:
        for industry, idx in out.groupby("industry", dropna=False).groups.items():
            subset = out.loc[list(idx)].copy()
            residuals, used_fallback = _huber_residuals(subset, "adjusted_book_to_market", predictors, min_samples=16)
            if residuals.notna().sum() >= 12:
                industry_residuals[industry] = residuals
                industry_fallback[industry] = used_fallback

    sector_residuals: dict[object, pd.Series] = {}
    sector_fallback: dict[object, bool] = {}
    if "sector" in out.columns:
        for sector, idx in out.groupby("sector", dropna=False).groups.items():
            subset = out.loc[list(idx)].copy()
            residuals, used_fallback = _huber_residuals(subset, "adjusted_book_to_market", predictors, min_samples=16)
            if residuals.notna().sum() >= 12:
                sector_residuals[sector] = residuals
                sector_fallback[sector] = used_fallback

    fallback_count = 0
    total_assigned = 0

    for idx, row in out.iterrows():
        industry = row.get("industry") if "industry" in out.columns else None
        sector = row.get("sector") if "sector" in out.columns else None

        assigned = False
        if industry in industry_residuals and pd.notna(industry_residuals[industry].get(idx)):
            out.at[idx, "residual_value_signal"] = float(industry_residuals[industry].loc[idx])
            out.at[idx, "residual_value_has_coverage"] = 1.0
            out.at[idx, "residual_value_peer_level"] = "industry"
            if industry_fallback.get(industry, False):
                fallback_count += 1
            assigned = True
        elif sector in sector_residuals and pd.notna(sector_residuals[sector].get(idx)):
            out.at[idx, "residual_value_signal"] = float(sector_residuals[sector].loc[idx])
            out.at[idx, "residual_value_has_coverage"] = 1.0
            out.at[idx, "residual_value_peer_level"] = "sector"
            if sector_fallback.get(sector, False):
                fallback_count += 1
            assigned = True
        elif pd.notna(global_residuals.get(idx)):
            out.at[idx, "residual_value_signal"] = float(global_residuals.loc[idx])
            out.at[idx, "residual_value_has_coverage"] = 1.0
            out.at[idx, "residual_value_peer_level"] = "global"
            if global_fallback:
                fallback_count += 1
            assigned = True

        if assigned:
            total_assigned += 1

    # Store fallback share for diagnostics
    if total_assigned > 0:
        out["share_residual_huber_fallback"] = fallback_count / total_assigned
    else:
        out["share_residual_huber_fallback"] = 0.0

    return out


def _groupwise_only_robust_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    min_group_size: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if group_col not in df.columns or value_col not in df.columns:
        return result

    numeric_col = pd.to_numeric(df[value_col], errors="coerce")

    def _apply_group(chunk: pd.Series) -> pd.Series:
        if chunk.notna().sum() < min_group_size:
            return pd.Series(np.nan, index=chunk.index, dtype=float)
        return robust_zscore(chunk)

    result = numeric_col.groupby(df[group_col], dropna=False).transform(_apply_group)
    return result


def _build_peer_relative_anomaly_signal(eligible: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = eligible.copy()
    out["peer_relative_anomaly_signal"] = np.nan
    out["peer_relative_anomaly_has_coverage"] = 0.0
    out["peer_relative_anomaly_peer_level"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["peer_relative_margin_component"] = np.nan
    out["peer_relative_reinvestment_component"] = np.nan
    out["peer_relative_estimate_component"] = np.nan
    out["peer_relative_dilution_component"] = np.nan
    out["peer_relative_ownership_component"] = np.nan

    component_inputs = {
        "peer_relative_margin_component": "peer_margin_trend_input",
        "peer_relative_reinvestment_component": "peer_reinvestment_efficiency_input",
        "peer_relative_estimate_component": "peer_estimate_drift_input",
        "peer_relative_dilution_component": "peer_dilution_discipline_input",
        "peer_relative_ownership_component": "peer_ownership_breadth_input",
    }
    available_inputs = {
        output_col: input_col
        for output_col, input_col in component_inputs.items()
        if input_col in out.columns and pd.to_numeric(out[input_col], errors="coerce").notna().any()
    }
    if len(available_inputs) < 2:
        return out

    min_group_size = 12
    global_scores = {
        output_col: robust_zscore(pd.to_numeric(out[input_col], errors="coerce"))
        for output_col, input_col in available_inputs.items()
    }
    sector_scores = {
        output_col: _groupwise_only_robust_zscore(out, input_col, "sector", min_group_size)
        for output_col, input_col in available_inputs.items()
    }
    industry_scores = {
        output_col: _groupwise_only_robust_zscore(out, input_col, "industry", min_group_size)
        for output_col, input_col in available_inputs.items()
    }

    for idx in out.index:
        chosen_level = None
        chosen_components: dict[str, float] = {}
        for level_name, score_map in [
            ("industry", industry_scores),
            ("sector", sector_scores),
            ("global", global_scores),
        ]:
            candidate_components = {
                output_col: float(series.loc[idx])
                for output_col, series in score_map.items()
                if pd.notna(series.loc[idx])
            }
            if len(candidate_components) >= 2:
                chosen_level = level_name
                chosen_components = candidate_components
                break

        if not chosen_components:
            continue

        for output_col, value in chosen_components.items():
            out.at[idx, output_col] = value
        out.at[idx, "peer_relative_anomaly_signal"] = float(sum(chosen_components.values()) / len(chosen_components))
        out.at[idx, "peer_relative_anomaly_has_coverage"] = float(min(1.0, len(chosen_components) / 5.0))
        out.at[idx, "peer_relative_anomaly_peer_level"] = chosen_level

    return out


def _groupwise_peer_signal(
    frame: pd.DataFrame,
    value_col: str,
    weight_col: str,
    group_col: str,
) -> Tuple[pd.Series, pd.Series]:
    signal = pd.Series(0.0, index=frame.index, dtype=float)
    coverage = pd.Series(0.0, index=frame.index, dtype=float)
    if group_col not in frame.columns or value_col not in frame.columns or weight_col not in frame.columns:
        return signal, coverage

    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    active = (weights > 0.0).astype(float)
    eff_weights = weights * (1.0 + values.abs()) * active

    # Vectorized leave-one-out via group totals
    grp = frame[group_col]
    group_total_weight = eff_weights.groupby(grp, dropna=False).transform("sum")
    group_total_signal = (values * eff_weights).groupby(grp, dropna=False).transform("sum")
    group_active_count = active.groupby(grp, dropna=False).transform("sum")
    group_size = grp.groupby(grp, dropna=False).transform("count")

    own_weight = eff_weights
    peer_weight = group_total_weight - own_weight
    peer_signal_total = group_total_signal - values * own_weight
    peer_count = group_active_count - (own_weight > 0.0).astype(float)

    valid = (peer_weight > 0.0) & (peer_count > 0) & (group_size > 1)
    signal = (peer_signal_total / peer_weight.replace(0.0, np.nan)).where(valid, 0.0).fillna(0.0)
    coverage = (peer_count / 5.0).clip(0.0, 1.0).where(valid, 0.0).fillna(0.0)

    return signal, coverage


def _news_macro_multiplier(news_signal: pd.Series, macro_state: str) -> pd.Series:
    state = (macro_state or "neutral").lower()
    signal = pd.to_numeric(news_signal, errors="coerce").fillna(0.0)
    multiplier = pd.Series(1.0, index=signal.index, dtype=float)

    if state == "expansion":
        multiplier.loc[signal > 0.0] = 1.12
        multiplier.loc[signal < 0.0] = 0.92
    elif state == "defensive":
        multiplier.loc[signal > 0.0] = 0.92
        multiplier.loc[signal < 0.0] = 1.15
    elif state == "inflation_stress":
        multiplier.loc[signal > 0.0] = 0.90
        multiplier.loc[signal < 0.0] = 1.12

    return multiplier


def _build_news_experiment_metrics(frame: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = frame.copy()
    raw_signal = _series_or_nan(out, "news_event_signal")
    raw_signal_filled = raw_signal.fillna(0.0)
    novelty_score = _series_or_zero(out, "news_novelty_score").clip(0.0, 1.0)
    saturation_score = _series_or_zero(out, "news_saturation_score").clip(0.0, 1.0)
    article_count = _series_or_zero(out, "news_article_count_recent")
    breadth = _series_or_zero(out, "news_event_breadth")
    article_scale = (article_count / max(1.0, float(getattr(config, "min_news_articles", 1)))).clip(0.0, 1.0)
    breadth_scale = (breadth / 3.0).clip(0.0, 1.0)
    base_peer_weight = (
        article_scale * (0.55 + 0.25 * breadth_scale + 0.20 * novelty_score) * (1.0 + raw_signal_filled.abs())
    ).clip(lower=0.0)

    out["news_novelty_multiplier"] = 1.0
    out["news_saturation_multiplier"] = 1.0
    if getattr(config, "use_news_novelty_saturation", False):
        out["news_novelty_multiplier"] = 0.85 + 0.30 * novelty_score
        out["news_saturation_multiplier"] = 1.0 - 0.35 * saturation_score

    out["news_peer_spillover_signal"] = 0.0
    out["news_peer_coverage"] = 0.0
    if getattr(config, "use_news_peer_spillover", False):
        peer_frame = out.copy()
        peer_frame["_tmp_news_peer_weight"] = base_peer_weight
        industry_signal, industry_coverage = _groupwise_peer_signal(
            peer_frame,
            value_col="news_event_signal",
            weight_col="_tmp_news_peer_weight",
            group_col="industry",
        )
        sector_signal, sector_coverage = _groupwise_peer_signal(
            peer_frame,
            value_col="news_event_signal",
            weight_col="_tmp_news_peer_weight",
            group_col="sector",
        )
        out["news_peer_spillover_signal"] = (0.70 * industry_signal + 0.30 * sector_signal).clip(-1.0, 1.0)
        out["news_peer_coverage"] = (0.70 * industry_coverage + 0.30 * sector_coverage).clip(0.0, 1.0)

    out["news_confirmation_signal"] = 0.0
    out["news_confirmation_coverage"] = 0.0
    if getattr(config, "use_news_confirmation", False):
        revision_support = np.tanh(4.0 * _series_or_nan(out, "revision_impulse_signal").fillna(0.0))
        pead_support = np.tanh(4.0 * _series_or_nan(out, "pead_signal").fillna(0.0))
        sue_support = np.tanh(0.75 * _series_or_nan(out, "sue_signal").fillna(0.0))
        support_signal = 0.55 * revision_support + 0.30 * pead_support + 0.15 * sue_support
        support_coverage = (
            0.55 * _series_or_zero(out, "revision_impulse_has_coverage").clip(0.0, 1.0)
            + 0.30 * _series_or_zero(out, "pead_has_setup_coverage").clip(0.0, 1.0)
            + 0.15 * _series_or_zero(out, "sue_has_coverage").clip(0.0, 1.0)
        ).clip(0.0, 1.0)
        out["news_confirmation_signal"] = (raw_signal_filled * support_signal).clip(-1.0, 1.0)
        out["news_confirmation_coverage"] = support_coverage.where(support_signal.abs() > 0.0, 0.0)

    effective_signal = raw_signal_filled.copy()
    if getattr(config, "use_news_novelty_saturation", False):
        effective_signal = (
            effective_signal
            * pd.to_numeric(out["news_novelty_multiplier"], errors="coerce").fillna(1.0)
            * pd.to_numeric(out["news_saturation_multiplier"], errors="coerce").fillna(1.0)
        )
    if getattr(config, "use_news_peer_spillover", False):
        effective_signal = (
            effective_signal
            + float(getattr(config, "news_peer_spillover_weight", 0.0))
            * pd.to_numeric(out["news_peer_spillover_signal"], errors="coerce").fillna(0.0)
        )
    if getattr(config, "use_news_confirmation", False):
        effective_signal = (
            effective_signal
            + float(getattr(config, "news_confirmation_weight", 0.0))
            * pd.to_numeric(out["news_confirmation_signal"], errors="coerce").fillna(0.0)
        )

    out["news_macro_multiplier"] = 1.0
    if getattr(config, "use_news_macro_weighting", False):
        out["news_macro_multiplier"] = _news_macro_multiplier(
            effective_signal,
            getattr(config, "macro_state", "neutral"),
        )
        effective_signal = effective_signal * pd.to_numeric(out["news_macro_multiplier"], errors="coerce").fillna(1.0)

    has_signal = raw_signal.notna() | (pd.to_numeric(out["news_peer_spillover_signal"], errors="coerce").fillna(0.0).abs() > 1e-9)
    out["news_event_effective_signal"] = effective_signal.clip(-1.5, 1.5).where(has_signal, np.nan)

    return out


def _add_pairwise_corr_rows(rows: list[dict], frame: pd.DataFrame, prefix: str, columns: list[str]) -> None:
    usable = [col for col in columns if col in frame.columns]
    for idx, left in enumerate(usable):
        left_series = pd.to_numeric(frame[left], errors="coerce")
        for right in usable[idx + 1 :]:
            right_series = pd.to_numeric(frame[right], errors="coerce")
            paired = pd.concat([left_series, right_series], axis=1).dropna()
            if len(paired) < 3:
                continue
            if paired.iloc[:, 0].nunique(dropna=True) < 2 or paired.iloc[:, 1].nunique(dropna=True) < 2:
                continue
            corr = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
            rows.append(
                {
                    "metric": f"{prefix}{left}__{right}",
                    "value": float(corr) if pd.notna(corr) else float("nan"),
                }
            )


def _clipped_life_cycle_strength(config: RankerConfig) -> float:
    if not getattr(config, "use_life_cycle", False):
        return 0.0
    return float(np.clip(float(getattr(config, "life_cycle_tilt_strength", 0.0) or 0.0), 0.0, 1.0))


def _build_life_cycle_context(
    eligible: pd.DataFrame,
    config: RankerConfig,
    core_weights: Dict[str, float],
) -> pd.DataFrame:
    out = eligible.copy()

    strength = _clipped_life_cycle_strength(config)
    gp = robust_zscore(_series_or_nan(out, "gross_profitability")).clip(-2.0, 2.0).fillna(0.0)
    sy = robust_zscore(_series_or_nan(out, "shareholder_yield")).clip(-2.0, 2.0).fillna(0.0)
    value = robust_zscore(_series_or_nan(out, "adjusted_book_to_market")).clip(-2.0, 2.0).fillna(0.0)
    pead = robust_zscore(_series_or_nan(out, "pead_signal")).clip(-2.0, 2.0).fillna(0.0)
    sue = robust_zscore(_series_or_nan(out, "sue_signal")).clip(-2.0, 2.0).fillna(0.0)
    revision = robust_zscore(_series_or_nan(out, "revision_impulse_signal")).clip(-2.0, 2.0).fillna(0.0)
    revenue_growth = robust_zscore(_series_or_nan(out, "revenue_growth_yoy")).clip(-2.0, 2.0).fillna(0.0)
    revenue_acceleration = robust_zscore(_series_or_nan(out, "revenue_acceleration")).clip(-2.0, 2.0).fillna(0.0)
    momentum = robust_zscore(_series_or_nan(out, "price_momentum_effective_signal")).clip(-2.0, 2.0).fillna(0.0)
    forensic = _series_or_nan(out, "forensic_penalty").clip(-2.0, 2.0).fillna(0.0)
    piotroski = ((_series_or_nan(out, "piotroski_score").fillna(5.0) - 5.0) / 2.0).clip(-2.0, 2.0)

    out["life_cycle_growth_score"] = (
        0.35 * revenue_acceleration
        + 0.20 * revenue_growth
        + 0.20 * sue
        + 0.15 * revision
        + 0.10 * gp
        + 0.10 * momentum
        - 0.20 * sy
        - 0.10 * value
    )
    out["life_cycle_mature_score"] = (
        0.55 * sy
        + 0.45 * gp
        + 0.20 * piotroski
        + 0.10 * value
        - 0.10 * revenue_acceleration
        - 0.10 * revision
        - 0.05 * momentum
    )
    out["life_cycle_recovery_score"] = (
        0.45 * value
        + 0.25 * momentum
        + 0.15 * revenue_acceleration
        - 0.25 * gp
        - 0.15 * sy
        - 0.15 * revision
        + 0.20 * forensic
    )

    score_frame = out[
        ["life_cycle_growth_score", "life_cycle_mature_score", "life_cycle_recovery_score"]
    ].copy()
    score_frame.columns = list(LIFE_CYCLE_STAGES)
    chosen_stage = score_frame.idxmax(axis=1)
    sorted_scores = np.sort(score_frame.to_numpy(dtype=float), axis=1)
    confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
    out["life_cycle_confidence"] = confidence
    out["life_cycle_stage"] = np.where(confidence >= 0.15, chosen_stage, "mature")
    out["life_cycle_tilt_strength_applied"] = strength

    out["life_cycle_core_weight_shareholder_yield"] = core_weights["shareholder_yield"]
    out["life_cycle_core_weight_gross_profitability"] = core_weights["gross_profitability"]
    out["life_cycle_core_weight_adjusted_book_to_market"] = core_weights["adjusted_book_to_market"]
    out["life_cycle_pead_multiplier"] = 1.0
    out["life_cycle_growth_multiplier"] = 1.0
    out["life_cycle_momentum_multiplier"] = 1.0
    out["life_cycle_revision_impulse_multiplier"] = 1.0
    out["life_cycle_forensic_multiplier"] = 1.0

    # Macro-aware stage templates: expansion favors growth/revision, defensive/inflation stress favor quality/preservation
    macro_state = (getattr(config, "macro_state", "neutral") or "neutral").lower()

    # Base multipliers by stage
    base_templates = {
        "growth": {
            "shareholder_yield": 1.0 - 0.45 * strength,
            "gross_profitability": 1.0 + 0.25 * strength,
            "adjusted_book_to_market": 1.0 - 0.20 * strength,
            "pead_multiplier": 1.0 + 0.50 * strength,
            "growth_multiplier": 1.0 + 0.50 * strength,
            "momentum_multiplier": 1.0 + 0.25 * strength,
            "revision_multiplier": 1.0 + 0.50 * strength,
            "forensic_multiplier": 1.0,
        },
        "mature": {
            "shareholder_yield": 1.0 + 0.25 * strength,
            "gross_profitability": 1.0 + 0.20 * strength,
            "adjusted_book_to_market": 1.0 - 0.10 * strength,
            "pead_multiplier": 1.0 - 0.10 * strength,
            "growth_multiplier": 1.0 - 0.15 * strength,
            "momentum_multiplier": 1.0 - 0.10 * strength,
            "revision_multiplier": 1.0 - 0.10 * strength,
            "forensic_multiplier": 1.0 + 0.10 * strength,
        },
        "recovery": {
            "shareholder_yield": 1.0 - 0.10 * strength,
            "gross_profitability": 1.0 - 0.20 * strength,
            "adjusted_book_to_market": 1.0 + 0.35 * strength,
            "pead_multiplier": 1.0 - 0.25 * strength,
            "growth_multiplier": 1.0 + 0.10 * strength,
            "momentum_multiplier": 1.0 + 0.35 * strength,
            "revision_multiplier": 1.0 - 0.25 * strength,
            "forensic_multiplier": 1.0 + 0.50 * strength,
        },
    }

    # Macro tilt adjustments
    if macro_state == "expansion":
        # Growth stage: stronger growth/revision/momentum tilt
        # Recovery stage: lighter value emphasis
        stage_templates = {
            "growth": {
                **base_templates["growth"],
                "growth_multiplier": 1.0 + 0.65 * strength,
                "revision_multiplier": 1.0 + 0.60 * strength,
                "momentum_multiplier": 1.0 + 0.35 * strength,
            },
            "mature": base_templates["mature"],
            "recovery": {
                **base_templates["recovery"],
                "adjusted_book_to_market": 1.0 + 0.25 * strength,  # lighter value
                "shareholder_yield": 1.0 - 0.05 * strength,  # lighter yield penalty
            },
        }
    elif macro_state in ("defensive", "inflation_stress"):
        # Mature stage: reduce dividend/yield bonus, favor balance-sheet/quality
        # Growth stage: less aggressive growth tilt
        stage_templates = {
            "growth": {
                **base_templates["growth"],
                "growth_multiplier": 1.0 + 0.35 * strength,
                "gross_profitability": 1.0 + 0.35 * strength,
            },
            "mature": {
                **base_templates["mature"],
                "shareholder_yield": 1.0 + 0.15 * strength,  # reduced from 0.25
                "gross_profitability": 1.0 + 0.30 * strength,  # increased quality emphasis
                "forensic_multiplier": 1.0 + 0.20 * strength,  # stronger forensic emphasis
            },
            "recovery": base_templates["recovery"],
        }
    else:
        stage_templates = base_templates

    for stage, template in stage_templates.items():
        mask = out["life_cycle_stage"] == stage
        if not mask.any():
            continue

        stage_weights = {
            factor: max(0.01, float(core_weights[factor]) * float(template[factor]))
            for factor in CORE_FACTORS
        }
        weight_sum = sum(stage_weights.values())
        for factor in CORE_FACTORS:
            out.loc[mask, f"life_cycle_core_weight_{factor}"] = stage_weights[factor] / weight_sum

        out.loc[mask, "life_cycle_pead_multiplier"] = float(template["pead_multiplier"])
        out.loc[mask, "life_cycle_growth_multiplier"] = float(template["growth_multiplier"])
        out.loc[mask, "life_cycle_momentum_multiplier"] = float(template["momentum_multiplier"])
        out.loc[mask, "life_cycle_revision_impulse_multiplier"] = float(template["revision_multiplier"])
        out.loc[mask, "life_cycle_forensic_multiplier"] = float(template["forensic_multiplier"])

    return out


def _split_core_component_weights(
    eligible: pd.DataFrame,
    config: RankerConfig,
    shareholder_weight: pd.Series,
    gross_profitability_weight: pd.Series,
    adjusted_book_weight: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    residual_weight = pd.Series(0.0, index=eligible.index, dtype=float)
    compounder_weight = pd.Series(0.0, index=eligible.index, dtype=float)
    use_v2 = str(getattr(config, "alpha_factor_spec", "legacy") or "legacy").lower() == "v2"
    residual_share = 0.55 if use_v2 else 0.45
    compounder_share = 0.25 if use_v2 else 0.30

    if getattr(config, "use_residual_valuation", False) and "residual_value_signal" in eligible.columns:
        has_signal = eligible["residual_value_signal"].notna()
        residual_weight.loc[has_signal] = adjusted_book_weight.loc[has_signal] * residual_share
        adjusted_book_weight.loc[has_signal] = adjusted_book_weight.loc[has_signal] * (1.0 - residual_share)

    if getattr(config, "use_compounder_persistence", False) and "compounder_persistence_signal" in eligible.columns:
        has_signal = eligible["compounder_persistence_signal"].notna()
        compounder_weight.loc[has_signal] = gross_profitability_weight.loc[has_signal] * compounder_share
        gross_profitability_weight.loc[has_signal] = gross_profitability_weight.loc[has_signal] * (1.0 - compounder_share)

    return (
        shareholder_weight,
        gross_profitability_weight,
        adjusted_book_weight,
        residual_weight,
        compounder_weight,
    )


def _build_recovery_transition_metrics(
    eligible: pd.DataFrame,
    config: RankerConfig,
    revision_component: pd.Series,
    momentum_component: pd.Series,
    estimate_component: pd.Series,
) -> pd.DataFrame:
    out = eligible.copy()
    out["recovery_transition_signal"] = np.nan
    out["recovery_transition_has_coverage"] = 0.0

    if not getattr(config, "use_recovery_transition", False):
        return out

    use_v2 = str(getattr(config, "alpha_factor_spec", "legacy") or "legacy").lower() == "v2"
    margin_component = robust_zscore(_series_or_nan(out, "recovery_margin_inflection")).clip(-2.0, 2.0)
    leverage_component = robust_zscore(_series_or_nan(out, "recovery_leverage_improvement")).clip(-2.0, 2.0)
    accrual_component = robust_zscore(_series_or_nan(out, "recovery_accrual_improvement")).clip(-2.0, 2.0)
    technical_component = momentum_component
    technical_coverage = _series_or_zero(out, "price_momentum_signal_coverage").clip(0.0, 1.0)
    if use_v2:
        technical_component = (
            0.45 * robust_zscore(_series_or_nan(out, "price_to_200dma")).clip(-2.0, 2.0).fillna(0.0)
            + 0.30 * robust_zscore(_series_or_nan(out, "recency_ratio")).clip(-2.0, 2.0).fillna(0.0)
            + 0.25 * robust_zscore(-_series_or_nan(out, "distance_from_high")).clip(-2.0, 2.0).fillna(0.0)
        )
        technical_coverage = (
            0.40 * _series_or_nan(out, "price_to_200dma").notna().astype(float)
            + 0.35 * _series_or_nan(out, "recency_ratio").notna().astype(float)
            + 0.25 * _series_or_nan(out, "distance_from_high").notna().astype(float)
        ).clip(0.0, 1.0)

    if use_v2:
        raw_signal = (
            0.22 * margin_component.fillna(0.0)
            + 0.18 * leverage_component.fillna(0.0)
            + 0.15 * accrual_component.fillna(0.0)
            + 0.22 * revision_component.fillna(0.0)
            + 0.13 * estimate_component.fillna(0.0)
            + 0.10 * technical_component.fillna(0.0)
        )
        coverage = (
            0.18 * _series_or_nan(out, "recovery_margin_inflection").notna().astype(float)
            + 0.16 * _series_or_nan(out, "recovery_leverage_improvement").notna().astype(float)
            + 0.14 * _series_or_nan(out, "recovery_accrual_improvement").notna().astype(float)
            + 0.22 * _series_or_zero(out, "revision_impulse_has_coverage").clip(0.0, 1.0)
            + 0.15 * _series_or_zero(out, "estimate_term_structure_has_coverage").clip(0.0, 1.0)
            + 0.15 * technical_coverage
        ).clip(0.0, 1.0)
    else:
        raw_signal = (
            0.30 * margin_component.fillna(0.0)
            + 0.20 * leverage_component.fillna(0.0)
            + 0.30 * revision_component.fillna(0.0)
            + 0.20 * technical_component.fillna(0.0)
        )
        coverage = (
            0.25 * _series_or_nan(out, "recovery_margin_inflection").notna().astype(float)
            + 0.20 * _series_or_nan(out, "recovery_leverage_improvement").notna().astype(float)
            + 0.30 * _series_or_zero(out, "revision_impulse_has_coverage").clip(0.0, 1.0)
            + 0.25 * technical_coverage
        ).clip(0.0, 1.0)
    recovery_mask = out.get("life_cycle_stage", pd.Series(index=out.index, dtype="object")).fillna("") == "recovery"
    coverage = coverage.where(recovery_mask, 0.0)
    signal = robust_zscore(raw_signal.where(coverage > 0.0, np.nan)).clip(-2.0, 2.0)

    out["recovery_transition_signal"] = signal.where(coverage > 0.0, np.nan)
    out["recovery_transition_has_coverage"] = coverage
    return out


def build_diagnostics(
    all_rows: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    beneish_gate_threshold: float | None = None,
    large_universe_forensic_mode: bool | None = None,
    universe_size: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    if not all_rows.empty:
        rows.append({"metric": "stage1_count_after_core_filters", "value": float(len(all_rows))})
        if universe_size is not None:
            rows.append({"metric": "collected_universe_size", "value": float(universe_size)})
        rows.append({"metric": "alpha_factor_spec", "value": str(all_rows.get("alpha_factor_spec", pd.Series(["legacy"])).iloc[0])})
        if beneish_gate_threshold is not None:
            rows.append({"metric": "forensic_gate_beneish_threshold", "value": float(beneish_gate_threshold)})
        if large_universe_forensic_mode is not None:
            rows.append(
                {
                    "metric": "large_universe_forensic_mode",
                    "value": float(bool(large_universe_forensic_mode)),
                }
            )

        # Count structural hard exclusions vs soft penalty candidates
        hard_exclusion_mask = ~(
            all_rows.get("passes_size", pd.Series(True, index=all_rows.index))
            & all_rows.get("passes_biotech_gate", pd.Series(True, index=all_rows.index))
            & all_rows.get("passes_factor_gate", pd.Series(True, index=all_rows.index))
        )
        rows.append({"metric": "structural_hard_exclusions", "value": float(hard_exclusion_mask.mean())})

        # Soft penalty candidates (would have been excluded under old hard gates)
        soft_penalty_candidates = (
            (all_rows.get("penalty_trend", 0.0) > 0.0)
            | (all_rows.get("penalty_quality", 0.0) > 0.0)
            | (all_rows.get("penalty_forensic_uncertainty", 0.0) > 0.15)  # above missing penalty
            | (all_rows.get("penalty_core_missing", 0.0) > 0.0)
            | (all_rows.get("penalty_momentum_coverage", 0.0) > 0.0)
        )
        rows.append({"metric": "soft_penalty_candidates", "value": float(soft_penalty_candidates.mean())})

        for col in [
            "passes_size",
            "passes_trend_gate",
            "passes_quality_gate",
            "passes_forensic_gate",
            "passes_factor_gate",
            "passes_momentum_gate",
            "passes_biotech_gate",
            "factor_non_null_count",
            "factor_positive_count",
            "core_factor_imputed_count",
            "core_factor_imputation_flag",
            "imputed_shareholder_yield",
            "imputed_gross_profitability",
            "imputed_adjusted_book_to_market",
            "binary_biotech_flag",
            "penalty_trend",
            "penalty_quality",
            "penalty_forensic_uncertainty",
            "penalty_core_missing",
            "penalty_momentum_coverage",
            "pead_has_setup_coverage",
            "sue_has_coverage",
            "revision_impulse_has_coverage",
            "revision_jerk_has_coverage",
            "estimate_term_structure_has_coverage",
            "revenue_growth_has_coverage",
            "residual_value_has_coverage",
            "compounder_persistence_has_coverage",
            "peer_relative_anomaly_has_coverage",
            "price_momentum_has_coverage",
            "price_momentum_signal_coverage",
            "price_momentum_proxy_used",
            "sentiment_has_coverage",
            "news_has_coverage",
            "news_shock_has_coverage",
            "capital_allocation_quality_has_coverage",
            "recovery_transition_has_coverage",
            "investment_restraint_has_coverage",
            "accrual_quality_has_coverage",
            "quality_acceleration_has_coverage",
            "insider_conviction_has_coverage",
            "news_theme_drift_has_coverage",
            "working_capital_stress_has_coverage",
            "analysis_identity_mismatch",
            "analysis_resolution_error",
            "overlay_error_flag",
            "sentiment_fetch_error",
            "news_fetch_error",
            "news_theme_drift_fetch_error",
            "insider_fetch_error",
            "price_history_fetch_error",
            "beneish_is_missing",
            "beneish_is_pathological_clipped",
            "intangible_adjustment_eligible",
            "intangible_adjustment_applied",
        ]:
            if col not in all_rows.columns:
                continue
            if col in {"factor_non_null_count", "factor_positive_count"}:
                rows.append({"metric": f"median_{col}", "value": float(all_rows[col].median())})
            else:
                rows.append(
                    {
                        "metric": f"share_{col}",
                        "value": float(pd.to_numeric(all_rows[col].astype(object), errors="coerce").fillna(0.0).mean()),
                    }
                )

        if "strict_issuer_resolution_error_count" in all_rows.columns:
            strict_errors = pd.to_numeric(all_rows["strict_issuer_resolution_error_count"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "metric": "share_strict_issuer_resolution_error",
                    "value": float((strict_errors > 0.0).mean()),
                }
            )
            rows.append(
                {
                    "metric": "median_strict_issuer_resolution_error_count",
                    "value": float(strict_errors.median()),
                }
            )

        _add_median_rows(
            rows,
            all_rows,
            "median_",
            [
                "revenue_per_employee",
                "gross_profit_per_employee",
                "piotroski_score",
                "beneish_m_score",
                "accrual_volatility",
                "earnings_momentum_signal",
                "earnings_momentum_coverage",
                "pead_signal",
                "pead_revision_component",
                "sue_signal",
                "sue_surprise_pct",
                "revision_impulse_signal",
                "revision_jerk_signal",
                "revision_jerk_persistence_days",
                "revision_jerk_persistence_pass",
                "revision_impulse_analyst_count",
                "revision_impulse_disagreement_penalty",
                "revision_short_divergence_component",
                "revision_breadth_7d",
                "revision_breadth_30d",
                "revision_breadth_acceleration",
                "float_absorption_signal",
                "squeeze_convexity_signal",
                "short_interest_shares",
                "short_interest_shares_prev",
                "short_interest_ratio",
                "short_interest_pct_float",
                "short_interest_pct_outstanding",
                "short_interest_change",
                "estimate_term_structure_signal",
                "estimate_term_structure_record_count",
                "estimate_term_structure_disagreement_trend",
                "estimate_term_structure_overlap_penalty",
                "revenue_growth_yoy",
                "revenue_acceleration",
                "percent_institutions",
                "institution_holder_count_latest",
                "institution_holder_count_prev",
                "institutional_breadth_delta",
                "institutional_ownership_from_holders",
                "institutional_ownership_delta",
                "institutional_top5_concentration_delta",
                "share_drift_1q",
                "share_drift_4q",
                "share_drift_persistence",
                "receivables_days",
                "receivables_days_prev",
                "receivables_days_delta",
                "inventory_days",
                "inventory_days_prev",
                "inventory_days_delta",
                "payables_days",
                "payables_days_prev",
                "payables_days_delta",
                "cash_conversion_cycle_days",
                "cash_conversion_cycle_days_prev",
                "cash_conversion_cycle_days_delta",
                "cash_conversion_cycle_convexity",
                "residual_value_signal",
                "share_residual_huber_fallback",
                "compounder_persistence_signal",
                "compounder_persistence_measure_count",
                "price_momentum_6m_ex_1m",
                "price_momentum_effective_signal",
                "sentiment_article_count_recent",
                "news_event_signal",
                "news_event_effective_signal",
                "news_event_breadth",
                "news_article_count_recent",
                "news_baseline_signal",
                "news_baseline_article_count",
                "news_article_volume_spike",
                "news_novelty_score",
                "news_saturation_score",
                "news_shock_signal",
                "news_peer_spillover_signal",
                "news_confirmation_signal",
                "news_macro_multiplier",
                "news_theme_drift_signal",
                "news_theme_drift_recent_intensity",
                "news_theme_drift_baseline_intensity",
                "capital_allocation_quality_signal",
                "investment_restraint_signal",
                "investment_restraint_measure_count",
                "accrual_quality_signal",
                "accrual_quality_measure_count",
                "quality_acceleration_signal",
                "quality_acceleration_measure_count",
                "peer_ownership_breadth_input",
                "recovery_margin_inflection",
                "recovery_leverage_improvement",
                "recovery_transition_signal",
                "insider_conviction_signal",
                "insider_ownership_confirmation_component",
                "insider_short_crowding_penalty",
                "working_capital_stress_penalty",
                "working_capital_cycle_stress",
                "accrual_quality_cycle_convexity",
                "financing_dependency_stress",
                "financing_dependency_burn_component",
                "financing_dependency_dilution_component",
                "financing_dependency_debt_component",
                "financing_dependency_revision_component",
                "capital_allocation_financing_dependency_component",
            ],
        )

        _add_median_rows(
            rows,
            all_rows,
            "median_",
            [
                "earnings_signal_confidence",
                "revision_signal_confidence",
                "revision_jerk_signal_confidence",
                "estimate_term_structure_signal_confidence",
                "growth_signal_confidence",
                "momentum_signal_confidence",
                "news_signal_confidence",
                "news_shock_signal_confidence",
                "capital_allocation_signal_confidence",
                "residual_value_signal_confidence",
                "compounder_persistence_signal_confidence",
                "recovery_transition_signal_confidence",
                "investment_restraint_signal_confidence",
                "accrual_quality_signal_confidence",
                "quality_acceleration_signal_confidence",
                "insider_conviction_signal_confidence",
                "news_theme_drift_signal_confidence",
                "technical_momentum_signal_confidence",
                "effective_core_share",
                "effective_optional_share",
                "contrib_shareholder_yield",
                "contrib_gross_profitability",
                "contrib_adjusted_book_to_market",
                "contrib_residual_value",
                "contrib_compounder_persistence",
                "contrib_earnings",
                "contrib_revision_impulse",
                "contrib_revision_jerk",
                "contrib_estimate_term_structure",
                "contrib_growth",
                "contrib_momentum",
                "contrib_news_event",
                "contrib_news_shock",
                "contrib_capital_allocation",
                "contrib_recovery_transition",
                "contrib_investment_restraint",
                "contrib_accrual_quality",
                "contrib_quality_acceleration",
                "contrib_insider_conviction",
                "contrib_news_theme_drift",
                "contrib_technical_momentum",
                "contrib_employee_efficiency",
                "contrib_forensic",
            ],
        )

        if "life_cycle_stage" in all_rows.columns:
            stage_shares = all_rows["life_cycle_stage"].fillna("unknown").value_counts(normalize=True)
            for stage, share in stage_shares.items():
                rows.append({"metric": f"life_cycle_share::{stage}", "value": float(share)})

    if not ranked.empty:
        rows.append({"metric": "final_ranked_count", "value": float(len(ranked))})
        _add_median_rows(
            rows,
            ranked,
            "median_",
            [
                "market_cap",
                "shareholder_yield",
                "gross_profitability",
                "adjusted_book_to_market",
                "earnings_momentum_signal",
                "earnings_momentum_coverage",
                "pead_signal",
                "sue_signal",
                "revision_impulse_signal",
                "revision_jerk_signal",
                "revision_short_divergence_component",
                "revision_breadth_7d",
                "revision_breadth_30d",
                "revision_breadth_acceleration",
                "float_absorption_signal",
                "squeeze_convexity_signal",
                "estimate_term_structure_signal",
                "estimate_term_structure_overlap_penalty",
                "revenue_growth_yoy",
                "revenue_acceleration",
                "short_interest_ratio",
                "short_interest_pct_float",
                "short_interest_change",
                "percent_institutions",
                "institutional_breadth_delta",
                "institutional_ownership_delta",
                "institutional_top5_concentration_delta",
                "share_drift_1q",
                "share_drift_4q",
                "share_drift_persistence",
                "receivables_days",
                "receivables_days_delta",
                "inventory_days",
                "inventory_days_delta",
                "payables_days",
                "payables_days_delta",
                "cash_conversion_cycle_days",
                "cash_conversion_cycle_days_delta",
                "cash_conversion_cycle_convexity",
                "residual_value_signal",
                "share_residual_huber_fallback",
                "residual_regression_backend",
                "compounder_persistence_signal",
                "peer_relative_anomaly_signal",
                "peer_ownership_breadth_input",
                "price_momentum_6m_ex_1m",
                "price_momentum_effective_signal",
                "piotroski_score",
                "revenue_per_employee",
                "gross_profit_per_employee",
                "forensic_penalty",
                "beneish_m_score",
                "accrual_volatility",
                "working_capital_stress_penalty",
                "working_capital_cycle_stress",
                "capital_allocation_quality_signal",
                "capital_allocation_financing_dependency_component",
                "financing_dependency_stress",
                "financing_dependency_burn_component",
                "financing_dependency_dilution_component",
                "financing_dependency_debt_component",
                "financing_dependency_revision_component",
                "investment_restraint_signal",
                "investment_restraint_measure_count",
                "accrual_quality_signal",
                "accrual_quality_measure_count",
                "accrual_quality_cycle_convexity",
                "quality_acceleration_signal",
                "quality_acceleration_measure_count",
                "recovery_transition_signal",
                "penalty_trend",
                "penalty_quality",
                "penalty_forensic_uncertainty",
                "penalty_core_missing",
                "penalty_momentum_coverage",
                "insider_conviction_signal",
                "insider_ownership_confirmation_component",
                "insider_short_crowding_penalty",
                "sentiment_article_count_recent",
                "news_event_signal",
                "news_event_effective_signal",
                "news_event_breadth",
                "news_article_count_recent",
                "news_baseline_signal",
                "news_baseline_article_count",
                "news_article_volume_spike",
                "news_novelty_score",
                "news_saturation_score",
                "news_shock_signal",
                "news_peer_spillover_signal",
                "news_confirmation_signal",
                "news_macro_multiplier",
                "news_theme_drift_signal",
            ],
        )

        _add_median_rows(
            rows,
            ranked,
            "median_",
            [
                "earnings_signal_confidence",
                "revision_signal_confidence",
                "revision_jerk_signal_confidence",
                "estimate_term_structure_signal_confidence",
                "growth_signal_confidence",
                "momentum_signal_confidence",
                "news_signal_confidence",
                "news_shock_signal_confidence",
                "capital_allocation_signal_confidence",
                "residual_value_signal_confidence",
                "compounder_persistence_signal_confidence",
                "peer_relative_anomaly_signal_confidence",
                "recovery_transition_signal_confidence",
                "investment_restraint_signal_confidence",
                "accrual_quality_signal_confidence",
                "quality_acceleration_signal_confidence",
                "insider_conviction_signal_confidence",
                "news_theme_drift_signal_confidence",
                "technical_momentum_signal_confidence",
                "effective_core_share",
                "effective_optional_share",
                "contrib_shareholder_yield",
                "contrib_gross_profitability",
                "contrib_adjusted_book_to_market",
                "contrib_residual_value",
                "contrib_compounder_persistence",
                "contrib_peer_relative_anomaly",
                "contrib_earnings",
                "contrib_revision_impulse",
                "contrib_revision_jerk",
                "contrib_estimate_term_structure",
                "contrib_growth",
                "contrib_momentum",
                "contrib_news_event",
                "contrib_news_shock",
                "contrib_capital_allocation",
                "contrib_recovery_transition",
                "contrib_investment_restraint",
                "contrib_accrual_quality",
                "contrib_quality_acceleration",
                "contrib_insider_conviction",
                "contrib_news_theme_drift",
                "contrib_technical_momentum",
                "contrib_employee_efficiency",
                "contrib_forensic",
            ],
        )

        _add_pairwise_corr_rows(
            rows,
            ranked,
            "ranked_contrib_corr::",
            [
                "contrib_shareholder_yield",
                "contrib_gross_profitability",
                "contrib_adjusted_book_to_market",
                "contrib_residual_value",
                "contrib_compounder_persistence",
                "contrib_peer_relative_anomaly",
                "contrib_earnings",
                "contrib_revision_impulse",
                "contrib_revision_jerk",
                "contrib_estimate_term_structure",
                "contrib_growth",
                "contrib_momentum",
                "contrib_news_event",
                "contrib_news_shock",
                "contrib_capital_allocation",
                "contrib_recovery_transition",
                "contrib_investment_restraint",
                "contrib_accrual_quality",
                "contrib_quality_acceleration",
                "contrib_insider_conviction",
                "contrib_news_theme_drift",
                "contrib_technical_momentum",
                "contrib_forensic",
            ],
        )

        if "life_cycle_stage" in ranked.columns:
            stage_shares = ranked["life_cycle_stage"].fillna("unknown").value_counts(normalize=True)
            for stage, share in stage_shares.items():
                rows.append({"metric": f"final_life_cycle_share::{stage}", "value": float(share)})

        if "sector" in ranked.columns:
            sector_weights = ranked["sector"].fillna("Unknown").value_counts(normalize=True)
            for sector, share in sector_weights.head(12).items():
                rows.append({"metric": f"sector_share::{sector}", "value": float(share)})

    return pd.DataFrame(rows)


def _build_forensic_penalty(eligible: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = eligible.copy()

    risk_components: list[tuple[str, float]] = []
    out["beneish_missing_penalty_applied"] = 0.0
    if config.use_beneish:
        out["z_beneish_risk"] = np.nan
        if "beneish_m_score" in out.columns and out["beneish_m_score"].notna().any():
            out["z_beneish_risk"] = robust_zscore(out["beneish_m_score"]).clip(0.0, 2.0)
            out["z_beneish_risk"] = _risk_excess_scale(out["z_beneish_risk"], trigger=0.40)

        missing_penalty = max(0.0, min(2.0, float(getattr(config, "missing_beneish_penalty", 0.0))))
        if missing_penalty > 0.0 and "beneish_is_missing" in out.columns:
            missing_mask = pd.to_numeric(out["beneish_is_missing"], errors="coerce").fillna(0.0) >= 1.0
            out.loc[missing_mask, "beneish_missing_penalty_applied"] = missing_penalty
            out.loc[missing_mask & out["z_beneish_risk"].isna(), "z_beneish_risk"] = float(
                np.clip(missing_penalty / 2.0, 0.0, 1.0)
            )

        risk_components.append(("z_beneish_risk", 1.0))
    else:
        out["z_beneish_risk"] = np.nan

    if config.use_accrual_volatility and "accrual_volatility" in out.columns and out["accrual_volatility"].notna().any():
        out["z_accrual_volatility_risk"] = robust_zscore(out["accrual_volatility"]).clip(0.0, 2.0)
        out["z_accrual_volatility_risk"] = _risk_excess_scale(out["z_accrual_volatility_risk"], trigger=0.45)
        risk_components.append(("z_accrual_volatility_risk", 0.65))
    else:
        out["z_accrual_volatility_risk"] = np.nan

    if getattr(config, "use_working_capital_stress", False) and "working_capital_stress_penalty" in out.columns:
        penalty = pd.to_numeric(out["working_capital_stress_penalty"], errors="coerce")
        out["z_working_capital_stress_risk"] = _risk_excess_scale((penalty.fillna(0.0) / 0.06).clip(0.0, 2.0), trigger=0.30)
        if penalty.notna().any():
            risk_components.append(("z_working_capital_stress_risk", 0.35))
    else:
        out["z_working_capital_stress_risk"] = np.nan

    if risk_components:
        weighted_total = pd.Series(0.0, index=out.index, dtype=float)
        available_weight = pd.Series(0.0, index=out.index, dtype=float)
        for column, weight in risk_components:
            component = pd.to_numeric(out[column], errors="coerce")
            has_component = component.notna()
            weighted_total = weighted_total + weight * component.fillna(0.0)
            available_weight = available_weight + weight * has_component.astype(float)
        out["forensic_penalty"] = (weighted_total / available_weight.replace(0.0, np.nan)).clip(0.0, 1.25)
    else:
        out["forensic_penalty"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Architectural classes: modular pipeline replacing the monolithic god function
# ---------------------------------------------------------------------------

class UniverseScreener:
    """Handles size, liquidity, biotech flags, and hard structural exclusions."""

    def __init__(self, config: RankerConfig):
        self.config = config

    def apply_sanity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with physically impossible data (not opinion-based gating)."""
        c = self.config
        sy_lo, sy_hi = getattr(c, "shareholder_yield_range", (-0.25, 0.25))
        bb_lo, bb_hi = getattr(c, "buyback_yield_range", (-0.20, 0.20))
        gp_lo, gp_hi = getattr(c, "gross_profitability_range", (0.0, 2.0))
        bm_lo, bm_hi = getattr(c, "adjusted_book_to_market_range", (0.0, 3.0))
        rr_lo, rr_hi = getattr(c, "recency_ratio_range", (0.50, 1.20))
        pd_lo, pd_hi = getattr(c, "price_to_200dma_range", (0.50, 2.50))

        return df[
            (df["market_cap"] > 0)
            & (df["shareholder_yield"].between(sy_lo, sy_hi, inclusive="both") | df["shareholder_yield"].isna())
            & (df["buyback_yield"].between(bb_lo, bb_hi, inclusive="both") | df["buyback_yield"].isna())
            & (df["gross_profitability"].between(gp_lo, gp_hi, inclusive="both") | df["gross_profitability"].isna())
            & (df["adjusted_book_to_market"].between(bm_lo, bm_hi, inclusive="both") | df["adjusted_book_to_market"].isna())
            & (df["recency_ratio"].between(rr_lo, rr_hi, inclusive="both") | df["recency_ratio"].isna())
            & (df["price_to_200dma"].between(pd_lo, pd_hi, inclusive="both") | df["price_to_200dma"].isna())
        ].copy()

    def apply_structural_gates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply only hard structural gates: size, biotech, minimum core factor availability."""
        c = self.config
        df["passes_size"] = df["market_cap"] >= c.min_market_cap

        df["binary_biotech_flag"] = _binary_biotech_flag(
            df, min_revenue=float(getattr(c, "binary_biotech_min_revenue", 1_000_000_000.0)),
        )
        df["passes_biotech_gate"] = True
        if getattr(c, "exclude_binary_biotech", False):
            df["passes_biotech_gate"] = df["binary_biotech_flag"] < 1.0

        factor_matrix = df[CORE_FACTORS].apply(pd.to_numeric, errors="coerce")
        df["factor_non_null_count"] = factor_matrix.notna().sum(axis=1)
        df["factor_positive_count"] = (factor_matrix > 0).sum(axis=1)
        df["passes_factor_gate"] = df["factor_non_null_count"] >= 1

        return df

    def apply_soft_penalties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additive soft penalty fields (not hard exclusions)."""
        c = self.config
        slope = getattr(c, "trend_penalty_slope", 2.5)
        trend_cap = getattr(c, "trend_penalty_cap", 1.0)
        quality_cap = getattr(c, "quality_penalty_cap", 0.5)
        forensic_miss = getattr(c, "forensic_missing_penalty", 0.15)
        impute_pen = getattr(c, "core_impute_penalty", 0.05)
        missing_pen = getattr(c, "core_missing_penalty", 0.15)
        momentum_pen = getattr(c, "momentum_missing_penalty", 0.10)

        beneish_gate_threshold, _ = _resolved_beneish_hard_filter_threshold(c, observed_universe_size=len(df))
        df["beneish_hard_filter_threshold"] = beneish_gate_threshold

        # Trend penalty
        df["penalty_trend"] = 0.0
        if c.require_above_200dma:
            price_to_dma = pd.to_numeric(df["price_to_200dma"], errors="coerce")
            trend_penalty = pd.Series(0.0, index=df.index)
            below_dma = price_to_dma < 1.0
            trend_penalty.loc[below_dma] = ((1.0 - price_to_dma.loc[below_dma]) * slope).clip(0.0, trend_cap)
            df["penalty_trend"] = trend_penalty.fillna(0.0)

        # Quality penalty
        df["penalty_quality"] = 0.0
        piotroski = pd.to_numeric(df["piotroski_score"], errors="coerce")
        min_pio = getattr(c, "min_piotroski_score", 4)
        quality_penalty = ((min_pio - piotroski) / min_pio * quality_cap).clip(0.0, quality_cap)
        quality_penalty = quality_penalty.where(piotroski.notna(), 0.0)
        df["penalty_quality"] = quality_penalty.fillna(0.0)

        # Forensic uncertainty penalty
        df["penalty_forensic_uncertainty"] = 0.0
        if c.use_beneish:
            beneish_score = pd.to_numeric(df["beneish_m_score"], errors="coerce")
            beneish_penalty = ((beneish_score - beneish_gate_threshold) / 2.0).clip(0.0, 1.0)
            beneish_penalty = beneish_penalty.where(beneish_score.notna(), forensic_miss)
            df["penalty_forensic_uncertainty"] = beneish_penalty.fillna(forensic_miss)

        # Core missing penalty
        df["penalty_core_missing"] = 0.0
        imputed_count = pd.to_numeric(df.get("core_factor_imputed_count", 0.0), errors="coerce").fillna(0.0)
        non_null_after = df[CORE_FACTORS].notna().sum(axis=1)
        df["penalty_core_missing"] = (impute_pen * imputed_count + missing_pen * (3 - non_null_after).clip(lower=0)).fillna(0.0)

        # Momentum coverage penalty
        df["penalty_momentum_coverage"] = 0.0
        if getattr(c, "use_price_momentum", False) and getattr(c, "require_real_momentum_coverage", False):
            coverage = pd.to_numeric(df["price_momentum_has_coverage"], errors="coerce").fillna(0.0)
            df["penalty_momentum_coverage"] = (1.0 - coverage.clip(0.0, 1.0)) * momentum_pen

        return df

    def impute_core_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute a single missing core factor per stock with sector median."""
        c = self.config
        df["core_factor_imputed_count"] = 0
        df["core_factor_imputation_flag"] = False
        for f in CORE_FACTORS:
            df[f"imputed_{f}"] = False

        min_coverage = max(8, getattr(c, "min_group_size", 8))
        if "sector" not in df.columns:
            return df

        # Vectorized sector median computation
        sector_medians: dict[str, dict] = {}
        for factor in CORE_FACTORS:
            sector_medians[factor] = (
                df.groupby("sector", dropna=False)[factor]
                .transform(lambda g: g.median() if g.notna().sum() >= min_coverage else np.nan)
            )

        exactly_one_missing = df["factor_non_null_count"] == 2
        for factor in CORE_FACTORS:
            missing_this = df[factor].isna() & exactly_one_missing
            has_median = sector_medians[factor].notna()
            fill_mask = missing_this & has_median
            if not fill_mask.any():
                continue
            df.loc[fill_mask, factor] = sector_medians[factor].loc[fill_mask]
            df.loc[fill_mask, "core_factor_imputed_count"] = 1
            df.loc[fill_mask, "core_factor_imputation_flag"] = True
            df.loc[fill_mask, f"imputed_{factor}"] = True

        return df

    def select_eligible(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return subset passing structural hard gates."""
        return df[
            df["passes_size"] & df["passes_biotech_gate"] & df["passes_factor_gate"]
        ].copy()


class SignalProcessor:
    """Calculates raw signals and cross-sectional robust z-scores.

    NaN-preserving: missing data stays NaN during signal construction.
    fillna(0.0) is only used at the final aggregation step.
    """

    def __init__(self, config: RankerConfig):
        self.config = config
        self._clip = getattr(config, "zscore_clip", 2.0)

    def build_earnings_momentum(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Collapse PEAD + SUE into single earnings_momentum_signal."""
        out = eligible
        out["earnings_momentum_signal"] = np.nan
        out["earnings_momentum_coverage"] = 0.0

        has_pead = "pead_signal" in out.columns and out["pead_signal"].notna()
        has_sue = "sue_signal" in out.columns and out["sue_signal"].notna()
        pead_cov = pd.to_numeric(out.get("pead_has_setup_coverage", 0.0), errors="coerce").fillna(0.0)
        sue_cov = pd.to_numeric(out.get("sue_has_coverage", 0.0), errors="coerce").fillna(0.0)

        if getattr(self.config, "use_pead", False) and has_pead.any():
            if has_sue.any():
                total_cov = pead_cov + sue_cov
                total_cov = total_cov.replace(0.0, np.nan)
                pw = (pead_cov / total_cov).fillna(0.6)
                sw = (sue_cov / total_cov).fillna(0.4)
                # NaN-safe: only fill where the signal itself exists
                pead_val = _series_or_nan(out, "pead_signal")
                sue_val = _series_or_nan(out, "sue_signal")
                out["earnings_momentum_signal"] = pw * pead_val.fillna(0.0) + sw * sue_val.fillna(0.0)
                out["earnings_momentum_coverage"] = ((pead_cov + sue_cov) / 2.0).clip(0.0, 1.0)
            else:
                out["earnings_momentum_signal"] = out["pead_signal"]
                out["earnings_momentum_coverage"] = pead_cov
        return out

    def gate_revision_jerk_persistence(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Zero out revision_jerk_signal when same-sign persistence fails.

        Rows without velocity data are not gated (no persistence info available).
        """
        min_days = getattr(self.config, "revision_jerk_persistence_min_days", 14)
        eligible["revision_jerk_persistence_days"] = 0.0
        eligible["revision_jerk_persistence_pass"] = False

        has_velocity_data = pd.Series(False, index=eligible.index)
        if "revision_jerk_recent_velocity" in eligible.columns and "revision_jerk_prior_velocity" in eligible.columns:
            recent_vel = pd.to_numeric(eligible["revision_jerk_recent_velocity"], errors="coerce")
            prior_vel = pd.to_numeric(eligible["revision_jerk_prior_velocity"], errors="coerce")
            has_velocity_data = recent_vel.notna() & prior_vel.notna()
            same_sign = ((recent_vel > 0) & (prior_vel > 0)) | ((recent_vel < 0) & (prior_vel < 0))
            eligible["revision_jerk_persistence_days"] = np.where(
                same_sign & has_velocity_data, float(min_days), 0.0,
            )
            eligible["revision_jerk_persistence_pass"] = eligible["revision_jerk_persistence_days"] >= float(min_days)

        if "revision_jerk_signal" in eligible.columns:
            eligible["revision_jerk_signal_raw"] = eligible["revision_jerk_signal"].copy()
            # Only gate rows that have velocity data but fail persistence
            persistence_fail = has_velocity_data & ~eligible["revision_jerk_persistence_pass"].fillna(False)
            eligible.loc[persistence_fail, "revision_jerk_signal"] = 0.0
            eligible.loc[persistence_fail, "revision_jerk_has_coverage"] = 0.0

        return eligible

    def add_zscores_and_clip(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional z-scores and clip to ±zscore_clip."""
        eligible = add_factor_zscores(eligible, self.config)
        clip = self._clip
        z_cols = [c for c in eligible.columns if c.startswith("z_")]
        for col in z_cols:
            eligible[col] = eligible[col].clip(-clip, clip)
        return eligible


class RiskOverlay:
    """Calculates forensic and risk penalties."""

    def __init__(self, config: RankerConfig):
        self.config = config

    def compute(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Delegate to existing _build_forensic_penalty."""
        return _build_forensic_penalty(eligible, self.config)


class AlphaAggregator:
    """Applies dynamic correlation-parity weights and computes final composite score.

    Philosophy: "Penalize redundancy, reward uniqueness."
    If Growth and Momentum are highly correlated today, their weights shrink.
    """

    def __init__(self, config: RankerConfig):
        self.config = config
        self._use_dynamic = getattr(config, "use_dynamic_weights", True)
        self._min_universe = getattr(config, "dynamic_weight_min_universe", 30)

    @staticmethod
    def get_dynamic_correlation_weights(df: pd.DataFrame, active_factors: list[str]) -> pd.Series:
        """Compute inverse-correlation parity weights across active z-score factors."""
        factor_data = df[active_factors].dropna()

        if factor_data.empty or len(factor_data) < 30:
            return pd.Series(1.0 / len(active_factors), index=active_factors)

        corr_matrix = factor_data.corr(method="spearman")
        abs_corr = corr_matrix.abs()
        corr_sums = abs_corr.sum(axis=1)

        raw_weights = 1.0 / corr_sums
        final_weights = raw_weights / raw_weights.sum()
        return final_weights

    def resolve_optional_weights(
        self,
        eligible: pd.DataFrame,
        configured_weights: dict[str, float],
        z_col_map: dict[str, str],
    ) -> dict[str, float]:
        """Optionally modulate configured optional weights with correlation parity.

        When use_dynamic_weights is True, the configured weights are blended 50/50
        with correlation-parity weights so that highly-correlated factors shrink
        while preserving the user's intent as a prior.
        """
        if not self._use_dynamic or len(eligible) < self._min_universe:
            return configured_weights

        active_z_cols = [
            z_col_map[name]
            for name, w in configured_weights.items()
            if w > 0.0 and z_col_map.get(name) in eligible.columns and eligible[z_col_map[name]].notna().any()
        ]
        if len(active_z_cols) < 2:
            return configured_weights

        corr_weights = self.get_dynamic_correlation_weights(eligible, active_z_cols)

        # Map back to signal names
        z_to_name = {v: k for k, v in z_col_map.items() if v in active_z_cols}
        blended: dict[str, float] = {}
        for name, cfg_w in configured_weights.items():
            z_col = z_col_map.get(name)
            if z_col and z_col in corr_weights.index:
                corr_w = float(corr_weights[z_col])
                # Blend: 50% config prior, 50% data-driven
                total_cfg = sum(w for w in configured_weights.values() if w > 0.0) or 1.0
                cfg_share = cfg_w / total_cfg
                blended[name] = max(0.0, 0.5 * cfg_share + 0.5 * corr_w) * total_cfg
            else:
                blended[name] = cfg_w
        return blended


def build_ranked_frame(raw_df: pd.DataFrame, config: RankerConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    df["alpha_factor_spec"] = str(getattr(config, "alpha_factor_spec", "legacy") or "legacy")

    beneish_gate_threshold, large_universe_forensic_mode = _resolved_beneish_hard_filter_threshold(
        config,
        observed_universe_size=len(df),
    )

    # Instantiate pipeline classes
    screener = UniverseScreener(config)
    signal_proc = SignalProcessor(config)
    risk_overlay = RiskOverlay(config)
    aggregator = AlphaAggregator(config)

    numeric_cols = [
        "market_cap",
        "shareholder_yield",
        "safe_dividend_yield",
        "gross_profitability",
        "reported_gross_profitability",
        "intangible_adjusted_gross_profitability",
        "adjusted_book_to_market",
        "reported_book_to_market",
        "intangible_adjusted_book_to_market",
        "rd_expense_ratio",
        "intangible_adjustment_eligible",
        "intangible_adjustment_applied",
        "dividend_yield",
        "buyback_yield",
        "payout_ratio",
        "shares_outstanding",
        "short_interest_shares",
        "short_interest_shares_prev",
        "short_interest_ratio",
        "short_interest_pct_float",
        "short_interest_pct_outstanding",
        "short_interest_change",
        "percent_institutions",
        "institution_holder_count_latest",
        "institution_holder_count_prev",
        "institutional_breadth_delta",
        "institutional_ownership_from_holders",
        "institutional_ownership_delta",
        "institutional_top5_concentration_delta",
        "share_drift_1q",
        "share_drift_4q",
        "share_drift_persistence",
        "receivables_days",
        "receivables_days_prev",
        "receivables_days_delta",
        "inventory_days",
        "inventory_days_prev",
        "inventory_days_delta",
        "payables_days",
        "payables_days_prev",
        "payables_days_delta",
        "cash_conversion_cycle_days",
        "cash_conversion_cycle_days_prev",
        "cash_conversion_cycle_days_delta",
        "cash_conversion_cycle_convexity",
        "price_proxy",
        "52_week_high",
        "200_day_ma",
        "recency_ratio",
        "distance_from_high",
        "price_to_200dma",
        "dividend_safety_pass",
        "earnings_surprise_pct",
        "pead_signal",
        "pead_signal_v2",
        "pead_surprise_component",
        "pead_decay_component",
        "pead_breadth_component",
        "pead_revision_component",
        "pead_analyst_count",
        "pead_filter_pass",
        "pead_has_setup_coverage",
        "sue_signal",
        "sue_has_coverage",
        "sue_surprise_raw",
        "sue_surprise_pct",
        "sue_std_error",
        "revision_impulse_signal",
        "revision_impulse_has_coverage",
        "revision_impulse_analyst_count",
        "revision_jerk_signal",
        "revision_jerk_has_coverage",
        "revision_jerk_recent_velocity",
        "revision_jerk_prior_velocity",
        "revision_jerk_component",
        "revision_impulse_drift_7d",
        "revision_impulse_drift_30d",
        "revision_impulse_breadth",
        "revision_impulse_growth_component",
        "revision_impulse_coverage_component",
        "revision_impulse_disagreement",
        "revision_impulse_disagreement_penalty",
        "revision_short_divergence_component",
        "revision_breadth_7d",
        "revision_breadth_30d",
        "revision_breadth_acceleration",
        "float_absorption_signal",
        "squeeze_convexity_signal",
        "estimate_term_structure_signal",
        "estimate_term_structure_has_coverage",
        "estimate_term_structure_record_count",
        "estimate_term_structure_persistence",
        "estimate_term_structure_improvement",
        "estimate_term_structure_disagreement_trend",
        "estimate_term_structure_coverage_component",
        "revenue_growth_yoy",
        "revenue_growth_yoy_prev",
        "revenue_acceleration",
        "revenue_growth_has_coverage",
        "reported_return_on_assets",
        "intangible_adjusted_return_on_assets",
        "return_on_assets",
        "reported_return_on_invested_capital",
        "intangible_adjusted_return_on_invested_capital",
        "return_on_invested_capital",
        "peer_margin_trend_input",
        "peer_reinvestment_efficiency_input",
        "peer_estimate_drift_input",
        "peer_dilution_discipline_input",
        "peer_ownership_breadth_input",
        "residual_value_signal",
        "residual_value_has_coverage",
        "peer_relative_anomaly_signal",
        "peer_relative_anomaly_has_coverage",
        "peer_relative_margin_component",
        "peer_relative_reinvestment_component",
        "peer_relative_estimate_component",
        "peer_relative_dilution_component",
        "peer_relative_ownership_component",
        "compounder_persistence_signal",
        "compounder_persistence_has_coverage",
        "compounder_persistence_measure_count",
        "compounder_persistence_level_component",
        "compounder_persistence_stability_component",
        "compounder_persistence_trend_component",
        "compounder_persistence_periodicity",
        "price_momentum_1m",
        "price_momentum_6m",
        "price_momentum_6m_ex_1m",
        "price_momentum_has_coverage",
        "price_momentum_effective_signal",
        "price_momentum_signal_coverage",
        "price_momentum_proxy_used",
        "sentiment_latest",
        "sentiment_speed",
        "sentiment_acceleration",
        "sentiment_count_days",
        "sentiment_article_count_recent",
        "sentiment_article_count_total",
        "sentiment_latest_count",
        "sentiment_filter_pass",
        "news_event_signal",
        "news_event_effective_signal",
        "news_event_breadth",
        "news_article_count_recent",
        "news_baseline_signal",
        "news_baseline_article_count",
        "news_article_volume_spike",
        "news_positive_article_share",
        "news_negative_article_share",
        "news_unique_title_ratio",
        "news_novelty_score",
        "news_saturation_score",
        "news_shock_signal",
        "news_shock_has_coverage",
        "news_novelty_multiplier",
        "news_saturation_multiplier",
        "news_peer_spillover_signal",
        "news_peer_coverage",
        "news_confirmation_signal",
        "news_confirmation_coverage",
        "news_macro_multiplier",
        "piotroski_score",
        "beneish_m_score",
        "beneish_is_missing",
        "beneish_is_pathological_clipped",
        "beneish_hard_filter_pass",
        "beneish_hard_filter_threshold",
        "beneish_missing_penalty_applied",
        "accrual_ratio",
        "accrual_volatility",
        "accrual_measure_count",
        "accrual_is_quarterly",
        "working_capital_stress_penalty",
        "working_capital_stress_has_coverage",
        "working_capital_receivables_stress",
        "working_capital_inventory_stress",
        "working_capital_payables_stress",
        "working_capital_cfo_stress",
        "working_capital_cycle_stress",
        "investment_restraint_signal",
        "investment_restraint_has_coverage",
        "investment_restraint_measure_count",
        "investment_restraint_asset_growth",
        "investment_restraint_noa_growth",
        "investment_restraint_acquisition_intensity",
        "investment_restraint_capex_intensity",
        "investment_restraint_share_issuance",
        "investment_restraint_debt_funded_expansion",
        "accrual_quality_signal",
        "accrual_quality_has_coverage",
        "accrual_quality_measure_count",
        "accrual_quality_level_component",
        "accrual_quality_stability_component",
        "accrual_quality_trend_component",
        "accrual_quality_periodicity",
        "accrual_quality_cash_conversion",
        "accrual_quality_margin_gap",
        "accrual_quality_working_capital_stretch",
        "accrual_quality_cycle_convexity",
        "quality_acceleration_signal",
        "quality_acceleration_has_coverage",
        "quality_acceleration_measure_count",
        "quality_acceleration_margin_delta",
        "quality_acceleration_return_delta",
        "quality_acceleration_turnover_delta",
        "quality_acceleration_cfo_margin_delta",
        "quality_acceleration_working_capital_delta",
        "quality_acceleration_periodicity",
        "capital_allocation_quality_signal",
        "capital_allocation_quality_has_coverage",
        "capital_allocation_buyback_component",
        "capital_allocation_funding_component",
        "capital_allocation_debt_component",
        "capital_allocation_payout_component",
        "capital_allocation_reinvestment_component",
        "capital_allocation_financing_dependency_component",
        "financing_dependency_stress",
        "financing_dependency_burn_component",
        "financing_dependency_dilution_component",
        "financing_dependency_debt_component",
        "financing_dependency_revision_component",
        "recovery_margin_inflection",
        "recovery_leverage_improvement",
        "recovery_accrual_improvement",
        "recovery_transition_signal",
        "recovery_transition_has_coverage",
        "insider_conviction_signal",
        "insider_conviction_has_coverage",
        "insider_conviction_buy_cluster",
        "insider_conviction_sell_pressure",
        "insider_conviction_trade_count",
        "insider_conviction_buy_person_count",
        "insider_conviction_sell_person_count",
        "insider_ownership_confirmation_component",
        "insider_short_crowding_penalty",
        "news_theme_drift_signal",
        "news_theme_drift_has_coverage",
        "news_theme_drift_recent_intensity",
        "news_theme_drift_baseline_intensity",
        "news_theme_drift_recent_article_count",
        "news_theme_drift_baseline_article_count",
        "technical_momentum_signal",
        "technical_momentum_has_coverage",
        "technical_rsi_14",
        "technical_macd_histogram",
        "technical_bbands_pct_b",
        "technical_adx_14",
        "technical_stoch_k",
        "technical_stoch_d",
        "forensic_penalty",
        "total_revenue",
        "full_time_employees",
        "revenue_per_employee",
        "gross_profit_per_employee",
        "earnings_signal_confidence",
        "revision_signal_confidence",
        "revision_jerk_signal_confidence",
        "estimate_term_structure_signal_confidence",
        "growth_signal_confidence",
        "momentum_signal_confidence",
        "news_signal_confidence",
        "news_shock_signal_confidence",
        "capital_allocation_signal_confidence",
        "residual_value_signal_confidence",
        "compounder_persistence_signal_confidence",
        "peer_relative_anomaly_signal_confidence",
        "recovery_transition_signal_confidence",
        "investment_restraint_signal_confidence",
        "accrual_quality_signal_confidence",
        "quality_acceleration_signal_confidence",
        "insider_conviction_signal_confidence",
        "news_theme_drift_signal_confidence",
        "technical_momentum_signal_confidence",
        "effective_core_share",
        "effective_optional_share",
        "contrib_shareholder_yield",
        "contrib_gross_profitability",
        "contrib_adjusted_book_to_market",
        "contrib_residual_value",
        "contrib_compounder_persistence",
        "contrib_peer_relative_anomaly",
        "contrib_earnings",
        "contrib_revision_impulse",
        "contrib_revision_jerk",
        "contrib_estimate_term_structure",
        "contrib_growth",
        "contrib_momentum",
        "contrib_news_event",
        "contrib_news_shock",
        "contrib_capital_allocation",
        "contrib_recovery_transition",
        "contrib_investment_restraint",
        "contrib_accrual_quality",
        "contrib_quality_acceleration",
        "contrib_insider_conviction",
        "contrib_news_theme_drift",
        "contrib_technical_momentum",
        "contrib_employee_efficiency",
        "contrib_forensic",
        "estimate_term_structure_overlap_penalty",
    ]

    df = _concat_missing_columns(df, {col: np.nan for col in numeric_cols})

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- UniverseScreener: sanity filters, structural gates, imputation, penalties ---
    df = screener.apply_sanity_filters(df)

    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    df = screener.apply_structural_gates(df)
    df = screener.impute_core_factors(df)
    df = screener.apply_soft_penalties(df)

    eligible = screener.select_eligible(df)

    if eligible.empty:
        return df, eligible, build_diagnostics(
            df,
            pd.DataFrame(),
            beneish_gate_threshold=beneish_gate_threshold,
            large_universe_forensic_mode=large_universe_forensic_mode,
            universe_size=int(getattr(config, "universe_size", len(raw_df)) or len(raw_df)),
        )

    if getattr(config, "use_residual_valuation", False):
        eligible = _build_residual_value_signal(eligible, config)
    else:
        eligible["residual_value_signal"] = np.nan
        eligible["residual_value_has_coverage"] = 0.0
        eligible["residual_value_peer_level"] = pd.Series(pd.NA, index=eligible.index, dtype="object")

    if getattr(config, "use_peer_relative_anomalies", False):
        eligible = _build_peer_relative_anomaly_signal(eligible, config)
    else:
        eligible["peer_relative_anomaly_signal"] = np.nan
        eligible["peer_relative_anomaly_has_coverage"] = 0.0
        eligible["peer_relative_anomaly_peer_level"] = pd.Series(pd.NA, index=eligible.index, dtype="object")
        eligible["peer_relative_margin_component"] = np.nan
        eligible["peer_relative_reinvestment_component"] = np.nan
        eligible["peer_relative_estimate_component"] = np.nan
        eligible["peer_relative_dilution_component"] = np.nan
        eligible["peer_relative_ownership_component"] = np.nan

    if getattr(config, "use_news_events", False):
        eligible = _build_news_experiment_metrics(eligible, config)
    else:
        eligible["news_event_effective_signal"] = np.nan

    # --- SignalProcessor: build composite signals, apply persistence gating, z-score ---
    eligible = signal_proc.build_earnings_momentum(eligible)
    eligible = signal_proc.gate_revision_jerk_persistence(eligible)
    eligible = signal_proc.add_zscores_and_clip(eligible)

    # --- RiskOverlay: forensic and risk penalties ---
    eligible = risk_overlay.compute(eligible)

    core_weights = get_macro_factor_weights(
        macro_state=getattr(config, "macro_state", "neutral"),
        fallback_regime=config.regime,
    )

    if getattr(config, "use_life_cycle", False) or getattr(config, "use_recovery_transition", False):
        eligible = _build_life_cycle_context(eligible, config, core_weights)
        shareholder_component_weight = eligible["life_cycle_core_weight_shareholder_yield"].fillna(
            core_weights["shareholder_yield"]
        )
        gross_profitability_component_weight = eligible["life_cycle_core_weight_gross_profitability"].fillna(
            core_weights["gross_profitability"]
        )
        adjusted_book_component_weight = eligible["life_cycle_core_weight_adjusted_book_to_market"].fillna(
            core_weights["adjusted_book_to_market"]
        )
        pead_multiplier = eligible["life_cycle_pead_multiplier"].fillna(1.0)
        growth_multiplier = eligible["life_cycle_growth_multiplier"].fillna(1.0)
        momentum_multiplier = eligible["life_cycle_momentum_multiplier"].fillna(1.0)
        revision_multiplier = eligible["life_cycle_revision_impulse_multiplier"].fillna(1.0)
        forensic_multiplier = eligible["life_cycle_forensic_multiplier"].fillna(1.0)
    else:
        shareholder_component_weight = pd.Series(
            core_weights["shareholder_yield"], index=eligible.index, dtype=float
        )
        gross_profitability_component_weight = pd.Series(
            core_weights["gross_profitability"], index=eligible.index, dtype=float
        )
        adjusted_book_component_weight = pd.Series(
            core_weights["adjusted_book_to_market"], index=eligible.index, dtype=float
        )
        pead_multiplier = pd.Series(1.0, index=eligible.index, dtype=float)
        growth_multiplier = pd.Series(1.0, index=eligible.index, dtype=float)
        momentum_multiplier = pd.Series(1.0, index=eligible.index, dtype=float)
        revision_multiplier = pd.Series(1.0, index=eligible.index, dtype=float)
        forensic_multiplier = pd.Series(1.0, index=eligible.index, dtype=float)

    (
        shareholder_component_weight,
        gross_profitability_component_weight,
        adjusted_book_component_weight,
        residual_component_weight,
        compounder_component_weight,
    ) = _split_core_component_weights(
        eligible,
        config,
        shareholder_component_weight.copy(),
        gross_profitability_component_weight.copy(),
        adjusted_book_component_weight.copy(),
    )

    # Earnings momentum weight using the new formalized earnings_momentum_signal
    earnings_momentum_weight = (
        float(getattr(config, "earnings_momentum_base_weight", 0.08))
        if config.use_pead and eligible["earnings_momentum_signal"].notna().any()
        else 0.0
    )
    revision_impulse_weight = (
        float(config.revision_impulse_weight)
        if getattr(config, "use_revision_impulse", False) and eligible["revision_impulse_signal"].notna().any()
        else 0.0
    )
    revision_jerk_weight = (
        float(getattr(config, "revision_jerk_weight", 0.0))
        if getattr(config, "use_revision_jerk", False) and eligible["revision_jerk_signal"].notna().any()
        else 0.0
    )
    estimate_term_structure_weight = (
        float(getattr(config, "estimate_term_structure_weight", 0.0))
        if getattr(config, "use_estimate_term_structure", False)
        and eligible["estimate_term_structure_signal"].notna().any()
        else 0.0
    )
    growth_weight = (
        float(config.growth_weight)
        if getattr(config, "use_growth_acceleration", False)
        and (
            eligible["revenue_growth_yoy"].notna().any()
            or eligible["revenue_acceleration"].notna().any()
        )
        else 0.0
    )
    quality_acceleration_weight = (
        float(getattr(config, "quality_acceleration_weight", 0.0))
        if getattr(config, "use_quality_acceleration", False)
        and eligible["quality_acceleration_signal"].notna().any()
        else 0.0
    )
    momentum_weight = (
        float(config.momentum_weight)
        if getattr(config, "use_price_momentum", False) and eligible["price_momentum_effective_signal"].notna().any()
        else 0.0
    )
    # News demotion: cap combined news contribution to small fixed optional-share ceiling
    max_combined_news_share = float(getattr(config, "max_combined_news_share", 0.04))

    raw_news_weight = float(getattr(config, "news_event_weight", 0.0))
    raw_news_shock_weight = float(getattr(config, "news_shock_weight", 0.0))

    news_weight = (
        raw_news_weight
        if getattr(config, "use_news_events", False) and eligible["news_event_effective_signal"].notna().any()
        else 0.0
    )
    news_shock_weight = (
        raw_news_shock_weight
        if getattr(config, "use_news_shock", False) and eligible["news_shock_signal"].notna().any()
        else 0.0
    )

    # Apply news cap: scale down if combined would exceed ceiling
    total_news_raw = news_weight + news_shock_weight
    if total_news_raw > max_combined_news_share:
        news_scale = max_combined_news_share / total_news_raw
        news_weight = news_weight * news_scale
        news_shock_weight = news_shock_weight * news_scale
    capital_allocation_weight = (
        float(getattr(config, "capital_allocation_weight", 0.0))
        if getattr(config, "use_capital_allocation_quality", False)
        and eligible["capital_allocation_quality_signal"].notna().any()
        else 0.0
    )
    investment_restraint_weight = (
        float(getattr(config, "investment_restraint_weight", 0.0))
        if getattr(config, "use_investment_restraint", False)
        and eligible["investment_restraint_signal"].notna().any()
        else 0.0
    )
    accrual_quality_weight = (
        float(getattr(config, "accrual_quality_weight", 0.0))
        if getattr(config, "use_accrual_quality", False)
        and eligible["accrual_quality_signal"].notna().any()
        else 0.0
    )
    insider_conviction_weight = (
        float(getattr(config, "insider_conviction_weight", 0.0))
        if getattr(config, "use_insider_conviction", False)
        and eligible["insider_conviction_signal"].notna().any()
        else 0.0
    )
    news_theme_drift_weight = (
        float(getattr(config, "news_theme_drift_weight", 0.0))
        if getattr(config, "use_news_theme_drift", False)
        and eligible["news_theme_drift_signal"].notna().any()
        else 0.0
    )
    technical_momentum_weight = (
        float(getattr(config, "technical_momentum_weight", 0.0))
        if getattr(config, "use_technical_momentum", False)
        and "technical_momentum_signal" in eligible.columns
        and eligible["technical_momentum_signal"].notna().any()
        else 0.0
    )
    peer_relative_anomaly_weight = (
        float(getattr(config, "peer_relative_anomaly_weight", 0.0))
        if getattr(config, "use_peer_relative_anomalies", False)
        and eligible["peer_relative_anomaly_signal"].notna().any()
        else 0.0
    )
    employee_weight = 0.0
    if getattr(config, "use_employee_efficiency", False):
        has_employee_signal = (
            eligible["revenue_per_employee"].notna().any()
            or eligible["gross_profit_per_employee"].notna().any()
        )
        if has_employee_signal:
            employee_weight = float(config.employee_efficiency_weight)

    recovery_transition_weight = 0.0

    employee_component = pd.Series(0.0, index=eligible.index, dtype=float)
    employee_efficiency_signal_confidence = pd.Series(0.0, index=eligible.index, dtype=float)
    if employee_weight > 0.0:
        employee_component_frame = pd.DataFrame(
            {
                "revenue_per_employee": eligible["z_revenue_per_employee"],
                "gross_profit_per_employee": eligible["z_gross_profit_per_employee"],
            }
        )
        employee_efficiency_signal_confidence = employee_component_frame.notna().mean(axis=1).fillna(0.0).clip(0.0, 1.0)
        employee_component = employee_component_frame.mean(axis=1).fillna(0.0)
    # Earnings momentum component using the new formalized signal
    earnings_component = (
        eligible["z_earnings_momentum_signal"].fillna(0.0)
        if "z_earnings_momentum_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    revision_impulse_component = (
        eligible["z_revision_impulse_signal"].fillna(0.0)
        if "z_revision_impulse_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    revision_jerk_component = (
        eligible["z_revision_jerk_signal"].fillna(0.0)
        if "z_revision_jerk_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    estimate_term_structure_component = (
        eligible["z_estimate_term_structure_signal"].fillna(0.0)
        if "z_estimate_term_structure_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    eligible["estimate_term_structure_overlap_penalty"] = 0.0
    estimate_overlap_predictors: dict[str, pd.Series] = {}
    if revision_impulse_weight > 0.0:
        estimate_overlap_predictors["revision_impulse"] = revision_impulse_component
    if revision_jerk_weight > 0.0:
        estimate_overlap_predictors["revision_jerk"] = revision_jerk_component
    if estimate_overlap_predictors and estimate_term_structure_weight > 0.0:
        estimate_term_structure_component, estimate_overlap_penalty = _residualize_against_predictors(
            estimate_term_structure_component,
            estimate_overlap_predictors,
        )
        eligible["estimate_term_structure_overlap_penalty"] = float(estimate_overlap_penalty)
    _yoy_share = float(getattr(config, "growth_component_yoy_share", 0.45))
    _accel_share = float(getattr(config, "growth_component_accel_share", 0.55))
    growth_component = (
        _yoy_share
        * (
            eligible["z_revenue_growth_yoy"].fillna(0.0)
            if "z_revenue_growth_yoy" in eligible.columns
            else pd.Series(0.0, index=eligible.index, dtype=float)
        )
        + _accel_share
        * (
            eligible["z_revenue_acceleration"].fillna(0.0)
            if "z_revenue_acceleration" in eligible.columns
            else pd.Series(0.0, index=eligible.index, dtype=float)
        )
    )
    quality_acceleration_component = (
        eligible["z_quality_acceleration_signal"].fillna(0.0)
        if "z_quality_acceleration_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    momentum_component = (
        eligible["z_price_momentum_effective_signal"].fillna(0.0)
        if "z_price_momentum_effective_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    news_component = (
        eligible["z_news_event_effective_signal"].fillna(0.0)
        if "z_news_event_effective_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    news_shock_component = (
        eligible["z_news_shock_signal"].fillna(0.0)
        if "z_news_shock_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    residual_value_component = (
        eligible["z_residual_value_signal"].fillna(0.0)
        if "z_residual_value_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    compounder_persistence_component = (
        eligible["z_compounder_persistence_signal"].fillna(0.0)
        if "z_compounder_persistence_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    peer_relative_anomaly_component = (
        eligible["z_peer_relative_anomaly_signal"].fillna(0.0)
        if "z_peer_relative_anomaly_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    capital_allocation_component = (
        eligible["z_capital_allocation_quality_signal"].fillna(0.0)
        if "z_capital_allocation_quality_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    investment_restraint_component = (
        eligible["z_investment_restraint_signal"].fillna(0.0)
        if "z_investment_restraint_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    accrual_quality_component = (
        eligible["z_accrual_quality_signal"].fillna(0.0)
        if "z_accrual_quality_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    insider_conviction_component = (
        eligible["z_insider_conviction_signal"].fillna(0.0)
        if "z_insider_conviction_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    news_theme_drift_component = (
        eligible["z_news_theme_drift_signal"].fillna(0.0)
        if "z_news_theme_drift_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    technical_momentum_component = (
        eligible["z_technical_momentum_signal"].fillna(0.0)
        if "z_technical_momentum_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )

    eligible = _build_recovery_transition_metrics(
        eligible,
        config,
        revision_component=revision_impulse_component,
        momentum_component=momentum_component,
        estimate_component=estimate_term_structure_component,
    )
    recovery_transition_component = (
        eligible["recovery_transition_signal"].fillna(0.0)
        if "recovery_transition_signal" in eligible.columns
        else pd.Series(0.0, index=eligible.index, dtype=float)
    )
    recovery_transition_weight = (
        float(getattr(config, "recovery_transition_weight", 0.0))
        if getattr(config, "use_recovery_transition", False)
        and eligible["recovery_transition_signal"].notna().any()
        else 0.0
    )

    earnings_coverage = (
        0.5 * _series_or_zero(eligible, "pead_has_setup_coverage")
        + 0.5 * _series_or_zero(eligible, "sue_has_coverage")
    ).clip(0.0, 1.0)
    revision_coverage = _series_or_zero(eligible, "revision_impulse_coverage_component").clip(0.0, 1.0)
    revision_disagreement_penalty = _series_or_zero(eligible, "revision_impulse_disagreement_penalty").clip(0.0, 1.0)
    growth_coverage = (
        0.6 * _series_or_zero(eligible, "revenue_growth_has_coverage")
        + 0.4 * _series_or_nan(eligible, "revenue_acceleration").notna().astype(float)
    ).clip(0.0, 1.0)
    momentum_history_coverage = _series_or_zero(eligible, "price_momentum_has_coverage").clip(0.0, 1.0)
    momentum_proxy_used = _series_or_zero(eligible, "price_momentum_proxy_used").clip(0.0, 1.0)

    eligible["earnings_signal_confidence"] = _coverage_scale(earnings_coverage)
    eligible["revision_signal_confidence"] = (
        _coverage_scale(revision_coverage, floor=0.60) * (0.80 + 0.20 * (1.0 - revision_disagreement_penalty))
    ).clip(0.0, 1.0)
    eligible["revision_jerk_signal_confidence"] = (
        _series_or_zero(eligible, "revision_jerk_has_coverage").clip(0.0, 1.0)
        * (0.50 + 0.50 * revision_coverage)
        * (0.70 + 0.30 * (1.0 - revision_disagreement_penalty))
    ).clip(0.0, 1.0)
    eligible["estimate_term_structure_signal_confidence"] = _coverage_scale(
        _series_or_zero(eligible, "estimate_term_structure_coverage_component").clip(0.0, 1.0),
        floor=0.60,
    ).clip(0.0, 1.0)
    eligible["estimate_term_structure_signal_confidence"] = (
        eligible["estimate_term_structure_signal_confidence"]
        * (1.0 - 0.35 * _series_or_zero(eligible, "estimate_term_structure_overlap_penalty").clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    eligible["growth_signal_confidence"] = _coverage_scale(growth_coverage)
    eligible["momentum_signal_confidence"] = (
        momentum_history_coverage.where(momentum_history_coverage >= 1.0, 0.0)
        + 0.70 * momentum_proxy_used.where(momentum_history_coverage < 1.0, 0.0)
    ).clip(0.0, 1.0)
    eligible.loc[eligible["momentum_signal_confidence"] > 0.0, "momentum_signal_confidence"] = (
        0.55 + 0.45 * eligible.loc[eligible["momentum_signal_confidence"] > 0.0, "momentum_signal_confidence"]
    )
    news_article_count = _series_or_zero(eligible, "news_article_count_recent")
    news_breadth = _series_or_zero(eligible, "news_event_breadth")
    news_count_scale = (news_article_count / max(1.0, float(getattr(config, "min_news_articles", 1)))).clip(0.0, 1.0)
    news_breadth_scale = (news_breadth / 3.0).clip(0.0, 1.0)
    news_peer_coverage = _series_or_zero(eligible, "news_peer_coverage").clip(0.0, 1.0)
    news_confirmation_coverage = _series_or_zero(eligible, "news_confirmation_coverage").clip(0.0, 1.0)
    news_peer_signal = _series_or_zero(eligible, "news_peer_spillover_signal")
    eligible["news_own_confidence"] = 0.0
    eligible["news_signal_confidence"] = 0.0
    news_mask = news_article_count > 0.0
    eligible.loc[news_mask, "news_own_confidence"] = (
        0.45 + 0.35 * news_count_scale.loc[news_mask] + 0.20 * news_breadth_scale.loc[news_mask]
    )
    eligible["news_signal_confidence"] = eligible["news_own_confidence"]
    if getattr(config, "use_news_peer_spillover", False):
        peer_confidence = (0.20 + 0.30 * news_peer_coverage + 0.15 * news_peer_signal.abs().clip(0.0, 1.0)).clip(0.0, 0.65)
        eligible["news_signal_confidence"] = np.maximum(eligible["news_signal_confidence"], peer_confidence)
    if getattr(config, "use_news_confirmation", False):
        eligible["news_signal_confidence"] = (
            eligible["news_signal_confidence"] + 0.10 * news_confirmation_coverage
        ).clip(0.0, 1.0)
    news_shock_volume_scale = (_series_or_zero(eligible, "news_article_volume_spike").clip(0.0, 3.0) / 3.0).clip(0.0, 1.0)
    eligible["news_shock_signal_confidence"] = (
        _series_or_zero(eligible, "news_shock_has_coverage").clip(0.0, 1.0)
        * (0.50 + 0.30 * news_shock_volume_scale + 0.20 * _series_or_zero(eligible, "news_novelty_score").clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    eligible["news_has_coverage"] = (
        news_article_count >= float(getattr(config, "min_news_articles", 1))
    ) | (
        getattr(config, "use_news_peer_spillover", False)
        & (news_peer_coverage >= 0.50)
        & (news_peer_signal.abs() > 0.0)
    )
    residual_peer_scale = _peer_level_scale(
        eligible.get("residual_value_peer_level", pd.Series(pd.NA, index=eligible.index, dtype="object")),
        {"industry": 1.0, "sector": 0.85, "global": 0.70},
        default=0.0,
    )
    eligible["residual_value_signal_confidence"] = (
        _series_or_zero(eligible, "residual_value_has_coverage").clip(0.0, 1.0) * residual_peer_scale
    ).clip(0.0, 1.0)

    compounder_periodicity = _series_or_zero(eligible, "compounder_persistence_periodicity").clip(0.0, 1.0)
    compounder_full_count = 5.0 + 3.0 * compounder_periodicity
    compounder_depth = (
        _series_or_zero(eligible, "compounder_persistence_measure_count") / compounder_full_count.replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    compounder_depth_scale = (0.50 + 0.50 * compounder_depth) * (0.75 + 0.25 * compounder_periodicity)
    eligible["compounder_persistence_signal_confidence"] = (
        _series_or_zero(eligible, "compounder_persistence_has_coverage").clip(0.0, 1.0) * compounder_depth_scale
    ).clip(0.0, 1.0)
    if getattr(config, "use_recovery_transition", False) and "life_cycle_stage" in eligible.columns:
        recovery_stage_mask = eligible["life_cycle_stage"].fillna("").astype(str).str.lower() == "recovery"
        eligible.loc[recovery_stage_mask, "compounder_persistence_signal_confidence"] = (
            eligible.loc[recovery_stage_mask, "compounder_persistence_signal_confidence"] * 0.65
        )

    eligible["capital_allocation_signal_confidence"] = _coverage_scale(
        _series_or_zero(eligible, "capital_allocation_quality_has_coverage"),
        floor=0.70,
    )
    peer_level_scale = _peer_level_scale(
        eligible.get("peer_relative_anomaly_peer_level", pd.Series(pd.NA, index=eligible.index, dtype="object")),
        {"industry": 1.0, "sector": 0.85, "global": 0.70},
        default=0.0,
    )
    eligible["peer_relative_anomaly_signal_confidence"] = _coverage_scale(
        _series_or_zero(eligible, "peer_relative_anomaly_has_coverage"),
        floor=0.65,
    ) * peer_level_scale
    eligible["recovery_transition_signal_confidence"] = _coverage_scale(
        _series_or_zero(eligible, "recovery_transition_has_coverage"),
        floor=0.65,
    )
    investment_depth = _depth_scale(_series_or_zero(eligible, "investment_restraint_measure_count"), 6.0)
    eligible["investment_restraint_signal_confidence"] = (
        _series_or_zero(eligible, "investment_restraint_has_coverage").clip(0.0, 1.0)
        * (0.60 + 0.40 * investment_depth)
    ).clip(0.0, 1.0)
    accrual_periodicity = _series_or_zero(eligible, "accrual_quality_periodicity").clip(0.0, 1.0)
    accrual_full_count = 5.0 + 3.0 * accrual_periodicity
    accrual_depth = (
        _series_or_zero(eligible, "accrual_quality_measure_count") / accrual_full_count.replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    eligible["accrual_quality_signal_confidence"] = (
        _series_or_zero(eligible, "accrual_quality_has_coverage").clip(0.0, 1.0)
        * (0.55 + 0.45 * accrual_depth)
        * (0.75 + 0.25 * accrual_periodicity)
    ).clip(0.0, 1.0)
    quality_periodicity = _series_or_zero(eligible, "quality_acceleration_periodicity").clip(0.0, 1.0)
    quality_full_count = 4.0 + 2.0 * quality_periodicity
    quality_depth = (
        _series_or_zero(eligible, "quality_acceleration_measure_count") / quality_full_count.replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    eligible["quality_acceleration_signal_confidence"] = (
        _series_or_zero(eligible, "quality_acceleration_has_coverage").clip(0.0, 1.0)
        * (0.55 + 0.45 * quality_depth)
        * (0.75 + 0.25 * quality_periodicity)
    ).clip(0.0, 1.0)
    insider_trade_depth = _depth_scale(_series_or_zero(eligible, "insider_conviction_trade_count"), 4.0)
    insider_participant_depth = _depth_scale(
        _series_or_zero(eligible, "insider_conviction_buy_person_count")
        + _series_or_zero(eligible, "insider_conviction_sell_person_count"),
        3.0,
    )
    eligible["insider_conviction_signal_confidence"] = (
        _series_or_zero(eligible, "insider_conviction_has_coverage").clip(0.0, 1.0)
        * (0.35 + 0.35 * insider_trade_depth + 0.20 * insider_participant_depth)
        * (0.70 + 0.30 * _series_or_zero(eligible, "insider_conviction_signal").abs().clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    news_theme_recent_count = _series_or_zero(eligible, "news_theme_drift_recent_article_count")
    news_theme_baseline_count = _series_or_zero(eligible, "news_theme_drift_baseline_article_count")
    news_theme_overlap_count = np.minimum(news_theme_recent_count, news_theme_baseline_count)
    news_theme_max_count = pd.concat([news_theme_recent_count, news_theme_baseline_count], axis=1).max(axis=1).clip(lower=1.0)
    news_theme_depth = _depth_scale(news_theme_overlap_count, 3.0)
    news_theme_balance = (news_theme_overlap_count / news_theme_max_count).clip(0.0, 1.0)
    eligible["news_theme_drift_signal_confidence"] = (
        _series_or_zero(eligible, "news_theme_drift_has_coverage").clip(0.0, 1.0)
        * (0.30 + 0.45 * news_theme_depth + 0.25 * news_theme_balance)
    ).clip(0.0, 1.0)

    eligible["technical_momentum_signal_confidence"] = (
        _series_or_zero(eligible, "technical_momentum_has_coverage").clip(0.0, 1.0)
        * (0.60 + 0.40 * _series_or_zero(eligible, "technical_momentum_signal").abs().clip(0.0, 1.0))
    ).clip(0.0, 1.0)

    core_floor = float(np.clip(float(getattr(config, "core_weight_floor", 0.60) or 0.60), 0.0, 1.0))
    max_optional_share = max(0.0, 1.0 - core_floor)
    configured_optional_weights = {
        "peer_relative_anomaly": peer_relative_anomaly_weight,
        "earnings": earnings_momentum_weight,
        "revision_impulse": revision_impulse_weight,
        "revision_jerk": revision_jerk_weight,
        "estimate_term_structure": estimate_term_structure_weight,
        "growth": growth_weight,
        "quality_acceleration": quality_acceleration_weight,
        "momentum": momentum_weight,
        "news_event": news_weight,
        "news_shock": news_shock_weight,
        "capital_allocation": capital_allocation_weight,
        "recovery_transition": recovery_transition_weight,
        "investment_restraint": investment_restraint_weight,
        "accrual_quality": accrual_quality_weight,
        "insider_conviction": insider_conviction_weight,
        "news_theme_drift": news_theme_drift_weight,
        "technical_momentum": technical_momentum_weight,
        "employee_efficiency": employee_weight,
    }
    # --- AlphaAggregator: optionally modulate with correlation parity ---
    _optional_z_col_map = {
        "peer_relative_anomaly": "z_peer_relative_anomaly_signal",
        "earnings": "z_earnings_momentum_signal",
        "revision_impulse": "z_revision_impulse_signal",
        "revision_jerk": "z_revision_jerk_signal",
        "estimate_term_structure": "z_estimate_term_structure_signal",
        "growth": "z_revenue_acceleration",
        "quality_acceleration": "z_quality_acceleration_signal",
        "momentum": "z_price_momentum_effective_signal",
        "news_event": "z_news_event_effective_signal",
        "news_shock": "z_news_shock_signal",
        "capital_allocation": "z_capital_allocation_quality_signal",
        "recovery_transition": "recovery_transition_signal",
        "investment_restraint": "z_investment_restraint_signal",
        "accrual_quality": "z_accrual_quality_signal",
        "insider_conviction": "z_insider_conviction_signal",
        "news_theme_drift": "z_news_theme_drift_signal",
        "technical_momentum": "z_technical_momentum_signal",
        "employee_efficiency": "z_revenue_per_employee",
    }
    configured_optional_weights = aggregator.resolve_optional_weights(
        eligible, configured_optional_weights, _optional_z_col_map,
    )

    configured_optional_total = float(sum(configured_optional_weights.values()))
    optional_budget_scale = (
        min(1.0, max_optional_share / configured_optional_total)
        if configured_optional_total > 0.0
        else 1.0
    )
    scaled_optional_weights = {
        name: float(weight) * optional_budget_scale
        for name, weight in configured_optional_weights.items()
    }

    optional_activation = {
        "peer_relative_anomaly": scaled_optional_weights["peer_relative_anomaly"]
        * eligible["peer_relative_anomaly_signal_confidence"].fillna(0.0),
        "earnings": scaled_optional_weights["earnings"] * eligible["earnings_signal_confidence"].fillna(0.0),
        "revision_impulse": scaled_optional_weights["revision_impulse"]
        * eligible["revision_signal_confidence"].fillna(0.0),
        "revision_jerk": scaled_optional_weights["revision_jerk"]
        * eligible["revision_jerk_signal_confidence"].fillna(0.0),
        "estimate_term_structure": scaled_optional_weights["estimate_term_structure"]
        * eligible["estimate_term_structure_signal_confidence"].fillna(0.0),
        "growth": scaled_optional_weights["growth"] * eligible["growth_signal_confidence"].fillna(0.0),
        "quality_acceleration": scaled_optional_weights["quality_acceleration"]
        * eligible["quality_acceleration_signal_confidence"].fillna(0.0),
        "momentum": scaled_optional_weights["momentum"] * eligible["momentum_signal_confidence"].fillna(0.0),
        "news_event": scaled_optional_weights["news_event"] * eligible["news_signal_confidence"].fillna(0.0),
        "news_shock": scaled_optional_weights["news_shock"] * eligible["news_shock_signal_confidence"].fillna(0.0),
        "capital_allocation": scaled_optional_weights["capital_allocation"]
        * eligible["capital_allocation_signal_confidence"].fillna(0.0),
        "recovery_transition": scaled_optional_weights["recovery_transition"]
        * eligible["recovery_transition_signal_confidence"].fillna(0.0),
        "investment_restraint": scaled_optional_weights["investment_restraint"]
        * eligible["investment_restraint_signal_confidence"].fillna(0.0),
        "accrual_quality": scaled_optional_weights["accrual_quality"]
        * eligible["accrual_quality_signal_confidence"].fillna(0.0),
        "insider_conviction": scaled_optional_weights["insider_conviction"]
        * eligible["insider_conviction_signal_confidence"].fillna(0.0),
        "news_theme_drift": scaled_optional_weights["news_theme_drift"]
        * eligible["news_theme_drift_signal_confidence"].fillna(0.0),
        "technical_momentum": scaled_optional_weights["technical_momentum"]
        * eligible["technical_momentum_signal_confidence"].fillna(0.0),
        "employee_efficiency": scaled_optional_weights["employee_efficiency"]
        * employee_efficiency_signal_confidence.fillna(0.0),
    }
    optional_activation_frame = pd.DataFrame(optional_activation, index=eligible.index)
    eligible["effective_optional_share"] = optional_activation_frame.sum(axis=1).clip(0.0, max_optional_share)
    eligible["effective_core_share"] = (1.0 - eligible["effective_optional_share"]).clip(core_floor, 1.0)

    core_effective_weights = _normalize_weight_block(
        {
            "shareholder_yield": shareholder_component_weight,
            "gross_profitability": gross_profitability_component_weight,
            "adjusted_book_to_market": adjusted_book_component_weight,
            "residual_value": residual_component_weight * eligible["residual_value_signal_confidence"].fillna(0.0),
            "compounder_persistence": compounder_component_weight
            * eligible["compounder_persistence_signal_confidence"].fillna(0.0),
        },
        eligible["effective_core_share"],
    )
    optional_effective_weights = _normalize_weight_block(
        {
            "peer_relative_anomaly": optional_activation["peer_relative_anomaly"],
            "earnings": optional_activation["earnings"] * pead_multiplier,
            "revision_impulse": optional_activation["revision_impulse"] * revision_multiplier,
            "revision_jerk": optional_activation["revision_jerk"],
            "estimate_term_structure": optional_activation["estimate_term_structure"],
            "growth": optional_activation["growth"] * growth_multiplier,
            "quality_acceleration": optional_activation["quality_acceleration"],
            "momentum": optional_activation["momentum"] * momentum_multiplier,
            "news_event": optional_activation["news_event"],
            "news_shock": optional_activation["news_shock"],
            "capital_allocation": optional_activation["capital_allocation"],
            "recovery_transition": optional_activation["recovery_transition"],
            "investment_restraint": optional_activation["investment_restraint"],
            "accrual_quality": optional_activation["accrual_quality"],
            "insider_conviction": optional_activation["insider_conviction"],
            "news_theme_drift": optional_activation["news_theme_drift"],
            "technical_momentum": optional_activation["technical_momentum"],
            "employee_efficiency": optional_activation["employee_efficiency"],
        },
        eligible["effective_optional_share"],
    )

    eligible["contrib_shareholder_yield"] = (
        core_effective_weights["shareholder_yield"] * eligible["z_shareholder_yield"].fillna(0.0)
    )
    eligible["contrib_gross_profitability"] = (
        core_effective_weights["gross_profitability"] * eligible["z_gross_profitability"].fillna(0.0)
    )
    eligible["contrib_adjusted_book_to_market"] = (
        core_effective_weights["adjusted_book_to_market"] * eligible["z_adjusted_book_to_market"].fillna(0.0)
    )
    eligible["contrib_residual_value"] = core_effective_weights["residual_value"] * residual_value_component
    eligible["contrib_compounder_persistence"] = (
        core_effective_weights["compounder_persistence"] * compounder_persistence_component
    )
    eligible["contrib_peer_relative_anomaly"] = (
        optional_effective_weights["peer_relative_anomaly"] * peer_relative_anomaly_component
    )
    eligible["contrib_earnings"] = optional_effective_weights["earnings"] * earnings_component
    eligible["contrib_growth"] = optional_effective_weights["growth"] * growth_component
    eligible["contrib_quality_acceleration"] = (
        optional_effective_weights["quality_acceleration"] * quality_acceleration_component
    )
    eligible["contrib_momentum"] = optional_effective_weights["momentum"] * momentum_component
    eligible["contrib_news_event"] = optional_effective_weights["news_event"] * news_component
    eligible["contrib_news_shock"] = optional_effective_weights["news_shock"] * news_shock_component
    eligible["contrib_revision_impulse"] = (
        optional_effective_weights["revision_impulse"] * revision_impulse_component
    )
    eligible["contrib_revision_jerk"] = optional_effective_weights["revision_jerk"] * revision_jerk_component
    eligible["contrib_estimate_term_structure"] = (
        optional_effective_weights["estimate_term_structure"] * estimate_term_structure_component
    )
    eligible["contrib_capital_allocation"] = (
        optional_effective_weights["capital_allocation"] * capital_allocation_component
    )
    eligible["contrib_recovery_transition"] = (
        optional_effective_weights["recovery_transition"] * recovery_transition_component
    )
    eligible["contrib_investment_restraint"] = (
        optional_effective_weights["investment_restraint"] * investment_restraint_component
    )
    eligible["contrib_accrual_quality"] = (
        optional_effective_weights["accrual_quality"] * accrual_quality_component
    )
    eligible["contrib_insider_conviction"] = (
        optional_effective_weights["insider_conviction"] * insider_conviction_component
    )
    eligible["contrib_news_theme_drift"] = (
        optional_effective_weights["news_theme_drift"] * news_theme_drift_component
    )
    eligible["contrib_technical_momentum"] = (
        optional_effective_weights["technical_momentum"] * technical_momentum_component
    )
    eligible["contrib_employee_efficiency"] = (
        optional_effective_weights["employee_efficiency"] * employee_component
    )
    eligible["contrib_forensic"] = (
        -float(config.forensic_weight) * forensic_multiplier * eligible["forensic_penalty"].fillna(0.0)
    )

    eligible["composite_score"] = (
        eligible["contrib_shareholder_yield"]
        + eligible["contrib_gross_profitability"]
        + eligible["contrib_adjusted_book_to_market"]
        + eligible["contrib_residual_value"]
        + eligible["contrib_compounder_persistence"]
        + eligible["contrib_peer_relative_anomaly"]
        + eligible["contrib_earnings"]
        + eligible["contrib_growth"]
        + eligible["contrib_quality_acceleration"]
        + eligible["contrib_momentum"]
        + eligible["contrib_news_event"]
        + eligible["contrib_news_shock"]
        + eligible["contrib_revision_impulse"]
        + eligible["contrib_revision_jerk"]
        + eligible["contrib_estimate_term_structure"]
        + eligible["contrib_capital_allocation"]
        + eligible["contrib_recovery_transition"]
        + eligible["contrib_investment_restraint"]
        + eligible["contrib_accrual_quality"]
        + eligible["contrib_insider_conviction"]
        + eligible["contrib_news_theme_drift"]
        + eligible["contrib_technical_momentum"]
        + eligible["contrib_employee_efficiency"]
        + eligible["contrib_forensic"]
    )

    # PEAD filter: soft penalty instead of hard exclusion
    eligible["penalty_pead_filter"] = 0.0
    eligible["pead_has_setup_coverage"] = pd.to_numeric(eligible["pead_has_setup_coverage"], errors="coerce").fillna(0.0)
    if config.use_pead and not eligible.empty:
        fails_pead = (
            eligible["pead_has_setup_coverage"] >= 1.0
        ) & (pd.to_numeric(eligible["pead_filter_pass"], errors="coerce").fillna(1.0) < 1.0)
        eligible.loc[fails_pead, "penalty_pead_filter"] = 0.25

    # Sentiment filter: soft penalty instead of hard exclusion
    eligible["penalty_sentiment_filter"] = 0.0
    if config.use_sentiment and not eligible.empty:
        eligible["sentiment_has_coverage"] = (
            pd.to_numeric(eligible["sentiment_count_days"], errors="coerce").fillna(0.0) >= config.min_sentiment_days
        ) & (
            pd.to_numeric(eligible["sentiment_article_count_recent"], errors="coerce").fillna(0.0)
            >= config.min_sentiment_articles_recent
        )

        fails_sentiment = (
            eligible["sentiment_has_coverage"]
            & (pd.to_numeric(eligible["sentiment_filter_pass"], errors="coerce").fillna(1.0) < 1.0)
        )
        eligible.loc[fails_sentiment, "penalty_sentiment_filter"] = 0.20
    else:
        eligible["sentiment_has_coverage"] = False
    if not getattr(config, "use_news_events", False) and not getattr(config, "use_news_shock", False):
        eligible["news_has_coverage"] = False

    # Apply soft penalties to composite score
    eligible["composite_score"] = eligible["composite_score"] - eligible["penalty_pead_filter"] - eligible["penalty_sentiment_filter"]

    df = _concat_missing_columns(
        df,
        {col: _empty_series_for_dtype(eligible[col].dtype, df.index) for col in eligible.columns},
    )

    for col in eligible.columns:
        if col in df.columns:
            source_dtype = eligible[col].dtype
            target_dtype = df[col].dtype
            if not pd.api.types.is_numeric_dtype(source_dtype) and pd.api.types.is_numeric_dtype(target_dtype):
                df[col] = df[col].astype("object")
        df.loc[eligible.index, col] = eligible[col]

    eligible = eligible.sort_values(
        by=["composite_score", "shareholder_yield", "gross_profitability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    eligible["rank"] = np.arange(1, len(eligible) + 1)
    eligible["regime"] = config.regime
    eligible["neutralize_by"] = config.neutralize_by
    eligible["macro_state"] = getattr(config, "macro_state", "neutral")

    return df, eligible, build_diagnostics(
        df,
        eligible,
        beneish_gate_threshold=beneish_gate_threshold,
        large_universe_forensic_mode=large_universe_forensic_mode,
        universe_size=int(getattr(config, "universe_size", len(raw_df)) or len(raw_df)),
    )


def _sector_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty or "sector" not in frame.columns:
        return float("nan"), float("nan")

    weights = frame["sector"].fillna("Unknown").value_counts(normalize=True)
    if weights.empty:
        return float("nan"), float("nan")

    return float(weights.iloc[0]), float((weights.pow(2)).sum())


def build_neutralization_comparison(sector_ranked: pd.DataFrame, none_ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    mode_frames = {"sector": sector_ranked, "none": none_ranked}
    top_n = max(1, int(top_n))

    for mode, frame in mode_frames.items():
        top = frame.head(top_n).copy()
        rows.append({"comparison": "single_mode", "mode": mode, "metric": "final_ranked_count", "value": float(len(frame))})
        rows.append({"comparison": "single_mode", "mode": mode, "metric": f"top_{top_n}_count", "value": float(len(top))})

        for col in [
            "composite_score",
            "shareholder_yield",
            "gross_profitability",
            "adjusted_book_to_market",
            "pead_signal",
            "sue_signal",
            "revision_impulse_signal",
            "revenue_growth_yoy",
            "revenue_acceleration",
            "price_momentum_6m_ex_1m",
            "price_momentum_effective_signal",
            "forensic_penalty",
            "beneish_m_score",
            "accrual_volatility",
        ]:
            if col in top.columns:
                rows.append(
                    {
                        "comparison": "single_mode",
                        "mode": mode,
                        "metric": f"top_{top_n}_median_{col}",
                        "value": float(pd.to_numeric(top[col], errors="coerce").median()),
                    }
                )

        top_sector_share, sector_hhi = _sector_concentration(top)
        rows.append({"comparison": "single_mode", "mode": mode, "metric": f"top_{top_n}_sector_top_share", "value": top_sector_share})
        rows.append({"comparison": "single_mode", "mode": mode, "metric": f"top_{top_n}_sector_hhi", "value": sector_hhi})

    top_sector_symbols = set(sector_ranked.head(top_n)["symbol"]) if "symbol" in sector_ranked.columns else set()
    top_none_symbols = set(none_ranked.head(top_n)["symbol"]) if "symbol" in none_ranked.columns else set()
    overlap = top_sector_symbols & top_none_symbols
    union = top_sector_symbols | top_none_symbols
    overlap_ratio = float(len(overlap) / len(union)) if union else float("nan")
    rows.append(
        {
            "comparison": "sector_vs_none",
            "mode": "pair",
            "metric": f"top_{top_n}_overlap_count",
            "value": float(len(overlap)),
        }
    )
    rows.append(
        {
            "comparison": "sector_vs_none",
            "mode": "pair",
            "metric": f"top_{top_n}_overlap_ratio",
            "value": overlap_ratio,
        }
    )

    if {"symbol", "rank"}.issubset(sector_ranked.columns) and {"symbol", "rank"}.issubset(none_ranked.columns):
        merged = (
            sector_ranked[["symbol", "rank"]]
            .rename(columns={"rank": "rank_sector"})
            .merge(
                none_ranked[["symbol", "rank"]].rename(columns={"rank": "rank_none"}),
                on="symbol",
                how="inner",
            )
        )
        if len(merged) >= 2:
            rank_corr = merged["rank_sector"].corr(merged["rank_none"], method="spearman")
            rows.append(
                {
                    "comparison": "sector_vs_none",
                    "mode": "pair",
                    "metric": "rank_spearman_corr",
                    "value": float(rank_corr) if pd.notna(rank_corr) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def build_revision_impulse_weight_comparison(weight_ranked: dict[float, pd.DataFrame], top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    top_n = max(1, int(top_n))
    ordered_weights = sorted(float(weight) for weight in weight_ranked.keys())
    if not ordered_weights:
        return pd.DataFrame(rows)

    for weight in ordered_weights:
        frame = weight_ranked.get(weight, pd.DataFrame()).copy()
        top = frame.head(top_n).copy()
        rows.append(
            {
                "comparison": "single_weight",
                "anchor_weight": float("nan"),
                "weight": float(weight),
                "metric": "final_ranked_count",
                "value": float(len(frame)),
            }
        )
        rows.append(
            {
                "comparison": "single_weight",
                "anchor_weight": float("nan"),
                "weight": float(weight),
                "metric": f"top_{top_n}_count",
                "value": float(len(top)),
            }
        )

        for col in [
            "composite_score",
            "shareholder_yield",
            "gross_profitability",
            "adjusted_book_to_market",
            "pead_signal",
            "sue_signal",
            "revision_impulse_signal",
            "revenue_growth_yoy",
            "revenue_acceleration",
            "price_momentum_6m_ex_1m",
            "price_momentum_effective_signal",
            "forensic_penalty",
            "beneish_m_score",
            "accrual_volatility",
        ]:
            if col in top.columns:
                rows.append(
                    {
                        "comparison": "single_weight",
                        "anchor_weight": float("nan"),
                        "weight": float(weight),
                        "metric": f"top_{top_n}_median_{col}",
                        "value": float(pd.to_numeric(top[col], errors="coerce").median()),
                    }
                )

        top_sector_share, sector_hhi = _sector_concentration(top)
        rows.append(
            {
                "comparison": "single_weight",
                "anchor_weight": float("nan"),
                "weight": float(weight),
                "metric": f"top_{top_n}_sector_top_share",
                "value": top_sector_share,
            }
        )
        rows.append(
            {
                "comparison": "single_weight",
                "anchor_weight": float("nan"),
                "weight": float(weight),
                "metric": f"top_{top_n}_sector_hhi",
                "value": sector_hhi,
            }
        )

    anchor_weight = ordered_weights[0]
    anchor_frame = weight_ranked.get(anchor_weight, pd.DataFrame()).copy()
    anchor_top_symbols = set(anchor_frame.head(top_n)["symbol"]) if "symbol" in anchor_frame.columns else set()

    for weight in ordered_weights[1:]:
        frame = weight_ranked.get(weight, pd.DataFrame()).copy()
        top_symbols = set(frame.head(top_n)["symbol"]) if "symbol" in frame.columns else set()
        overlap = anchor_top_symbols & top_symbols
        union = anchor_top_symbols | top_symbols
        overlap_ratio = float(len(overlap) / len(union)) if union else float("nan")

        rows.append(
            {
                "comparison": "anchor_vs_weight",
                "anchor_weight": float(anchor_weight),
                "weight": float(weight),
                "metric": f"top_{top_n}_overlap_count",
                "value": float(len(overlap)),
            }
        )
        rows.append(
            {
                "comparison": "anchor_vs_weight",
                "anchor_weight": float(anchor_weight),
                "weight": float(weight),
                "metric": f"top_{top_n}_overlap_ratio",
                "value": overlap_ratio,
            }
        )

        if {"symbol", "rank"}.issubset(anchor_frame.columns) and {"symbol", "rank"}.issubset(frame.columns):
            merged = (
                anchor_frame[["symbol", "rank"]]
                .rename(columns={"rank": "rank_anchor"})
                .merge(
                    frame[["symbol", "rank"]].rename(columns={"rank": "rank_weight"}),
                    on="symbol",
                    how="inner",
                )
            )
            if len(merged) >= 2:
                rank_corr = merged["rank_anchor"].corr(merged["rank_weight"], method="spearman")
                rows.append(
                    {
                        "comparison": "anchor_vs_weight",
                        "anchor_weight": float(anchor_weight),
                        "weight": float(weight),
                        "metric": "rank_spearman_corr",
                        "value": float(rank_corr) if pd.notna(rank_corr) else float("nan"),
                    }
                )

    return pd.DataFrame(rows)


def print_error_summary(errors: pd.DataFrame, max_rows: int = 10) -> None:
    import sys

    if errors.empty:
        return

    print(f"Skipped {len(errors)} symbols with errors.", file=sys.stderr)

    if "error_stage" in errors.columns:
        print("Error breakdown by stage:", file=sys.stderr)
        for stage, count in errors["error_stage"].fillna("unknown").value_counts().items():
            print(f"  - {stage}: {count}", file=sys.stderr)

    print("Sample errors:", file=sys.stderr)
    sample_cols = [c for c in ["symbol", "error_stage", "error"] if c in errors.columns]
    for _, row in errors[sample_cols].head(max_rows).iterrows():
        print(
            f"  - {row.get('symbol', '?')} [{row.get('error_stage', 'unknown')}]: {str(row.get('error', ''))[:250]}",
            file=sys.stderr,
        )
