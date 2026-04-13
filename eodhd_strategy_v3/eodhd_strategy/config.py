from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class RankerConfig:
    api_token: str
    cache_dir: Path
    refresh: bool
    workers: int
    min_market_cap: float
    dividend_source: str
    regime: str
    use_pead: bool
    pead_lookback_days: int
    pead_half_life_days: int
    min_pead_analysts: int
    use_revision_impulse: bool
    min_revision_analysts: int
    revision_impulse_weight: float
    use_estimate_term_structure: bool
    estimate_term_structure_weight: float
    use_growth_acceleration: bool
    growth_weight: float
    alpha_factor_spec: Literal["legacy", "v2"]
    use_residual_valuation: bool
    use_compounder_persistence: bool
    use_intangible_adjustments: bool
    use_price_momentum: bool
    require_real_momentum_coverage: bool
    momentum_weight: float
    use_life_cycle: bool
    life_cycle_tilt_strength: float
    use_sentiment: bool
    sentiment_lookback_days: int
    min_sentiment_accel: float
    min_sentiment_articles_recent: int
    use_news_events: bool
    news_lookback_days: int
    min_news_articles: int
    news_event_weight: float
    use_news_peer_spillover: bool
    news_peer_spillover_weight: float
    use_news_novelty_saturation: bool
    use_news_confirmation: bool
    news_confirmation_weight: float
    use_news_macro_weighting: bool
    use_beneish: bool
    use_accrual_volatility: bool
    use_working_capital_stress: bool
    forensic_weight: float
    missing_beneish_penalty: float
    use_capital_allocation_quality: bool
    capital_allocation_weight: float
    use_recovery_transition: bool
    recovery_transition_weight: float
    use_insider_conviction: bool
    insider_conviction_weight: float
    use_news_theme_drift: bool
    news_theme_drift_weight: float
    use_peer_relative_anomalies: bool
    peer_relative_anomaly_weight: float
    exclude_binary_biotech: bool
    binary_biotech_min_revenue: float
    dividend_payout_cap: float
    max_distance_from_high: float
    require_above_200dma: bool
    neutralize_by: str
    min_group_size: int
    overlay_top_n: int
    output: Path
    min_sentiment_days: int
    min_piotroski_score: int
    pead_max_abs_surprise_pct: float
    pead_max_age_days: int
    macro_state: str
    universe_size: int
    use_employee_efficiency: bool
    employee_efficiency_weight: float
    analysis_from_primary_ticker: bool = False
    exclude_special_situations: bool = False
    price_momentum_source_mode: Literal["auto", "history_only", "trend_proxy"] = "auto"
    use_investment_restraint: bool = False
    investment_restraint_weight: float = 0.04
    use_accrual_quality: bool = False
    accrual_quality_weight: float = 0.05
    use_quality_acceleration: bool = False
    quality_acceleration_weight: float = 0.05
    core_weight_floor: float = 0.60
    use_revision_jerk: bool = False
    revision_jerk_weight: float = 0.04
    use_news_shock: bool = False
    news_shock_weight: float = 0.04


@dataclass
class PortfolioConfig:
    top_n_positions: int
    max_position_weight: float
    sector_cap: float
    buy_rank_buffer: int
    hold_rank_buffer: int
    defer_if_macro_event_within_days: int
    rebalance_country: str
    previous_holdings_path: Optional[Path]
    output: Path

    force_sell_on_dividend_break: bool = True
    force_sell_on_sentiment_break: bool = True
    force_sell_on_trend_break: bool = True
    force_sell_below_200dma: float = 0.95
