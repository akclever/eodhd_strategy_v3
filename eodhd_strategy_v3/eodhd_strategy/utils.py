from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ALLOWED_US_EXCHANGES = {'US', 'NYSE', 'NASDAQ', 'AMEX', 'ARCA'}


def utc_today_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz='UTC').normalize()


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(',', '')
        if cleaned in {'', 'None', 'null', 'nan', 'NA', 'N/A', '-'}:
            return None
        try:
            result = float(cleaned)
            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except ValueError:
            return None
    return None


def pick_first(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            val = to_float(mapping.get(key))
            if val is not None:
                return val
    return None


def normalize_records(section: Any) -> List[Dict[str, Any]]:
    if not section:
        return []
    if isinstance(section, list):
        records = [x for x in section if isinstance(x, dict)]
    elif isinstance(section, dict):
        records = [x for x in section.values() if isinstance(x, dict)]
    else:
        return []

    def sort_key(rec: Dict[str, Any]) -> str:
        return str(rec.get('dateFormatted') or rec.get('date') or rec.get('filing_date') or '')

    records.sort(key=sort_key, reverse=True)
    return records


def robust_zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)
    clipped = s.clip(lower=lo, upper=hi)
    mean = clipped.mean()
    std = clipped.std(ddof=0)
    if std is None or std == 0 or np.isnan(std):
        z = clipped * 0.0
    else:
        z = (clipped - mean) / std
    if not higher_is_better:
        z = -z
    return z
