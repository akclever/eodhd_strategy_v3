from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .client import EODHDClient
from .data_provider import DataProvider

ClientLike = EODHDClient | DataProvider

HIGH_IMPACT_KEYWORDS = ['cpi', 'pmi', 'payroll', 'fomc', 'fed', 'inflation', 'rate decision', 'interest rate', 'nfp']


@dataclass
class RebalanceWindowDecision:
    target_date: str
    recommended_date: str
    blocked_dates: List[str]
    blocking_events: List[str]


def _coerce_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if 'events' in payload and isinstance(payload['events'], list):
            return [x for x in payload['events'] if isinstance(x, dict)]
        return [x for x in payload.values() if isinstance(x, dict)]
    return []


def _event_label(event: Dict[str, Any]) -> str:
    parts = [str(event.get('event') or ''), str(event.get('type') or ''), str(event.get('title') or ''), str(event.get('description') or '')]
    return ' '.join(x for x in parts if x).strip() or 'Unknown macro event'


def is_high_impact_event(event: Dict[str, Any]) -> bool:
    return any(k in _event_label(event).lower() for k in HIGH_IMPACT_KEYWORDS)


def recommend_rebalance_date(client: ClientLike, start_date: str, country: str = 'US', defer_if_within_days: int = 1, max_delay_days: int = 5) -> RebalanceWindowDecision:
    start = pd.Timestamp(start_date).normalize()
    end = start + pd.Timedelta(days=max_delay_days + defer_if_within_days)
    rows = _coerce_rows(client.get_economic_events(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), country=country))
    blocked_dates = []
    blocking_events = []
    for event in rows:
        date_val = pd.to_datetime(event.get('date') or event.get('datetime') or event.get('time'), errors='coerce')
        if pd.isna(date_val):
            continue
        if is_high_impact_event(event):
            d = date_val.normalize().strftime('%Y-%m-%d')
            blocked_dates.append(d)
            blocking_events.append(f'{d} :: {_event_label(event)}')
    blocked_dates = sorted(set(blocked_dates))
    current = start
    while current.strftime('%Y-%m-%d') in blocked_dates or any((pd.Timestamp(d) - current).days in range(0, defer_if_within_days + 1) for d in blocked_dates):
        current += pd.Timedelta(days=1)
        if (current - start).days > max_delay_days:
            break
    return RebalanceWindowDecision(target_date=start.strftime('%Y-%m-%d'), recommended_date=current.strftime('%Y-%m-%d'), blocked_dates=blocked_dates, blocking_events=blocking_events)
