"""
NLP Staging Area for Earnings Transcripts

This module handles the preparation of earnings call transcripts for NLP analysis.
It fetches transcripts from FMP, extracts relevant sections (Management Discussion
and Q&A), and provides a mock FinBERT scoring function for sentiment analysis.

The cleaned text blocks can be passed to a local Deep Learning model (FinBERT)
to compute sentiment scores that map to news_shock_signal and news_theme_drift_signal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSection:
    """Represents a section of an earnings call transcript."""
    section_type: str  # "management_discussion" or "qa"
    text: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    speaker: Optional[str] = None


def _extract_transcript_sections(transcript_text: str) -> List[TranscriptSection]:
    """
    Extract Management Discussion and Q&A sections from transcript text.
    
    This function attempts to identify and separate different sections of an
    earnings call transcript, removing boilerplate safe-harbor statements.
    
    Args:
        transcript_text: Raw transcript text
        
    Returns:
        List of TranscriptSection objects
    """
    sections = []
    
    # Common patterns for section headers
    mgmt_patterns = [
        r"(?i)(management\s+discussion|prepared\s+remarks|presentation)",
        r"(?i)(ceo\s+remarks|executive\s+remarks)",
        r"(?i)(operating\s+results|financial\s+overview)",
    ]
    
    qa_patterns = [
        r"(?i)(question\s+and\s+answer|q\s*&\s*a|questions?\s+and\s+answers?)",
        r"(?i)(analyst\s+questions|call\s+for\s+questions)",
    ]
    
    # Remove safe-harbor boilerplate
    safe_harbor_patterns = [
        r"(?i)(safe\s+harbor|forward-looking\s+statements?).*?(?=management|question)",
        r"(?i)(this\s+call\s+contains\s+forward-looking\s+statements)",
        r"(?i)(we\s+may\s+make\s+forward-looking\s+statements)",
    ]
    
    cleaned_text = transcript_text
    for pattern in safe_harbor_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL)
    
    # Try to identify sections
    current_section = None
    current_text = []
    
    lines = cleaned_text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for management section start
        is_mgmt = any(re.search(pattern, line) for pattern in mgmt_patterns)
        is_qa = any(re.search(pattern, line) for pattern in qa_patterns)
        
        if is_mgmt and current_section != "management_discussion":
            # Save previous section if exists
            if current_section and current_text:
                sections.append(TranscriptSection(
                    section_type=current_section,
                    text="\n".join(current_text)
                ))
            current_section = "management_discussion"
            current_text = []
        elif is_qa and current_section != "qa":
            # Save previous section if exists
            if current_section and current_text:
                sections.append(TranscriptSection(
                    section_type=current_section,
                    text="\n".join(current_text)
                ))
            current_section = "qa"
            current_text = []
        
        current_text.append(line)
    
    # Don't forget the last section
    if current_section and current_text:
        sections.append(TranscriptSection(
            section_type=current_section,
            text="\n".join(current_text)
        ))
    
    # If no sections identified, treat entire text as management discussion
    if not sections and cleaned_text:
        sections.append(TranscriptSection(
            section_type="management_discussion",
            text=cleaned_text
        ))
    
    return sections


def _mock_finbert_score(text: str) -> float:
    """
    Mock FinBERT sentiment scoring function.
    
    This is a placeholder for actual FinBERT model inference.
    In production, this would load a pre-trained FinBERT model and
    compute sentiment scores from the text.
    
    For now, it uses simple keyword-based sentiment analysis as a proxy.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment score between -1.0 (bearish) and +1.0 (bullish)
    """
    # Simple keyword-based sentiment as placeholder
    positive_keywords = [
        "growth", "increase", "strong", "beat", "exceed", "outperform",
        "positive", "excellent", "record", "improve", "expand", "gain",
        "profit", "margin", "success", "opportunity", "momentum"
    ]
    
    negative_keywords = [
        "decline", "decrease", "weak", "miss", "underperform", "negative",
        "challenge", "concern", "risk", "loss", "cut", "reduce", "pressure",
        "difficult", "uncertain", "slow", "down", "headwind"
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    # Normalize to [-1, 1]
    score = (positive_count - negative_count) / total
    return score


def compute_transcript_sentiment(transcript_sections: List[TranscriptSection]) -> Dict[str, float]:
    """
    Compute sentiment scores from transcript sections.
    
    Args:
        transcript_sections: List of transcript sections
        
    Returns:
        Dictionary with sentiment metrics
    """
    if not transcript_sections:
        return {
            "news_shock_signal": 0.0,
            "news_theme_drift_signal": 0.0,
            "news_shock_has_coverage": 0.0,
            "news_theme_drift_has_coverage": 0.0,
        }
    
    # Compute sentiment for each section
    mgmt_sentiment = 0.0
    qa_sentiment = 0.0
    mgmt_count = 0
    qa_count = 0
    
    for section in transcript_sections:
        score = _mock_finbert_score(section.text)
        
        if section.section_type == "management_discussion":
            mgmt_sentiment += score
            mgmt_count += 1
        elif section.section_type == "qa":
            qa_sentiment += score
            qa_count += 1
    
    # Average sentiment by section type
    avg_mgmt_sentiment = mgmt_sentiment / mgmt_count if mgmt_count > 0 else 0.0
    avg_qa_sentiment = qa_sentiment / qa_count if qa_count > 0 else 0.0
    
    # Combined sentiment (weighted average)
    combined_sentiment = (avg_mgmt_sentiment * 0.6 + avg_qa_sentiment * 0.4)
    
    return {
        "news_shock_signal": combined_sentiment,
        "news_theme_drift_signal": avg_mgmt_sentiment,  # Theme drift based on management discussion
        "news_shock_has_coverage": 1.0 if mgmt_count > 0 or qa_count > 0 else 0.0,
        "news_theme_drift_has_coverage": 1.0 if mgmt_count > 0 else 0.0,
    }


def save_cleaned_transcripts(
    symbol: str,
    transcript_sections: List[TranscriptSection],
    output_dir: Path
) -> List[Path]:
    """
    Save cleaned transcript sections to files.
    
    Args:
        symbol: Stock symbol
        transcript_sections: List of transcript sections
        output_dir: Directory to save files
        
    Returns:
        List of file paths that were saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, section in enumerate(transcript_sections):
        filename = f"{symbol}_{section.section_type}_{timestamp}_{i}.txt"
        filepath = output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Section: {section.section_type}\n")
            if section.speaker:
                f.write(f"Speaker: {section.speaker}\n")
            if section.start_time:
                f.write(f"Time: {section.start_time} - {section.end_time}\n")
            f.write("\n")
            f.write(section.text)
        
        saved_paths.append(filepath)
        logger.info(f"Saved transcript section to {filepath}")
    
    return saved_paths


def fetch_fmp_earnings_transcripts(
    symbol: str,
    lookback_hours: int = 72
) -> List[Dict[str, Any]]:
    """
    Fetch earnings transcripts from FMP (synchronous placeholder).
    
    This is a placeholder for the actual FMP API call. In production,
    this would use the FMP client to fetch transcripts asynchronously.
    
    Args:
        symbol: Stock symbol
        lookback_hours: Hours to look back for recent transcripts
        
    Returns:
        List of transcript data dictionaries
    """
    # Placeholder - in production, this would call FMP API
    logger.info(f"Fetching earnings transcripts for {symbol} (lookback={lookback_hours}h)")
    
    # Return empty list for now
    # In production: return await fmp_client.fetch_earnings_transcripts(symbol, lookback_hours)
    return []


def process_earnings_transcripts_for_symbol(
    symbol: str,
    transcripts: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Process earnings transcripts for a single symbol.
    
    Args:
        symbol: Stock symbol
        transcripts: List of transcript data from FMP
        output_dir: Optional directory to save cleaned transcripts
        
    Returns:
        Dictionary with sentiment metrics
    """
    if not transcripts:
        return {
            "symbol": symbol,
            "news_shock_signal": 0.0,
            "news_theme_drift_signal": 0.0,
            "news_shock_has_coverage": 0.0,
            "news_theme_drift_has_coverage": 0.0,
        }
    
    all_sections = []
    
    for transcript in transcripts:
        transcript_text = transcript.get("content", transcript.get("text", ""))
        if not transcript_text:
            continue
        
        sections = _extract_transcript_sections(transcript_text)
        all_sections.extend(sections)
    
    # Save cleaned transcripts if output dir provided
    if output_dir and all_sections:
        save_cleaned_transcripts(symbol, all_sections, output_dir)
    
    # Compute sentiment
    sentiment = compute_transcript_sentiment(all_sections)
    
    return {
        "symbol": symbol,
        **sentiment
    }


def process_bulk_earnings_transcripts(
    transcripts_data: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Process earnings transcripts for multiple symbols in bulk.
    
    Args:
        transcripts_data: List of transcript data for multiple symbols
        output_dir: Optional directory to save cleaned transcripts
        
    Returns:
        DataFrame with sentiment metrics for each symbol
    """
    if not transcripts_data:
        return pd.DataFrame()
    
    # Group transcripts by symbol
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for transcript in transcripts_data:
        symbol = transcript.get("symbol", "")
        if symbol:
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(transcript)
    
    # Process each symbol
    results = []
    for symbol, symbol_transcripts in by_symbol.items():
        result = process_earnings_transcripts_for_symbol(
            symbol,
            symbol_transcripts,
            output_dir
        )
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Ensure symbol is first column
        cols = ["symbol"] + [col for col in df.columns if col != "symbol"]
        df = df[cols]
    
    logger.info(f"Processed earnings transcripts for {len(df)} symbols")
    return df


def create_nlp_temp_directory(base_dir: Path = Path(".fmp_nlp_cache")) -> Path:
    """
    Create a temporary directory for NLP staging.
    
    Args:
        base_dir: Base directory for NLP cache
        
    Returns:
        Path to the temporary directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = base_dir / f"transcripts_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def map_transcript_sentiment_to_ranker_columns(
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map transcript sentiment DataFrame to ranker-expected columns.
    
    Args:
        sentiment_df: DataFrame from process_bulk_earnings_transcripts
        
    Returns:
        DataFrame with ranker column names
    """
    if sentiment_df.empty:
        return pd.DataFrame()
    
    df = sentiment_df.copy()
    
    # Ensure required columns exist
    required_mapping = {
        "news_shock_signal": "news_shock_signal",
        "news_theme_drift_signal": "news_theme_drift_signal",
        "news_shock_has_coverage": "news_shock_has_coverage",
        "news_theme_drift_has_coverage": "news_theme_drift_has_coverage",
    }
    
    output = pd.DataFrame()
    output["symbol"] = df["symbol"]
    
    for fmp_col, ranker_col in required_mapping.items():
        if fmp_col in df.columns:
            output[ranker_col] = pd.to_numeric(df[fmp_col], errors="coerce").fillna(0.0)
        else:
            output[ranker_col] = 0.0
    
    return output
