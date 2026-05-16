# src/agents/strategy_agent.py
from typing import List, Union
from ..schemas import Recommendation, Candle, NewsItem
from textblob import TextBlob


# ── Technical Indicators ────────────────────────────────────────────────────

def _ema(closes: List[float], period: int) -> List[float]:
    """Exponential Moving Average."""
    if len(closes) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def _rsi(closes: List[float], period: int = 14) -> float:
    """RSI over the last `period` candles. Returns value 0–100."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(closes: List[float]) -> tuple[float, float]:
    """Returns (macd_line, signal_line). Positive macd_line = bullish."""
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if not ema12 or not ema26:
        return 0.0, 0.0
    # Align lengths
    diff = len(ema12) - len(ema26)
    ema12 = ema12[diff:]
    macd_line = [ema12[i] - ema26[i] for i in range(len(ema26))]
    signal = _ema(macd_line, 9)
    if not signal:
        return macd_line[-1], 0.0
    return macd_line[-1], signal[-1]


# ── Sentiment ────────────────────────────────────────────────────────────────

def _headline_sentiment(text: str) -> float:
    try:
        return TextBlob(text.strip()).sentiment.polarity if text.strip() else 0.0
    except Exception:
        return 0.0


# ── Main Strategy ────────────────────────────────────────────────────────────

def simple_strategy(
    pair: str,
    candles: List[Union[Candle, dict]],
    news: List[Union[NewsItem, dict]],
) -> Recommendation:
    """
    Multi-signal strategy:
      - Price momentum (daily move)
      - RSI (overbought/oversold)
      - EMA 20/50 crossover
      - MACD crossover
      - News sentiment
    Confidence is derived from how many signals agree.
    """

    # Normalize candles
    cleaned = []
    for c in candles:
        if isinstance(c, dict):
            try:
                cleaned.append(Candle(**c))
            except Exception:
                continue
        elif isinstance(c, Candle):
            cleaned.append(c)

    if len(cleaned) < 2:
        return Recommendation(
            pair=pair, stance="AVOID", confidence=0.0,
            horizon_hours=24, rationale=["Not enough candle data"], news=[],
        )

    closes = [c.close for c in cleaned]
    rationale = []

    # ── 1. Price momentum ───────────────────────────────────────────────────
    daily_move = closes[-1] - closes[-2]
    momentum_signal = 1 if daily_move > 0.0005 else (-1 if daily_move < -0.0005 else 0)
    rationale.append(f"Daily move: {daily_move:+.5f} → {'BUY' if momentum_signal == 1 else 'SELL' if momentum_signal == -1 else 'NEUTRAL'}")

    # ── 2. RSI ──────────────────────────────────────────────────────────────
    rsi = _rsi(closes)
    if rsi < 35:
        rsi_signal = 1   # oversold → BUY
    elif rsi > 65:
        rsi_signal = -1  # overbought → SELL
    else:
        rsi_signal = 0
    rationale.append(f"RSI(14): {rsi:.1f} → {'oversold/BUY' if rsi_signal == 1 else 'overbought/SELL' if rsi_signal == -1 else 'neutral'}")

    # ── 3. EMA 20/50 crossover ──────────────────────────────────────────────
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    if ema20 and ema50:
        ema_signal = 1 if ema20[-1] > ema50[-1] else -1
        rationale.append(f"EMA20({ema20[-1]:.5f}) {'>' if ema_signal == 1 else '<'} EMA50({ema50[-1]:.5f}) → {'bullish' if ema_signal == 1 else 'bearish'}")
    else:
        ema_signal = 0
        rationale.append("EMA crossover: insufficient data")

    # ── 4. MACD ─────────────────────────────────────────────────────────────
    macd_line, signal_line = _macd(closes)
    if macd_line != 0.0 or signal_line != 0.0:
        macd_signal = 1 if macd_line > signal_line else -1
        rationale.append(f"MACD({macd_line:+.6f}) vs Signal({signal_line:+.6f}) → {'bullish' if macd_signal == 1 else 'bearish'}")
    else:
        macd_signal = 0
        rationale.append("MACD: insufficient data")

    # ── 5. News sentiment ───────────────────────────────────────────────────
    cleaned_news, scores, seen = [], [], set()
    for n in (news or [])[:10]:
        if isinstance(n, dict):
            try:
                n = NewsItem(**n)
            except Exception:
                continue
        title = (n.title or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        s = _headline_sentiment(title)
        scores.append(s)
        cleaned_news.append(n)
        rationale.append(f"{title} [{n.source or 'Unknown'}] sentiment={s:+.2f}")

    avg_sentiment = sum(scores) / len(scores) if scores else 0.0
    sentiment_signal = 1 if avg_sentiment > 0.15 else (-1 if avg_sentiment < -0.15 else 0)
    rationale.append(f"Avg news sentiment: {avg_sentiment:+.2f} → {'positive' if sentiment_signal == 1 else 'negative' if sentiment_signal == -1 else 'neutral'}")

    # ── Aggregate signals ───────────────────────────────────────────────────
    signals = [momentum_signal, rsi_signal, ema_signal, macd_signal, sentiment_signal]
    active = [s for s in signals if s != 0]
    bull = active.count(1)
    bear = active.count(-1)

    if not active:
        stance = "AVOID"
        confidence = 0.3
    elif bull > bear:
        stance = "BUY"
        confidence = round(0.4 + (bull / len(signals)) * 0.6, 2)
    elif bear > bull:
        stance = "SELL"
        confidence = round(0.4 + (bear / len(signals)) * 0.6, 2)
    else:
        stance = "AVOID"
        confidence = 0.35  # signals split evenly — no edge

    rationale.append(f"Signal tally — BUY:{bull} SELL:{bear} NEUTRAL:{signals.count(0)} → {stance} @ {confidence:.0%}")
    rationale.append("This is not financial advice.")

    return Recommendation(
        pair=pair,
        stance=stance,
        confidence=confidence,
        horizon_hours=24,
        rationale=rationale,
        news=cleaned_news,
    )
