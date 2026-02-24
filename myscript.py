#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit V5 WebSocket Scalp Signal Bot (NO TRADING)
- Universe (малопопулярные пары) выбирается через REST tickers (1 запрос на все).
- Реалтайм данные через WebSocket:
  * tickers.{symbol}
  * publicTrade.{symbol}  -> агрегируем 1s OHLCV
  * kline.1.{symbol}      -> 1m свечи
  * allLiquidation.{symbol}

Сигнал печатается ТОЛЬКО если:
- UP/DOWN явно доминирует (нет "спорно")
- confidence >= threshold
- совпадают направления 1m + 1s логики
- ликвидность/спред/трейды проходят фильтры
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import aiohttp
import numpy as np


# ----------------------------
# QUICK TUNING (edit values here)
# ----------------------------
# Это главный блок для ручной настройки поведения бота прямо в файле.
# Можно менять значения ниже без поиска по всему скрипту.
#
# CONFIDENCE_THRESHOLD: минимальная уверенность сигнала (чем выше, тем строже фильтр).
# MIN_MARGIN: минимальный отрыв лучшего направления (UP/DOWN) от альтернативы.
# MIN_TRADES_60S: минимальное число сделок за 60с для подтверждения активности.
#
# Быстрый ориентир по текущим ключевым значениям:
# CONFIDENCE_THRESHOLD=82;MIN_MARGIN=14;MIN_TRADES_60S=10
CONFIDENCE_THRESHOLD_DEFAULT = 82.0
MIN_MARGIN_DEFAULT = 14.0
MIN_TRADES_60S_DEFAULT = 10


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    # REST (для выбора "малопопулярных" и стартового прогрева 1m)
    rest_base_url: str = "https://api.bybit.com"
    category: str = "linear"  # spot | linear | inverse
    quote_suffix: str = "USDT"

    # WS endpoint (public)
    ws_url: str = "wss://stream.bybit.com/v5/public/linear"

    # Universe: low-turnover range (малопопулярные, но не совсем мертвые)
    universe_size: int = 60
    min_turnover_24h: float = 75_000.0
    max_turnover_24h: float = 12_000_000.0
    max_spread_pct: float = 0.006  # 0.6%

    # Warmup: сколько 1m свечей подтянуть REST-ом на старт (ускоряет запуск)
    seed_1m_limit: int = 120

    # 1s bars window (для анализа секундного графика)
    bars_1s_maxlen: int = 900   # 15 минут
    candles_1m_maxlen: int = 400

    # Filters for "не спорно"
    min_trades_60s: int = MIN_TRADES_60S_DEFAULT
    min_volume_60s: float = 0.0  # можно поднять, если хочешь отсечь тонкие
    min_1m_candles_ready: int = 30
    min_1s_bars_ready: int = 120

    # Signal strictness
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT
    min_score_threshold: float = 75.0
    min_margin: float = MIN_MARGIN_DEFAULT          # best_score - other_score
    cooldown_secs: int = 180          # per symbol+direction

    # Evaluation throttle
    eval_min_interval_secs: float = 1.0

    # Maintenance
    universe_refresh_secs: int = 300   # пересборка раз в 5 минут
    ping_interval_secs: int = 20       # по рекомендации Bybit

    # Output
    log_path: str = "scalp_signals.log"
    debug_print_interval_secs: float = 10.0
    forced_signal_interval_secs: int = 240


# ----------------------------
# REST helper
# ----------------------------

class BybitRest:
    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.session = session
        self.base_url = base_url.rstrip("/")

    async def get_tickers(self, category: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v5/market/tickers"
        async with self.session.get(url, params={"category": category}, timeout=15) as r:
            data = await r.json()
            if data.get("retCode") not in (0, "0", None):
                raise RuntimeError(f"REST tickers retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
            return data.get("result", {}).get("list", []) or []

    async def get_kline(self, category: str, symbol: str, interval: str, limit: int) -> List[List[str]]:
        url = f"{self.base_url}/v5/market/kline"
        params = {"category": category, "symbol": symbol, "interval": interval, "limit": str(limit)}
        async with self.session.get(url, params=params, timeout=15) as r:
            data = await r.json()
            if data.get("retCode") not in (0, "0", None):
                raise RuntimeError(f"REST kline retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
            return data.get("result", {}).get("list", []) or []


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        return float(s) if s else default
    except Exception:
        return default


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Bar1s:
    sec: int
    o: float
    h: float
    l: float
    c: float
    v: float
    buy_v: float
    sell_v: float
    trades: int


@dataclass
class Candle1m:
    start_ms: int
    o: float
    h: float
    l: float
    c: float
    v: float
    turnover: float


@dataclass
class SymbolState:
    symbol: str
    ticker: Dict[str, Any] = field(default_factory=dict)

    bars_1s: Deque[Bar1s] = field(default_factory=lambda: deque(maxlen=900))
    candles_1m: Deque[Candle1m] = field(default_factory=lambda: deque(maxlen=400))

    # current 1s builder
    cur_sec: Optional[int] = None
    cur_o: float = float("nan")
    cur_h: float = float("nan")
    cur_l: float = float("nan")
    cur_c: float = float("nan")
    cur_v: float = 0.0
    cur_buy_v: float = 0.0
    cur_sell_v: float = 0.0
    cur_trades: int = 0

    # liquidation events (ts_ms, side, size)
    liq: Deque[Tuple[int, str, float]] = field(default_factory=lambda: deque(maxlen=300))

    # for throttling / cooldown
    last_eval_ts: float = 0.0
    last_signal: Dict[str, float] = field(default_factory=dict)  # "UP"/"DOWN" -> ts

    # open interest history from tickers (ts_ms, oi)
    oi_hist: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=200))


# ----------------------------
# Indicators (fast-ish, last value only)
# ----------------------------

def ema_last(x: np.ndarray, span: int) -> float:
    if len(x) == 0:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    e = float(x[0])
    for v in x[1:]:
        e = alpha * float(v) + (1.0 - alpha) * e
    return e


def rsi_last(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    dif = np.diff(close.astype(float))
    gains = np.maximum(dif, 0.0)
    losses = np.maximum(-dif, 0.0)

    # Wilder smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_last(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    if len(close) < period:
        return (float("nan"), float("nan"), float("nan"))
    w = close[-period:].astype(float)
    mid = float(np.mean(w))
    sd = float(np.std(w))
    return (mid - num_std * sd, mid, mid + num_std * sd)


def atr_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    # Wilder smoothing
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + float(tr[i])) / period
    return atr


def adx_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    # simplified ADX last value
    if len(close) < (period * 2) + 2:
        return float("nan")

    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))

    # Wilder smoothing for TR, +DM, -DM
    atr = np.mean(tr[:period])
    p_dm = np.mean(plus_dm[:period])
    m_dm = np.mean(minus_dm[:period])

    dx_list = []
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
        p_dm = (p_dm * (period - 1) + plus_dm[i]) / period
        m_dm = (m_dm * (period - 1) + minus_dm[i]) / period

        if atr == 0:
            continue
        p_di = 100.0 * (p_dm / atr)
        m_di = 100.0 * (m_dm / atr)
        denom = (p_di + m_di)
        if denom == 0:
            continue
        dx = 100.0 * (abs(p_di - m_di) / denom)
        dx_list.append(dx)

    if len(dx_list) < period:
        return float("nan")

    # ADX as Wilder smoothing of DX
    adx = float(np.mean(dx_list[:period]))
    for v in dx_list[period:]:
        adx = (adx * (period - 1) + float(v)) / period
    return adx


def sigmoid(x: float) -> float:
    # stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ----------------------------
# Scoring (UP/DOWN + confidence)
# ----------------------------

@dataclass
class Signal:
    ts: float
    symbol: str
    direction: str  # UP / DOWN
    confidence: float
    price: float
    spread_pct: float
    reasons: List[str]
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        dt = datetime.fromtimestamp(self.ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return (
            f"[{dt}] {self.direction:<4} {self.symbol} | confidence={self.confidence:.0f}% "
            f"| price={self.price:g} | spread={self.spread_pct*100:.2f}% | horizon=5-10m\n"
            f"  reasons: " + "; ".join(self.reasons[:10])
        )

    def to_json(self) -> str:
        return json.dumps({
            "ts": self.ts,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "price": self.price,
            "spread_pct": self.spread_pct,
            "reasons": self.reasons,
            "debug": self.debug,
        }, ensure_ascii=False)


def evaluate(cfg: Config, st: SymbolState) -> Optional[Signal]:
    now = time.time()
    if now - st.last_eval_ts < cfg.eval_min_interval_secs:
        return None
    st.last_eval_ts = now

    # ----- basic ticker fields
    last_price = _sf(st.ticker.get("lastPrice"))
    bid = _sf(st.ticker.get("bid1Price"))
    ask = _sf(st.ticker.get("ask1Price"))
    bid_sz = _sf(st.ticker.get("bid1Size"), 0.0)
    ask_sz = _sf(st.ticker.get("ask1Size"), 0.0)
    turnover24h = _sf(st.ticker.get("turnover24h"), 0.0)

    if not (math.isfinite(last_price) and math.isfinite(bid) and math.isfinite(ask) and ask > 0 and bid > 0):
        return None

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid if mid > 0 else float("nan")
    if not math.isfinite(spread_pct) or spread_pct > cfg.max_spread_pct:
        return None

    # ----- data readiness
    if len(st.candles_1m) < cfg.min_1m_candles_ready:
        return None
    if len(st.bars_1s) < cfg.min_1s_bars_ready:
        return None

    # ----- 1s last 60s liquidity filters
    bars = list(st.bars_1s)
    last_60 = [b for b in bars if b.sec >= bars[-1].sec - 59]
    trades_60 = sum(b.trades for b in last_60)
    vol_60 = sum(b.v for b in last_60)
    if trades_60 < cfg.min_trades_60s:
        return None
    if vol_60 < cfg.min_volume_60s:
        return None

    # =========================
    # Build arrays for indicators
    # =========================
    c1m = np.array([c.c for c in st.candles_1m], dtype=float)
    h1m = np.array([c.h for c in st.candles_1m], dtype=float)
    l1m = np.array([c.l for c in st.candles_1m], dtype=float)
    v1m = np.array([c.v for c in st.candles_1m], dtype=float)

    c1s = np.array([b.c for b in st.bars_1s], dtype=float)
    v1s = np.array([b.v for b in st.bars_1s], dtype=float)
    buy1s = np.array([b.buy_v for b in st.bars_1s], dtype=float)
    sell1s = np.array([b.sell_v for b in st.bars_1s], dtype=float)

    # =========================
    # 1m analyzers (trend/strength/overheat)
    # =========================
    ema9 = ema_last(c1m[-120:], 9)
    ema21 = ema_last(c1m[-120:], 21)
    rsi1m = rsi_last(c1m[-200:], 14)
    atr1m = atr_last(h1m[-200:], l1m[-200:], c1m[-200:], 14)
    adx1m = adx_last(h1m[-220:], l1m[-220:], c1m[-220:], 14)
    bb_lo, bb_mid, bb_hi = bollinger_last(c1m[-200:], 20, 2.0)

    # VWAP last 10 candles (10m)
    vwap_10 = float(np.sum(c1m[-10:] * v1m[-10:]) / max(1e-12, np.sum(v1m[-10:]))) if np.sum(v1m[-10:]) > 0 else float("nan")

    trend_1m_up = (math.isfinite(ema9) and math.isfinite(ema21) and ema9 > ema21)
    trend_1m_down = (math.isfinite(ema9) and math.isfinite(ema21) and ema9 < ema21)

    # =========================
    # 1s analyzers (momentum/flow)
    # =========================
    # Momentum over 10s / 60s
    mom10 = (c1s[-1] / c1s[-11] - 1.0) if len(c1s) >= 11 else float("nan")
    mom60 = (c1s[-1] / c1s[-61] - 1.0) if len(c1s) >= 61 else float("nan")

    # 1s RSI over last 180s
    rsi1s = rsi_last(c1s[-220:], 14)

    # Trade imbalance (last 30s)
    win30 = 30
    buy30 = float(np.sum(buy1s[-win30:]))
    sell30 = float(np.sum(sell1s[-win30:]))
    tot30 = buy30 + sell30
    imbalance30 = (buy30 / tot30) if tot30 > 0 else 0.5  # 0..1

    # Bid/Ask top-of-book pressure (from ticker sizes)
    pressure = 0.0
    if (bid_sz + ask_sz) > 0:
        pressure = (bid_sz - ask_sz) / (bid_sz + ask_sz)  # [-1..1]

    # Liquidations last 60s
    ts_ms_cut = int((time.time() - 60) * 1000)
    liq_buy = sum(sz for (ts_ms, side, sz) in st.liq if ts_ms >= ts_ms_cut and side == "Buy")   # long liquidated
    liq_sell = sum(sz for (ts_ms, side, sz) in st.liq if ts_ms >= ts_ms_cut and side == "Sell") # short liquidated

    # Open interest change (approx) over last ~5 minutes (from tickers)
    oi_change_5m = float("nan")
    if len(st.oi_hist) >= 2:
        # find point ~5m ago
        now_ms = int(time.time() * 1000)
        target = now_ms - 5 * 60 * 1000
        oi_now = st.oi_hist[-1][1]
        oi_old = None
        for (tms, oi) in reversed(st.oi_hist):
            if tms <= target:
                oi_old = oi
                break
        if oi_old is not None and oi_old != 0:
            oi_change_5m = (oi_now - oi_old) / oi_old

    funding = _sf(st.ticker.get("fundingRate"))
    # =========================
    # Scoring
    # =========================
    up = 0.0
    dn = 0.0
    up_r: List[str] = []
    dn_r: List[str] = []

    analyzers_used = 0
    analyzers_agree_up = 0
    analyzers_agree_dn = 0

    def vote_up(weight: float, reason: str):
        nonlocal up, analyzers_used, analyzers_agree_up
        up += weight
        up_r.append(reason)
        analyzers_used += 1
        analyzers_agree_up += 1

    def vote_dn(weight: float, reason: str):
        nonlocal dn, analyzers_used, analyzers_agree_dn
        dn += weight
        dn_r.append(reason)
        analyzers_used += 1
        analyzers_agree_dn += 1

    def neutral_vote():
        nonlocal analyzers_used
        analyzers_used += 1

    # 1m trend
    if trend_1m_up:
        vote_up(14, "1m EMA9>EMA21")
    elif trend_1m_down:
        vote_dn(14, "1m EMA9<EMA21")
    else:
        neutral_vote()

    # 1m RSI (не лезем в перегрев)
    if math.isfinite(rsi1m):
        if rsi1m < 35:
            vote_up(8, f"1m RSI={rsi1m:.1f} (oversold)")
        elif rsi1m > 65:
            vote_dn(8, f"1m RSI={rsi1m:.1f} (overbought)")
        else:
            neutral_vote()

    # 1m ADX - сила тренда (если слабый, то спорно -> снижаем шанс сигналов)
    if math.isfinite(adx1m):
        if adx1m >= 22:
            # усиливаем текущее направление тренда
            if trend_1m_up:
                vote_up(10, f"1m ADX={adx1m:.0f} (trend ok)")
            elif trend_1m_down:
                vote_dn(10, f"1m ADX={adx1m:.0f} (trend ok)")
            else:
                neutral_vote()
        else:
            # слабый тренд — это не сигнал, но это повод НЕ быть уверенным
            neutral_vote()

    # 1m Bollinger impulse
    if math.isfinite(bb_lo) and math.isfinite(bb_hi):
        if c1m[-1] > bb_hi:
            vote_up(6, "1m close > upper Bollinger (impulse)")
        elif c1m[-1] < bb_lo:
            vote_dn(6, "1m close < lower Bollinger (impulse)")
        else:
            neutral_vote()

    # VWAP deviation (10m)
    if math.isfinite(vwap_10) and vwap_10 > 0:
        dev = (c1m[-1] / vwap_10 - 1.0)
        if dev > 0.0015 and trend_1m_up:
            vote_up(6, f"price above 10m VWAP ({dev*100:.2f}%)")
        elif dev < -0.0015 and trend_1m_down:
            vote_dn(6, f"price below 10m VWAP ({dev*100:.2f}%)")
        else:
            neutral_vote()

    # 1s momentum (10s + 60s) — базовый скальп-фактор
    if math.isfinite(mom10) and mom10 > 0.0008:
        vote_up(10, f"10s momentum {mom10*100:.2f}%")
    elif math.isfinite(mom10) and mom10 < -0.0008:
        vote_dn(10, f"10s momentum {mom10*100:.2f}%")
    else:
        neutral_vote()

    if math.isfinite(mom60) and mom60 > 0.0020:
        vote_up(12, f"60s momentum {mom60*100:.2f}%")
    elif math.isfinite(mom60) and mom60 < -0.0020:
        vote_dn(12, f"60s momentum {mom60*100:.2f}%")
    else:
        neutral_vote()

    # 1s RSI (перегрев/перепроданность) — как фильтр и подтверждение
    if math.isfinite(rsi1s):
        if rsi1s < 30:
            vote_up(6, f"1s RSI={rsi1s:.0f} (oversold)")
        elif rsi1s > 70:
            vote_dn(6, f"1s RSI={rsi1s:.0f} (overbought)")
        else:
            neutral_vote()

    # trade imbalance last 30s
    if tot30 > 0:
        if imbalance30 >= 0.60:
            vote_up(10, f"buy/sell imbalance={imbalance30:.2f}")
        elif imbalance30 <= 0.40:
            vote_dn(10, f"buy/sell imbalance={imbalance30:.2f}")
        else:
            neutral_vote()

    # top-of-book pressure
    if pressure > 0.18:
        vote_up(8, f"bid/ask pressure={pressure:+.2f}")
    elif pressure < -0.18:
        vote_dn(8, f"bid/ask pressure={pressure:+.2f}")
    else:
        neutral_vote()

    # Liquidation bursts (когда ликвидируют одну сторону — часто движение ускоряется)
    # По доке: S=Sell -> ликвидировали шорты; S=Buy -> ликвидировали лонги
    if (liq_sell + liq_buy) > 0:
        if liq_sell > liq_buy * 2 and liq_sell > 0:
            vote_up(8, "liquidation burst (shorts)")
        elif liq_buy > liq_sell * 2 and liq_buy > 0:
            vote_dn(8, "liquidation burst (longs)")
        else:
            neutral_vote()

    # OI change (подтверждение продолжения)
    if math.isfinite(oi_change_5m):
        if oi_change_5m > 0.02 and math.isfinite(mom60) and mom60 > 0:
            vote_up(6, f"OI +{oi_change_5m*100:.1f}% (5m) + price up")
        elif oi_change_5m > 0.02 and math.isfinite(mom60) and mom60 < 0:
            vote_dn(6, f"OI +{oi_change_5m*100:.1f}% (5m) + price down")
        else:
            neutral_vote()

    # Funding as crowding filter (слабый, чтобы не ломать скальп)
    if math.isfinite(funding):
        if funding > 0.0015:
            vote_dn(3, f"funding high {funding:+.4f} (crowded longs)")
        elif funding < -0.0015:
            vote_up(3, f"funding low {funding:+.4f} (crowded shorts)")
        else:
            neutral_vote()

    # =========================
    # Decision logic: strict "не спорно"
    # =========================
    best_dir = "UP" if up >= dn else "DOWN"
    best = up if best_dir == "UP" else dn
    other = dn if best_dir == "UP" else up
    margin = best - other

    # Agreement ratio (как доп. страховка от "спорно")
    agree = (analyzers_agree_up if best_dir == "UP" else analyzers_agree_dn)
    agree_ratio = agree / max(1, analyzers_used)

    # Confidence from margin + agreement (жестче)
    # sigmoid(margin/12) -> 0..1, потом умножаем на agreement_ratio
    conf = 100.0 * sigmoid(margin / 12.0) * (0.75 + 0.25 * agree_ratio)

    # Gate: must align 1m trend and 60s momentum with direction
    if best_dir == "UP":
        if not trend_1m_up:
            return None
        if not (math.isfinite(mom60) and mom60 > 0):
            return None
    else:
        if not trend_1m_down:
            return None
        if not (math.isfinite(mom60) and mom60 < 0):
            return None

    # Strict thresholds
    if best < cfg.min_score_threshold:
        return None
    if margin < cfg.min_margin:
        return None
    if conf < cfg.confidence_threshold:
        return None

    # Cooldown
    last_sig_ts = st.last_signal.get(best_dir, 0.0)
    if now - last_sig_ts < cfg.cooldown_secs:
        return None
    st.last_signal[best_dir] = now

    reasons = up_r if best_dir == "UP" else dn_r
    dbg = {
        "up_score": up, "down_score": dn, "margin": margin,
        "agree_ratio": agree_ratio,
        "rsi1m": rsi1m, "adx1m": adx1m, "atr1m": atr1m,
        "mom10": mom10, "mom60": mom60, "rsi1s": rsi1s,
        "imbalance30": imbalance30, "pressure": pressure,
        "liq_buy_60s": liq_buy, "liq_sell_60s": liq_sell,
        "oi_change_5m": oi_change_5m, "funding": funding,
        "turnover24h": turnover24h, "trades_60s": trades_60, "vol_60": vol_60,
    }

    return Signal(
        ts=now,
        symbol=st.symbol,
        direction=best_dir,
        confidence=conf,
        price=last_price,
        spread_pct=spread_pct,
        reasons=reasons,
        debug=dbg,
    )


def evaluate_best_effort(st: SymbolState) -> Optional[Signal]:
    """Всегда пытаемся собрать "лучший" сигнал по символу без жестких гейтов."""
    now = time.time()

    last_price = _sf(st.ticker.get("lastPrice"))
    bid = _sf(st.ticker.get("bid1Price"))
    ask = _sf(st.ticker.get("ask1Price"))
    if not (math.isfinite(last_price) and math.isfinite(bid) and math.isfinite(ask) and ask > 0 and bid > 0):
        return None

    spread_pct = (ask - bid) / ((bid + ask) / 2.0)

    if len(st.candles_1m) < 20 or len(st.bars_1s) < 61:
        return None

    c1m = np.array([c.c for c in st.candles_1m], dtype=float)
    c1s = np.array([b.c for b in st.bars_1s], dtype=float)
    buy1s = np.array([b.buy_v for b in st.bars_1s], dtype=float)
    sell1s = np.array([b.sell_v for b in st.bars_1s], dtype=float)

    ema9 = ema_last(c1m[-120:], 9)
    ema21 = ema_last(c1m[-120:], 21)
    mom10 = (c1s[-1] / c1s[-11] - 1.0) if len(c1s) >= 11 else 0.0
    mom60 = (c1s[-1] / c1s[-61] - 1.0) if len(c1s) >= 61 else 0.0

    buy30 = float(np.sum(buy1s[-30:]))
    sell30 = float(np.sum(sell1s[-30:]))
    total30 = buy30 + sell30
    imbalance30 = (buy30 / total30) if total30 > 0 else 0.5

    up = 0.0
    dn = 0.0
    reasons_up: List[str] = []
    reasons_dn: List[str] = []

    if math.isfinite(ema9) and math.isfinite(ema21):
        if ema9 > ema21:
            up += 22
            reasons_up.append("EMA9>EMA21")
        elif ema9 < ema21:
            dn += 22
            reasons_dn.append("EMA9<EMA21")

    if mom10 > 0:
        up += min(18, abs(mom10) * 18000)
        reasons_up.append(f"10S MOM {mom10*100:.2f}%")
    elif mom10 < 0:
        dn += min(18, abs(mom10) * 18000)
        reasons_dn.append(f"10S MOM {mom10*100:.2f}%")

    if mom60 > 0:
        up += min(24, abs(mom60) * 12000)
        reasons_up.append(f"60S MOM {mom60*100:.2f}%")
    elif mom60 < 0:
        dn += min(24, abs(mom60) * 12000)
        reasons_dn.append(f"60S MOM {mom60*100:.2f}%")

    if imbalance30 > 0.5:
        up += (imbalance30 - 0.5) * 40
        reasons_up.append(f"BUY FLOW {imbalance30:.2f}")
    elif imbalance30 < 0.5:
        dn += (0.5 - imbalance30) * 40
        reasons_dn.append(f"SELL FLOW {imbalance30:.2f}")

    best_dir = "UP" if up >= dn else "DOWN"
    best = max(up, dn)
    other = min(up, dn)
    margin = best - other
    confidence = max(35.0, min(99.0, 50.0 + margin))

    reasons = reasons_up if best_dir == "UP" else reasons_dn
    if not reasons:
        reasons = ["LOW-EDGE SNAPSHOT"]

    return Signal(
        ts=now,
        symbol=st.symbol,
        direction=best_dir,
        confidence=confidence,
        price=last_price,
        spread_pct=spread_pct,
        reasons=reasons,
        debug={"up_score": up, "down_score": dn, "best_effort": True},
    )


# ----------------------------
# Universe selection (малопопулярные)
# ----------------------------

def pick_universe(cfg: Config, tickers: List[Dict[str, Any]]) -> List[str]:
    rows: List[Tuple[str, float, float]] = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or not isinstance(sym, str):
            continue
        if not sym.endswith(cfg.quote_suffix):
            continue

        turnover = _sf(t.get("turnover24h"), 0.0)
        if turnover < cfg.min_turnover_24h or turnover > cfg.max_turnover_24h:
            continue

        bid = _sf(t.get("bid1Price"))
        ask = _sf(t.get("ask1Price"))
        if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0):
            continue
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid if mid > 0 else 1.0
        if spread > cfg.max_spread_pct:
            continue

        rows.append((sym, turnover, spread))

    # low turnover first
    rows.sort(key=lambda x: x[1])
    return [sym for sym, _, _ in rows[: cfg.universe_size]]


# ----------------------------
# WS client
# ----------------------------

class BybitWSBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.states: Dict[str, SymbolState] = {}
        self.topics: List[str] = []
        self._stop = False
        self.msg_stats: Dict[str, int] = {
            "tickers": 0,
            "trades": 0,
            "kline_1m": 0,
            "liq": 0,
            "signals": 0,
        }
        self.last_debug_ts: float = 0.0
        self.last_forced_signal_ts: float = 0.0

    def _print_debug(self):
        now = time.time()
        if (now - self.last_debug_ts) < self.cfg.debug_print_interval_secs:
            return
        self.last_debug_ts = now

        active_symbols = sum(1 for st in self.states.values() if st.bars_1s or st.candles_1m or st.ticker)
        print(
            "_________ "
            f"flow tickers={self.msg_stats['tickers']} trades={self.msg_stats['trades']} "
            f"kline_1m={self.msg_stats['kline_1m']} liq={self.msg_stats['liq']} "
            f"signals={self.msg_stats['signals']} active_symbols={active_symbols}/{len(self.states)}"
        )

    def _make_topics(self, symbols: List[str]) -> List[str]:
        out = []
        for s in symbols:
            out.append(f"tickers.{s}")
            out.append(f"publicTrade.{s}")
            out.append(f"kline.1.{s}")
            out.append(f"allLiquidation.{s}")
        return out

    def _emit_forced_top_signal(self):
        now = time.time()
        if (now - self.last_forced_signal_ts) < self.cfg.forced_signal_interval_secs:
            return

        best: Optional[Signal] = None
        for st in self.states.values():
            cand = evaluate_best_effort(st)
            if cand is None:
                continue
            if (best is None) or (cand.confidence > best.confidence):
                best = cand

        self.last_forced_signal_ts = now

        if best is None:
            print("🔥 TOP SIGNAL: NO READY SYMBOLS YET (NOT ENOUGH DATA)")
            return

        print(
            "🔥 TOP SIGNAL 4M | "
            f"{best.direction} {best.symbol} | CONFIDENCE={best.confidence:.1f}% | "
            f"PRICE={best.price:g} | SPREAD={best.spread_pct*100:.2f}%"
        )
        try:
            payload = {
                "forced_top_signal": True,
                "ts": best.ts,
                "symbol": best.symbol,
                "direction": best.direction,
                "confidence": best.confidence,
                "price": best.price,
                "spread_pct": best.spread_pct,
                "reasons": best.reasons,
                "debug": best.debug,
            }
            with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    async def _seed_1m_history(self, rest: BybitRest, symbols: List[str]) -> None:
        # concurrency-limited seeding
        sem = asyncio.Semaphore(10)

        async def seed_one(sym: str):
            async with sem:
                try:
                    rows = await rest.get_kline(self.cfg.category, sym, "1", self.cfg.seed_1m_limit)
                    # REST kline list is reverse sorted; build ascending, close-confirmed
                    # row: [startTime, open, high, low, close, volume, turnover]
                    candles: List[Candle1m] = []
                    for r in reversed(rows):
                        if len(r) < 7:
                            continue
                        candles.append(Candle1m(
                            start_ms=int(r[0]),
                            o=float(r[1]), h=float(r[2]), l=float(r[3]), c=float(r[4]),
                            v=float(r[5]), turnover=float(r[6]),
                        ))
                    st = self.states[sym]
                    st.candles_1m.clear()
                    for c in candles:
                        st.candles_1m.append(c)
                except Exception:
                    # тихо: если какой-то символ не отдал — не ломаем запуск
                    return

        await asyncio.gather(*(seed_one(s) for s in symbols))

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse):
        while not self._stop:
            try:
                await ws.send_str(json.dumps({"op": "ping"}))
            except Exception:
                return
            await asyncio.sleep(self.cfg.ping_interval_secs)

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse, topics: List[str]):
        # chunk by args total length safeguard (Bybit args length limit per connection)
        # We keep it conservative.
        chunk: List[str] = []
        chunk_len = 0
        MAX_CHARS = 18000

        async def send_chunk(ch: List[str]):
            if not ch:
                return
            msg = {"op": "subscribe", "args": ch}
            await ws.send_str(json.dumps(msg))

        for t in topics:
            add_len = len(t) + 3
            if chunk_len + add_len > MAX_CHARS:
                await send_chunk(chunk)
                chunk = []
                chunk_len = 0
            chunk.append(t)
            chunk_len += add_len

        await send_chunk(chunk)

    def _merge_ticker(self, st: SymbolState, data: Dict[str, Any], ts_ms: Optional[int]):
        # tickers has snapshot/delta. If a field is missing, it hasn't changed.
        st.ticker.update(data)
        if ts_ms is not None:
            oi = _sf(st.ticker.get("openInterest"))
            if math.isfinite(oi):
                st.oi_hist.append((ts_ms, oi))

    def _finalize_1s_bar(self, st: SymbolState, sec: int):
        if st.cur_sec is None:
            return
        bar = Bar1s(
            sec=st.cur_sec,
            o=st.cur_o, h=st.cur_h, l=st.cur_l, c=st.cur_c,
            v=st.cur_v, buy_v=st.cur_buy_v, sell_v=st.cur_sell_v,
            trades=st.cur_trades,
        )
        st.bars_1s.append(bar)

        # reset current
        st.cur_sec = sec
        st.cur_o = float("nan")
        st.cur_h = float("nan")
        st.cur_l = float("nan")
        st.cur_c = float("nan")
        st.cur_v = 0.0
        st.cur_buy_v = 0.0
        st.cur_sell_v = 0.0
        st.cur_trades = 0

    def _fill_missing_seconds(self, st: SymbolState, from_sec: int, to_sec: int):
        # fill (from_sec+1 .. to_sec-1) with flat bars at last known close
        if not st.bars_1s:
            return
        last_close = st.bars_1s[-1].c
        for s in range(from_sec + 1, to_sec):
            st.bars_1s.append(Bar1s(
                sec=s,
                o=last_close, h=last_close, l=last_close, c=last_close,
                v=0.0, buy_v=0.0, sell_v=0.0,
                trades=0,
            ))

    def _on_trade(self, st: SymbolState, tr: Dict[str, Any]):
        # trade fields: T(ms), s, S(Buy/Sell), v(size), p(price)
        t_ms = int(tr.get("T"))
        sec = t_ms // 1000
        price = _sf(tr.get("p"))
        size = _sf(tr.get("v"), 0.0)
        side = tr.get("S")

        if not math.isfinite(price):
            return

        if st.cur_sec is None:
            st.cur_sec = sec

        # if time moved forward
        if sec != st.cur_sec:
            # finalize previous
            prev_sec = st.cur_sec
            self._finalize_1s_bar(st, sec)

            # fill gaps
            if prev_sec is not None and sec > prev_sec + 1:
                self._fill_missing_seconds(st, prev_sec, sec)

        # init bar open if needed
        if not math.isfinite(st.cur_o):
            st.cur_o = price
            st.cur_h = price
            st.cur_l = price
            st.cur_c = price
        else:
            st.cur_h = max(st.cur_h, price)
            st.cur_l = min(st.cur_l, price)
            st.cur_c = price

        st.cur_v += size
        st.cur_trades += 1
        if side == "Buy":
            st.cur_buy_v += size
        elif side == "Sell":
            st.cur_sell_v += size

    def _on_kline_1m(self, st: SymbolState, k: Dict[str, Any]):
        # store only confirmed closed candles
        if not k.get("confirm", False):
            return
        start_ms = int(k.get("start"))
        c = Candle1m(
            start_ms=start_ms,
            o=_sf(k.get("open")),
            h=_sf(k.get("high")),
            l=_sf(k.get("low")),
            c=_sf(k.get("close")),
            v=_sf(k.get("volume"), 0.0),
            turnover=_sf(k.get("turnover"), 0.0),
        )
        # avoid duplicates if reconnect sends same candle
        if st.candles_1m and st.candles_1m[-1].start_ms == c.start_ms:
            st.candles_1m[-1] = c
        else:
            st.candles_1m.append(c)

    def _on_liq(self, st: SymbolState, liq: Dict[str, Any]):
        # fields: T, s, S(Buy/Sell), v(size), p(bankruptcy price)
        t_ms = int(liq.get("T"))
        side = str(liq.get("S"))
        size = _sf(liq.get("v"), 0.0)
        if size > 0:
            st.liq.append((t_ms, side, size))

    async def run(self):
        cfg = self.cfg
        print("WS Scalp Signal Bot started (NO TRADING). Ctrl+C to stop.")
        print(f"WS: {cfg.ws_url}")
        print(f"Category: {cfg.category} | Universe: {cfg.universe_size} low-turnover symbols")

        connector = aiohttp.TCPConnector(limit=200)
        async with aiohttp.ClientSession(connector=connector) as session:
            rest = BybitRest(session, cfg.rest_base_url)

            # main loop: keep reconnecting WS if needed
            symbols: List[str] = []
            last_universe_refresh = 0.0

            while not self._stop:
                try:
                    # refresh universe
                    if (time.time() - last_universe_refresh) > cfg.universe_refresh_secs or not symbols:
                        all_tickers = await rest.get_tickers(cfg.category)
                        symbols = pick_universe(cfg, all_tickers)
                        last_universe_refresh = time.time()

                        self.states = {s: SymbolState(symbol=s) for s in symbols}
                        # set deques maxlen from cfg
                        for st in self.states.values():
                            st.bars_1s = deque(maxlen=cfg.bars_1s_maxlen)
                            st.candles_1m = deque(maxlen=cfg.candles_1m_maxlen)

                        # seed 1m history to avoid long warmup
                        await self._seed_1m_history(rest, symbols)

                        self.topics = self._make_topics(symbols)
                        print(f"Universe updated: {len(symbols)} symbols, topics={len(self.topics)}")
                        print("_________ WS connected, waiting for market data...")

                    async with session.ws_connect(cfg.ws_url, heartbeat=None, autoping=False) as ws:
                        self._stop = False
                        ping_task = asyncio.create_task(self._ping_loop(ws))

                        await self._subscribe(ws, self.topics)

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                except Exception:
                                    continue

                                # ignore command responses / pongs
                                if isinstance(data, dict) and data.get("op") in ("pong", "ping"):
                                    continue
                                if isinstance(data, dict) and data.get("type") == "COMMAND_RESP":
                                    continue

                                topic = data.get("topic")
                                if not topic:
                                    continue

                                # Ticker / Trade / Kline / Liquidation messages
                                if topic.startswith("tickers."):
                                    self.msg_stats["tickers"] += 1
                                    payload = data.get("data")
                                    ts_ms = data.get("ts")
                                    if isinstance(payload, dict):
                                        sym = payload.get("symbol")
                                        if sym in self.states:
                                            self._merge_ticker(self.states[sym], payload, ts_ms if isinstance(ts_ms, int) else None)
                                    elif isinstance(payload, list):
                                        for item in payload:
                                            if isinstance(item, dict):
                                                sym = item.get("symbol")
                                                if sym in self.states:
                                                    self._merge_ticker(self.states[sym], item, ts_ms if isinstance(ts_ms, int) else None)

                                elif topic.startswith("publicTrade."):
                                    self.msg_stats["trades"] += 1
                                    payload = data.get("data")
                                    if isinstance(payload, list) and payload:
                                        # symbol can be in each trade item, but topic also carries it
                                        for tr in payload:
                                            if not isinstance(tr, dict):
                                                continue
                                            sym = tr.get("s")
                                            if sym in self.states:
                                                self._on_trade(self.states[sym], tr)

                                        # after updating, try evaluate for these symbols
                                        touched = set(tr.get("s") for tr in payload if isinstance(tr, dict))
                                        for sym in touched:
                                            if sym in self.states:
                                                sig = evaluate(cfg, self.states[sym])
                                                if sig:
                                                    self.msg_stats["signals"] += 1
                                                    print(sig.to_text())
                                                    try:
                                                        with open(cfg.log_path, "a", encoding="utf-8") as f:
                                                            f.write(sig.to_json() + "\n")
                                                    except Exception:
                                                        pass

                                elif topic.startswith("kline.1."):
                                    self.msg_stats["kline_1m"] += 1
                                    payload = data.get("data")
                                    if isinstance(payload, list):
                                        for k in payload:
                                            if not isinstance(k, dict):
                                                continue
                                            # symbol is in topic, but in payload sometimes not
                                            sym = topic.split(".")[-1]
                                            if sym in self.states:
                                                self._on_kline_1m(self.states[sym], k)

                                elif topic.startswith("allLiquidation."):
                                    self.msg_stats["liq"] += 1
                                    payload = data.get("data")
                                    sym = topic.split(".")[-1]
                                    if sym in self.states:
                                        if isinstance(payload, list):
                                            for liq in payload:
                                                if isinstance(liq, dict):
                                                    self._on_liq(self.states[sym], liq)
                                        elif isinstance(payload, dict):
                                            self._on_liq(self.states[sym], payload)

                                self._print_debug()
                                self._emit_forced_top_signal()

                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                                break

                        ping_task.cancel()

                except KeyboardInterrupt:
                    self._stop = True
                    print("Stopping...")
                    return
                except Exception as e:
                    # reconnect
                    print(f"WS loop error: {e}. Reconnecting...")
                    await asyncio.sleep(1.5)


def load_config() -> Config:
    cfg = Config()

    # allow override via env
    cfg.rest_base_url = os.getenv("BYBIT_REST_URL", cfg.rest_base_url)
    cfg.category = os.getenv("BYBIT_CATEGORY", cfg.category)

    # choose WS url based on category if user didn't override
    ws_override = os.getenv("BYBIT_WS_URL", "").strip()
    if ws_override:
        cfg.ws_url = ws_override
    else:
        if cfg.category == "linear":
            cfg.ws_url = "wss://stream.bybit.com/v5/public/linear"
        elif cfg.category == "inverse":
            cfg.ws_url = "wss://stream.bybit.com/v5/public/inverse"
        elif cfg.category == "spot":
            cfg.ws_url = "wss://stream.bybit.com/v5/public/spot"
        else:
            cfg.ws_url = "wss://stream.bybit.com/v5/public/linear"

    cfg.universe_size = int(os.getenv("UNIVERSE_SIZE", cfg.universe_size))
    cfg.min_turnover_24h = float(os.getenv("MIN_TURNOVER_24H", cfg.min_turnover_24h))
    cfg.max_turnover_24h = float(os.getenv("MAX_TURNOVER_24H", cfg.max_turnover_24h))
    cfg.max_spread_pct = float(os.getenv("MAX_SPREAD_PCT", cfg.max_spread_pct))

    cfg.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", cfg.confidence_threshold))
    cfg.min_score_threshold = float(os.getenv("MIN_SCORE_THRESHOLD", cfg.min_score_threshold))
    cfg.min_margin = float(os.getenv("MIN_MARGIN", cfg.min_margin))
    cfg.cooldown_secs = int(os.getenv("COOLDOWN_SECS", cfg.cooldown_secs))

    cfg.min_trades_60s = int(os.getenv("MIN_TRADES_60S", cfg.min_trades_60s))
    cfg.min_1m_candles_ready = int(os.getenv("MIN_1M_READY", cfg.min_1m_candles_ready))
    cfg.min_1s_bars_ready = int(os.getenv("MIN_1S_READY", cfg.min_1s_bars_ready))

    cfg.universe_refresh_secs = int(os.getenv("UNIVERSE_REFRESH_SECS", cfg.universe_refresh_secs))
    cfg.ping_interval_secs = int(os.getenv("PING_INTERVAL_SECS", cfg.ping_interval_secs))

    cfg.log_path = os.getenv("LOG_PATH", cfg.log_path)
    cfg.debug_print_interval_secs = float(os.getenv("DEBUG_PRINT_INTERVAL_SECS", cfg.debug_print_interval_secs))
    cfg.forced_signal_interval_secs = int(os.getenv("FORCED_SIGNAL_INTERVAL_SECS", cfg.forced_signal_interval_secs))
    return cfg


async def main():
    cfg = load_config()
    bot = BybitWSBot(cfg)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
