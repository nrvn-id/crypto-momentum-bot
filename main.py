"""
OKX Trend Scanner Bot (GitHub Actions edition)
------------------------------------------------
Pengganti bot momentum lama. Tiap kali dijalankan (via GitHub Actions,
cron tiap 1 jam), script ini:
1. Ambil candle 15m dari OKX (public API, tanpa API key) untuk pair yang dipantau
2. Hitung skor trend 0-100 (EMA alignment, Supertrend, ADX, Ichimoku, RSI, volume)
3. Kirim SATU pesan ringkasan ke Telegram berisi semua pair, diurutkan dari
   skor tertinggi, dengan tanda khusus untuk yang sinyalnya kuat.

ENV VARS (diisi lewat GitHub Secrets, nama sama seperti bot lama):
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import math

import requests
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

OKX_BASE_URL = "https://www.okx.com"

SYMBOLS = [
    "HYPE-USDT-SWAP",
    "KAITO-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "BTC-USDT-SWAP",
]

BAR = "15m"                # timeframe candle yang dibaca tiap run
CANDLE_LIMIT = 200
STRONG_SCORE_THRESHOLD = 75  # dapat tanda 🔥 kalau skor >= ini

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ============================== DATA FETCH ==============================

def fetch_candles(inst_id: str, bar: str = BAR, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX error untuk {inst_id}: {data}")

    rows = data["data"]
    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df = df.iloc[::-1].reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    return df


# ============================== INDIKATOR ==============================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_atr = atr(df, length)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean().fillna(0)


def supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2
    tr_atr = atr(df, length)
    upper_basic = hl2 + mult * tr_atr
    lower_basic = hl2 - mult * tr_atr

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        final_upper.iloc[i] = (
            upper_basic.iloc[i]
            if (upper_basic.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1])
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            lower_basic.iloc[i]
            if (lower_basic.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1])
            else final_lower.iloc[i - 1]
        )

    for i in range(1, len(df)):
        if df["close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and final_lower.iloc[i] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            if direction.iloc[i] == -1 and final_upper.iloc[i] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

    return direction


def ichimoku(df: pd.DataFrame):
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    return {"bull": close > cloud_top, "bear": close < cloud_bottom}


# ============================== SCORING ==============================

def score_label(score: int) -> str:
    if score >= 80:
        return "VERY STRONG"
    if score >= 60:
        return "STRONG"
    if score >= 40:
        return "MODERATE"
    return "WEAK"


def analyze(df: pd.DataFrame, inst_id: str) -> dict:
    close = df["close"]

    ema20, ema50, ema100, ema200 = ema(close, 20), ema(close, 50), ema(close, 100), ema(close, 200)
    ema_bull = ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1] > ema200.iloc[-1]
    ema_bear = ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1] < ema200.iloc[-1]

    st_dir = supertrend(df)
    st_bull = st_dir.iloc[-1] == 1

    adx_series = adx(df)
    adx_now = adx_series.iloc[-1]
    adx_rising = adx_now > adx_series.iloc[-3] if len(adx_series) > 3 else False

    ich = ichimoku(df)
    ich_bull = bool(ich["bull"].iloc[-1])
    ich_bear = bool(ich["bear"].iloc[-1])

    rsi_now = rsi(close).iloc[-1]
    rsi_bull = rsi_now > 55
    rsi_bear = rsi_now < 45

    vol_avg = df["vol"].rolling(20).mean().iloc[-1]
    vol_now = df["vol"].iloc[-1]
    vol_ratio = (vol_now / vol_avg) if vol_avg and not math.isnan(vol_avg) else 1.0
    vol_high = vol_ratio >= 1.5

    checks_long = [ema_bull, st_bull, adx_rising, ich_bull, rsi_bull, vol_high]
    checks_short = [ema_bear, not st_bull, adx_rising, ich_bear, rsi_bear, vol_high]

    long_score = int(sum(checks_long) / len(checks_long) * 100)
    short_score = int(sum(checks_short) / len(checks_short) * 100)

    if long_score >= short_score and long_score > 0:
        direction, score = "LONG", long_score
    elif short_score > long_score:
        direction, score = "SHORT", short_score
    else:
        direction, score = "NEUTRAL", 0

    return {
        "inst_id": inst_id,
        "price": float(close.iloc[-1]),
        "direction": direction,
        "score": score,
        "label": score_label(score),
        "vol_ratio": round(float(vol_ratio), 2),
        "adx": round(float(adx_now), 1),
    }


# ============================== TELEGRAM ==============================

def format_summary_message(results: list) -> str:
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    lines = [f"📊 *OKX Trend Scan* ({BAR})\n"]
    for r in ranked:
        flag = "🔥 " if r["score"] >= STRONG_SCORE_THRESHOLD and r["direction"] != "NEUTRAL" else ""
        arrow = "🟢" if r["direction"] == "LONG" else ("🔴" if r["direction"] == "SHORT" else "⚪")
        lines.append(
            f"{flag}{arrow} *{r['inst_id']}* — {r['score']} ({r['label']}) | {r['direction']}\n"
            f"    Harga: {r['price']:,.4f} | ADX {r['adx']} | Vol {r['vol_ratio']}x"
        )
    lines.append("\n_Data: OKX public API. Bukan sinyal beli, lakukan riset lanjutan._")
    return "\n".join(lines)


def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


# ============================== MAIN ==============================

def main():
    results = []
    for inst_id in SYMBOLS:
        try:
            df = fetch_candles(inst_id)
            if len(df) < 60:
                print(f"[SKIP] {inst_id}: candle terlalu sedikit ({len(df)})")
                continue
            results.append(analyze(df, inst_id))
        except Exception as e:
            print(f"[ERROR] {inst_id}: {e}")

    if not results:
        print("Tidak ada data yang berhasil dianalisis, tidak mengirim pesan.")
        return

    message = format_summary_message(results)
    send_telegram(message)
    print("Terkirim.")
    print(message)


if __name__ == "__main__":
    main()
