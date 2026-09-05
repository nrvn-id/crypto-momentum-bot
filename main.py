"""
OKX Trend Scanner Bot (GitHub Actions edition) — v2
------------------------------------------------------
Perubahan dari versi sebelumnya:
- Timeframe candle: 1H (bukan 15m)
- Scan SEMUA pair futures perpetual USDT-margined yang aktif di OKX
  (bukan cuma 4 pair tetap)
- Hanya kirim pair yang STRONG BUY: direction == LONG dan score >= STRONG_SCORE_THRESHOLD

ENV VARS (GitHub Secrets, nama sama seperti sebelumnya):
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import math
import time

import requests
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

OKX_BASE_URL = "https://www.okx.com"

BAR = "1H"                     # timeframe candle
CANDLE_LIMIT = 200
SETTLE_CCY = "USDT"            # cuma pair futures yang settle-nya USDT
STRONG_SCORE_THRESHOLD = 75    # dianggap "strong buy" kalau skor >= ini
MAX_RESULTS_IN_MESSAGE = 25    # batasi jumlah baris per pesan biar tidak kepanjangan
REQUEST_DELAY_SEC = 0.15       # jeda antar request supaya tidak kena rate limit OKX

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ============================== DATA FETCH ==============================

def fetch_active_swap_instruments(settle_ccy: str = SETTLE_CCY) -> list:
    """Ambil semua instId futures perpetual (SWAP) yang aktif & settle di USDT."""
    url = f"{OKX_BASE_URL}/api/v5/public/instruments"
    resp = requests.get(url, params={"instType": "SWAP"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise RuntimeError(f"Gagal ambil daftar instrument OKX: {data}")

    instruments = data["data"]
    filtered = [
        i["instId"] for i in instruments
        if i.get("settleCcy") == settle_ccy and i.get("state") == "live"
    ]
    return sorted(filtered)


def fetch_candles(inst_id: str, bar: str = BAR, limit: int = CANDLE_LIMIT, retries: int = 3) -> pd.DataFrame:
    url = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": str(limit)}

    data = None
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(1 + attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX error untuk {inst_id}: {data}")
        break
    if data is None:
        raise RuntimeError(f"Rate limited terus untuk {inst_id}, dilewati")

    rows = data["data"]
    if not rows:
        raise RuntimeError(f"Tidak ada data candle untuk {inst_id}")

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

    st_dir = supertrend(df)
    st_bull = st_dir.iloc[-1] == 1

    adx_series = adx(df)
    adx_now = adx_series.iloc[-1]
    adx_rising = adx_now > adx_series.iloc[-3] if len(adx_series) > 3 else False

    ich = ichimoku(df)
    ich_bull = bool(ich["bull"].iloc[-1])

    rsi_now = rsi(close).iloc[-1]
    rsi_bull = rsi_now > 55

    vol_avg = df["vol"].rolling(20).mean().iloc[-1]
    vol_now = df["vol"].iloc[-1]
    vol_ratio = (vol_now / vol_avg) if vol_avg and not math.isnan(vol_avg) else 1.0
    vol_high = vol_ratio >= 1.5

    ema_bear = ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1] < ema200.iloc[-1]
    ich_bear = bool(ich["bear"].iloc[-1])
    rsi_bear = rsi_now < 45

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

def format_strong_buy_message(strong_buys: list, total_scanned: int) -> list:
    """Return list of message chunks (Telegram limit ~4096 char per pesan)."""
    header = (
        f"📊 *OKX Strong Buy Scan* ({BAR})\n"
        f"Discan: {total_scanned} pair futures | Kriteria: LONG & skor >= {STRONG_SCORE_THRESHOLD}\n"
    )

    if not strong_buys:
        return [header + "\nTidak ada pair yang memenuhi kriteria strong buy saat ini."]

    shown = strong_buys[:MAX_RESULTS_IN_MESSAGE]
    lines = [header, ""]
    for r in shown:
        lines.append(
            f"🟢 *{r['inst_id']}* — {r['score']} ({r['label']})\n"
            f"    Harga: {r['price']:,.4f} | ADX {r['adx']} | Vol {r['vol_ratio']}x"
        )

    if len(strong_buys) > MAX_RESULTS_IN_MESSAGE:
        lines.append(f"\n...dan {len(strong_buys) - MAX_RESULTS_IN_MESSAGE} pair lain juga strong buy.")

    lines.append("\n_Data: OKX public API. Bukan sinyal beli, lakukan riset lanjutan._")

    text = "\n".join(lines)

    max_len = 3800
    if len(text) <= max_len:
        return [text]

    chunks, current = [], header + "\n"
    for line in lines[2:]:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


# ============================== MAIN ==============================

def main():
    symbols = fetch_active_swap_instruments()
    print(f"Total pair futures USDT aktif di OKX: {len(symbols)}")

    results = []
    for inst_id in symbols:
        try:
            df = fetch_candles(inst_id)
            if len(df) < 60:
                print(f"[SKIP] {inst_id}: candle terlalu sedikit ({len(df)})")
                continue
            results.append(analyze(df, inst_id))
        except Exception as e:
            print(f"[ERROR] {inst_id}: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    print(f"Berhasil dianalisis: {len(results)} / {len(symbols)}")

    strong_buys = [r for r in results if r["direction"] == "LONG" and r["score"] >= STRONG_SCORE_THRESHOLD]
    strong_buys.sort(key=lambda r: r["score"], reverse=True)
    print(f"Strong buy ditemukan: {len(strong_buys)}")

    chunks = format_strong_buy_message(strong_buys, total_scanned=len(results))
    for chunk in chunks:
        send_telegram(chunk)

    print("Terkirim.")


if __name__ == "__main__":
    main()