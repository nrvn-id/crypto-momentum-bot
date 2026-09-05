"""
OKX Trend Scanner Bot (GitHub Actions edition) — v3
------------------------------------------------------
Ganti total metode filter dari versi sebelumnya (EMA/Supertrend/ADX/Ichimoku
score) menjadi mengikuti cara riset manual kamu:

1. TIMEFRAME ENTRY (1H): cari StochRSI %K cross up %D dari area oversold
   -> ini "pemicu" sinyal, nandain momentum baru mulai belok naik.
2. KONFIRMASI 4H: RSI & StochRSI belum overbought/overextended (masih ada
   ruang naik), bias netral-ke-bullish.
3. KONFIRMASI 1D: sama seperti 4H tapi di timeframe harian -> memastikan
   trend besar juga tidak sedang jenuh beli.
   Pair cuma lolos jadi kandidat kalau LOLOS KETIGANYA. Ini sengaja ketat
   supaya yang lolos cuma sedikit pair yang benar teknikal, bukan yang
   sudah "terbang tinggi".
4. Untuk pair yang lolos, dihitung level Fibonacci retracement/extension
   dari swing 4H sebagai REFERENSI Entry/TP/SL (bukan rekomendasi final —
   tetap riset manual sebelum entry).

Optimisasi request: candle 4H & 1D cuma diambil untuk pair yang SUDAH lolos
cross di 1H, jadi tidak perlu fetch 3x untuk semua ratusan pair.

ENV VARS (GitHub Secrets):
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import time

import requests
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

OKX_BASE_URL = "https://www.okx.com"

ENTRY_BAR = "1H"     # timeframe pemicu sinyal (StochRSI cross)
CONFIRM_BAR_1 = "4H"
CONFIRM_BAR_2 = "1D"

CANDLE_LIMIT_ENTRY = 200
CANDLE_LIMIT_CONFIRM = 150

SETTLE_CCY = "USDT"
REQUEST_DELAY_SEC = 0.15

RSI_LENGTH = 14
STOCH_LENGTH = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3
OVERSOLD = 20
OVERBOUGHT = 80
CROSS_LOOKBACK = 5          # cek "baru saja oversold" dalam N candle terakhir
CONFIRM_RSI_MAX = 70        # di atas ini dianggap sudah terlalu jenuh beli
CONFIRM_RSI_MIN = 45        # di bawah ini dianggap belum ada bias bullish

FIB_LOOKBACK = 60           # jumlah candle 4H untuk cari swing high/low

MAX_RESULTS_IN_MESSAGE = 10

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ============================== DATA FETCH ==============================

def fetch_active_swap_instruments(settle_ccy: str = SETTLE_CCY) -> list:
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


def fetch_candles(inst_id: str, bar: str, limit: int, retries: int = 3) -> pd.DataFrame:
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
            raise RuntimeError(f"OKX error untuk {inst_id} ({bar}): {data}")
        break
    if data is None:
        raise RuntimeError(f"Rate limited terus untuk {inst_id} ({bar})")

    rows = data["data"]
    if not rows:
        raise RuntimeError(f"Tidak ada data candle untuk {inst_id} ({bar})")

    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df = df.iloc[::-1].reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    return df


# ============================== INDIKATOR ==============================

def rsi(series: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def stoch_rsi(close: pd.Series, rsi_length=RSI_LENGTH, stoch_length=STOCH_LENGTH,
              k_smooth=STOCH_K_SMOOTH, d_smooth=STOCH_D_SMOOTH):
    """Return (rsi_series, k, d). K/D dalam skala 0-100."""
    rsi_series = rsi(close, rsi_length)
    min_rsi = rsi_series.rolling(stoch_length).min()
    max_rsi = rsi_series.rolling(stoch_length).max()
    raw = (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    k = raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return rsi_series, k.fillna(50), d.fillna(50)


def fibonacci_levels(df: pd.DataFrame, lookback: int = FIB_LOOKBACK) -> dict:
    recent = df.tail(lookback)
    swing_low = float(recent["low"].min())
    swing_high = float(recent["high"].max())
    diff = swing_high - swing_low
    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "ext_1.272": swing_high + 0.272 * diff,
        "ext_1.618": swing_high + 0.618 * diff,
    }


# ============================== SIGNAL LOGIC ==============================

def entry_cross_signal(df: pd.DataFrame) -> dict:
    """StochRSI %K cross up %D di candle terakhir, dan sempat oversold
    dalam beberapa candle sebelumnya (bukan cross di tengah range)."""
    _, k, d = stoch_rsi(df["close"])
    if len(k) < CROSS_LOOKBACK + 2:
        return {"ok": False}

    crossed_up = k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
    recently_oversold = (
        k.iloc[-(CROSS_LOOKBACK + 1):-1].min() <= OVERSOLD
        or d.iloc[-(CROSS_LOOKBACK + 1):-1].min() <= OVERSOLD
    )
    return {
        "ok": bool(crossed_up and recently_oversold),
        "k": round(float(k.iloc[-1]), 1),
        "d": round(float(d.iloc[-1]), 1),
    }


def confirm_not_overextended(df: pd.DataFrame) -> dict:
    """Cek RSI & StochRSI di timeframe lebih besar: belum jenuh beli,
    tapi juga bukan bias bearish."""
    rsi_series, k, d = stoch_rsi(df["close"])
    rsi_now = float(rsi_series.iloc[-1])
    k_now = float(k.iloc[-1])
    ok = (CONFIRM_RSI_MIN <= rsi_now <= CONFIRM_RSI_MAX) and (k_now <= OVERBOUGHT)
    return {"ok": ok, "rsi": round(rsi_now, 1), "k": round(k_now, 1), "d": round(float(d.iloc[-1]), 1)}


# ============================== SCAN ==============================

def scan_instrument(inst_id: str):
    """Return dict kandidat kalau lolos 1H trigger + 4H & 1D confirm, else None."""
    df_1h = fetch_candles(inst_id, ENTRY_BAR, CANDLE_LIMIT_ENTRY)
    if len(df_1h) < 60:
        return None
    entry = entry_cross_signal(df_1h)
    if not entry["ok"]:
        return None

    time.sleep(REQUEST_DELAY_SEC)
    df_4h = fetch_candles(inst_id, CONFIRM_BAR_1, CANDLE_LIMIT_CONFIRM)
    if len(df_4h) < 60:
        return None
    confirm_4h = confirm_not_overextended(df_4h)
    if not confirm_4h["ok"]:
        return None

    time.sleep(REQUEST_DELAY_SEC)
    df_1d = fetch_candles(inst_id, CONFIRM_BAR_2, CANDLE_LIMIT_CONFIRM)
    if len(df_1d) < 60:
        return None
    confirm_1d = confirm_not_overextended(df_1d)
    if not confirm_1d["ok"]:
        return None

    price = float(df_1h["close"].iloc[-1])
    fib = fibonacci_levels(df_4h)

    return {
        "inst_id": inst_id,
        "price": price,
        "entry_1h": entry,
        "confirm_4h": confirm_4h,
        "confirm_1d": confirm_1d,
        "fib": fib,
    }


# ============================== TELEGRAM ==============================

def format_candidate_block(c: dict) -> str:
    fib = c["fib"]
    sl_ref = fib["0.786"] if c["price"] > fib["0.786"] else fib["swing_low"]
    tp1 = fib["swing_high"]
    tp2 = fib["ext_1.272"]
    tp3 = fib["ext_1.618"]

    return (
        f"🟢 *{c['inst_id']}*\n"
        f"Harga sekarang: {c['price']:,.4f}\n"
        f"1H StochRSI cross up dari oversold: K {c['entry_1h']['k']} / D {c['entry_1h']['d']} ✅\n"
        f"4H konfirmasi: RSI {c['confirm_4h']['rsi']} | StochRSI K {c['confirm_4h']['k']}/D {c['confirm_4h']['d']}\n"
        f"1D konfirmasi: RSI {c['confirm_1d']['rsi']} | StochRSI K {c['confirm_1d']['k']}/D {c['confirm_1d']['d']}\n"
        f"Referensi Fibonacci (swing 4H):\n"
        f"  Entry ref: {c['price']:,.4f}\n"
        f"  SL ref: {sl_ref:,.4f}\n"
        f"  TP1: {tp1:,.4f} | TP2: {tp2:,.4f} | TP3: {tp3:,.4f}"
    )


def format_message(candidates: list, total_scanned: int) -> list:
    header = (
        f"📊 *OKX Entry Scan* — StochRSI cross (1H) + konfirmasi 4H & 1D\n"
        f"Discan: {total_scanned} pair futures\n"
    )

    if not candidates:
        return [header + "\nTidak ada pair yang lolos ketiga filter saat ini. Cek lagi jam berikutnya."]

    shown = candidates[:MAX_RESULTS_IN_MESSAGE]
    blocks = [header]
    for c in shown:
        blocks.append(format_candidate_block(c))

    if len(candidates) > MAX_RESULTS_IN_MESSAGE:
        blocks.append(f"...dan {len(candidates) - MAX_RESULTS_IN_MESSAGE} pair lain juga lolos.")

    blocks.append("_Level Fibonacci & TP/SL cuma referensi teknikal, tetap riset manual sebelum entry._")

    text = "\n\n".join(blocks)
    max_len = 3800
    if len(text) <= max_len:
        return [text]

    # pecah per blok kalau kepanjangan
    chunks, current = [], header + "\n"
    for block in blocks[1:]:
        if len(current) + len(block) + 2 > max_len:
            chunks.append(current)
            current = ""
        current += block + "\n\n"
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

    candidates = []
    scanned = 0
    for inst_id in symbols:
        try:
            result = scan_instrument(inst_id)
            scanned += 1
            if result:
                candidates.append(result)
                print(f"[CANDIDATE] {inst_id}")
        except Exception as e:
            print(f"[ERROR] {inst_id}: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    print(f"Berhasil discan: {scanned} / {len(symbols)} | Kandidat lolos: {len(candidates)}")

    chunks = format_message(candidates, total_scanned=scanned)
    for chunk in chunks:
        send_telegram(chunk)

    print("Terkirim.")


if __name__ == "__main__":
    main()
