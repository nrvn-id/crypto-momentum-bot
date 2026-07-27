import os
import requests

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

BINANCE_URL = "https://api.binance.com/api/v3/ticker/24hr"

def fetch_top_movers(top_n=10):
    resp = requests.get(BINANCE_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Filter hanya pair USDT, buang stablecoin & leveraged token
    excluded = ("UP", "DOWN", "BULL", "BEAR")
    filtered = [
        d for d in data
        if d["symbol"].endswith("USDT")
           and not d["symbol"].startswith(("USDC", "USDT", "FDUSD", "TUSD"))
           and not any(x in d["symbol"] for x in excluded)
    ]

    for d in filtered:
        d["priceChangePercent"] = float(d["priceChangePercent"])
        d["quoteVolume"] = float(d["quoteVolume"])

    # Ranking sederhana: kombinasi volume tinggi + kenaikan harga positif
    movers = [d for d in filtered if d["priceChangePercent"] > 0]
    ranked = sorted(
        movers,
        key=lambda d: (d["priceChangePercent"] * 0.5 + (d["quoteVolume"] / 1_000_000) * 0.5),
        reverse=True
    )
    return ranked[:top_n]

def format_message(tokens):
    lines = ["🚀 *Top 10 Momentum Crypto Hari Ini*\n"]
    for i, t in enumerate(tokens, 1):
        symbol = t["symbol"].replace("USDT", "")
        change = t["priceChangePercent"]
        volume_m = t["quoteVolume"] / 1_000_000
        lines.append(f"{i}. *{symbol}* — {change:+.2f}% | Vol: ${volume_m:,.1f}M")
    lines.append("\n_Data: Binance 24h. Bukan sinyal beli, lakukan riset lanjutan._")
    return "\n".join(lines)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()

if __name__ == "__main__":
    top_tokens = fetch_top_movers(10)
    message = format_message(top_tokens)
    send_telegram(message)
    print("Terkirim.")