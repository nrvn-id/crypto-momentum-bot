import os
import requests

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

def fetch_top_movers(top_n=10):
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "24h",
    }
    resp = requests.get(COINGECKO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filtered = [
        d for d in data
        if d.get("price_change_percentage_24h") is not None
           and d.get("total_volume") is not None
           and d["price_change_percentage_24h"] > 0
    ]

    ranked = sorted(
        filtered,
        key=lambda d: (
                d["price_change_percentage_24h"] * 0.5
                + (d["total_volume"] / 1_000_000) * 0.5
        ),
        reverse=True
    )
    return ranked[:top_n]

def format_message(tokens):
    lines = ["🚀 *Top 10 Momentum Crypto Hari Ini*\n"]
    for i, t in enumerate(tokens, 1):
        symbol = t["symbol"].upper()
        change = t["price_change_percentage_24h"]
        volume_m = t["total_volume"] / 1_000_000
        lines.append(f"{i}. *{symbol}* — {change:+.2f}% | Vol: ${volume_m:,.1f}M")
    lines.append("\n_Data: CoinGecko 24h. Bukan sinyal beli, lakukan riset lanjutan._")
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