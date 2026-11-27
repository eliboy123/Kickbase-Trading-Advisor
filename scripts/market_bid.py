import argparse
import os
import sys
import requests
from kickbase_api.config import BASE_URL, get_json_with_token
from kickbase_api.league import get_league_id
from kickbase_api.user import login

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--player", required=True, help="Exact surname")
    p.add_argument("--amount", required=True, type=int)
    p.add_argument("--mode", choices=["preview", "place"], default="preview")
    p.add_argument("--confirmation", default="")
    args = p.parse_args()

    if not 500_000 <= args.amount <= 30_000_000:
        p.error("Bid must be between 500k and 30m EUR.")
    if args.mode == "place" and args.confirmation != "PLACE BID":
        p.error("Live bid requires explicit confirmation: PLACE BID")

    email, password = os.getenv("KICK_USER"), os.getenv("KICK_PASS")
    if not email or not password:
        raise RuntimeError("Missing KICK_USER / KICK_PASS secrets.")
    token = login(email, password)
    league_id = get_league_id(token, "kickbase 1")
    if not league_id:
        raise RuntimeError("League not found.")
    listings = get_json_with_token(f"{BASE_URL}/leagues/{league_id}/market", token)
    market = {str(p["i"]): p for p in listings.get("it", []) if p.get("i") is not None}
    found = []
    from urllib.parse import quote
    search = get_json_with_token(
        f"{BASE_URL}/competitions/1/players/search?query={quote(args.player)}", token
    )
    for result in search.get("it", []):
        player_id = result.get("pi", result.get("i"))
        if str(player_id) not in market:
            continue
        info = get_json_with_token(
            f"{BASE_URL}/competitions/1/players/{player_id}", token
        )
        full_name = (str(info.get("fn", "")) + " " + str(info.get("ln", ""))).strip()
        if args.player.casefold().strip() in (str(info.get("ln", "")).casefold(), full_name.casefold()):
            found.append((str(player_id), full_name, info))
    if len(found) != 1:
        raise RuntimeError(f"Found {len(found)} unique matches on the market; no bid submitted.")
    player_id, name, info = found[0]
    market_value = market[player_id].get("mv", info.get("mv"))
    if not isinstance(market_value, (int, float)) or market_value <= 0:
        raise RuntimeError("Market value unavailable; no bid submitted.")
    if args.amount > round(market_value * 1.20):
        raise RuntimeError("Bid exceeds maximum allowed 20% overpay; no bid submitted.")
    print(f"League: kickbase 1 | Player: {name} ({player_id})")
    print(f"MV: {market_value:,} EUR | Bid: {args.amount:,} EUR")
    if args.mode == "preview":
        print("PREVIEW ONLY. No bid submitted.")
        return
    response = requests.post(
        f"{BASE_URL}/leagues/{league_id}/market/{player_id}/offers",
        json={"price": args.amount},
        headers={"Authorization": f"Bearer {token}"},
        timeout=25,
    )
    response.raise_for_status()
    print(f"Bid request accepted: HTTP {response.status_code}. Verify in Kickbase app.")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)
