"""List exactly one owned Kickbase player on the transfer market.

Preview mode is read-only. The confirmed live mode sends one POST without retries.
"""
import argparse
import os
import sys
import unicodedata

import requests

from kickbase_api.config import BASE_URL, get_json_with_token
from kickbase_api.league import get_league_id
from kickbase_api.user import get_players_in_squad, login


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def identify_player(squad, requested):
    matches = []
    for item in squad:
        identifier = item.get("i")
        if identifier is None:
            continue
        first = item.get("fn") or ""
        last = item.get("ln") or ""
        full = f"{first} {last}".strip()
        if (normalize(requested) == normalize(str(identifier))
                or normalize(requested) == normalize(full)
                or (last and normalize(requested) == normalize(last))):
            matches.append((identifier, full, item))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one player in your squad matching {requested!r}; "
            f"found {len(matches)}. No player listed. "
            "Use the exact full name or player ID."
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, help="Exact player name or ID")
    parser.add_argument("--mode", choices=("preview", "list"), default="preview")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()

    if args.mode == "list" and args.confirmation != "LIST PLAYER":
        parser.error("Live listing requires confirmation LIST PLAYER")

    email = os.getenv("KICK_USER")
    password = os.getenv("KICK_PASS")
    if not email or not password:
        raise RuntimeError("Missing GitHub secrets KICK_USER or KICK_PASS.")
    token = login(email, password)

    # Do not silently list players in a different league.
    league_info = get_json_with_token(f"{BASE_URL}/leagues/selection", token)
    matching = [
        entry for entry in league_info.get("it", [])
        if entry.get("n") == "kickbase 1"
    ]
    if len(matching) != 1:
        raise RuntimeError("League 'kickbase 1' not uniquely found. No player listed.")
    league_id = matching[0]["i"]

    data = get_players_in_squad(token, league_id)
    player_id, full_name, item = identify_player(data.get("it", []), args.player)

    # Prevent duplicate listings if already present in active market data.
    market = get_json_with_token(f"{BASE_URL}/leagues/{league_id}/market", token)
    if any(str(x.get("i")) == str(player_id) for x in market.get("it", [])):
        raise RuntimeError(
            f"{full_name} is already on the market. No additional listing sent."
        )

    print(f"League: kickbase 1 | Player: {full_name} | ID: {player_id}")
    market_value = item.get("mv")
    if isinstance(market_value, (int, float)):
        print(f"Squad-reported market value: {market_value:,.0f} EUR")
    print("Action: list this owned player on Kickbase transfer market.")
    print("This does not accept a buyer's offer or instantly sell the player.")
    if args.mode == "preview":
        print("PREVIEW ONLY: No listing submitted.")
        return

    # Do not retry POST if its outcome is unknown; avoid accidental duplicates.
    response = requests.post(
        f"{BASE_URL}/leagues/{league_id}/market/{player_id}/sell",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    print(
        f"Listing request accepted by API (HTTP {response.status_code}). "
        "Check the Kickbase app to verify market status."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Listing helper stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
