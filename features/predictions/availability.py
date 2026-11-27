"""Read-only player availability flags from Kickbase player detail responses.

The source is Kickbase's own status code (st), not a medical diagnosis
or a verified prediction that the player will start.
"""
import math
import pandas as pd
from kickbase_api.config import BASE_URL, get_json_with_token

FLAGS = (
    (1, "Verletzt"),
    (2, "Angeschlagen"),
    (4, "Reha/Aufbautraining"),
    (8, "Rotsperre"),
    (16, "Gelb-Rot-Sperre"),
    (32, "Gelbsperre"),
    (64, "Nicht im Team"),
    (128, "Nicht in Liga"),
    (256, "Abwesend"),
)
RESTRICTED = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256


def decode_status(raw):
    """Return German labels, without treating missing/unknown data as fit."""
    if raw is None or isinstance(raw, bool):
        return "Unbekannt"
    try:
        code = int(raw)
    except (ValueError, TypeError, OverflowError):
        return "Unbekannt"
    if code < 0:
        return "Unbekannt"
    if code == 0:
        return "Kein Ausfall gemeldet"
    descriptions = [label for mask, label in FLAGS if code & mask]
    if code & ~RESTRICTED:
        descriptions.append(f"Unbekannter Statuscode {code & ~RESTRICTED}")
    return " / ".join(descriptions) if descriptions else "Unbekannt"


def get_current_availability(token, player_ids, competition_id=1):
    """Fetch fresh status only for current squad/market players, never entire league.

    Individual failures are marked unknown and do not stop the predictions.
    """
    rows = []
    unique_ids = pd.to_numeric(pd.Series(list(player_ids)), errors="coerce").dropna()
    for player_id in unique_ids.astype("int64").drop_duplicates():
        item = {
            "player_id": str(int(player_id)),
            "availability_status": "Unbekannt",
            "availability_note": None,
            "availability_code": None,
        }
        try:
            data = get_json_with_token(
                f"{BASE_URL}/competitions/{competition_id}/players/{int(player_id)}",
                token,
            )
            raw = data.get("st")
            # Some responses only contain a list of individual status flags.
            if raw is None and isinstance(data.get("stl"), list):
                flags = [int(flag) for flag in data["stl"] if str(flag).isdigit()]
                if flags:
                    raw = 0
                    for flag in flags:
                        raw |= flag
            item["availability_status"] = decode_status(raw)
            item["availability_code"] = raw
            note = data.get("stxt")
            item["availability_note"] = str(note).strip() if isinstance(note, str) and note.strip() else None
        except Exception as exc:
            print(f"Availability unavailable for player {int(player_id)}: {type(exc).__name__}")
        rows.append(item)
    result = pd.DataFrame(rows, columns=[
        "player_id", "availability_status", "availability_note", "availability_code"
    ])
    # Match the existing predictions dataframe, where player_id is stored as text.
    result["player_id"] = result["player_id"].astype("string").astype(object)
    return result
