import requests
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

BASE_URL = "https://api.kickbase.com/v4"

def get_json_with_token(url, token):
    """Fetch JSON data with retry handling for rate limits."""
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(5):
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            delay = min(60 * (2 ** attempt), 300)

            if retry_after:
                try:
                    delay = min(300, max(1, int(retry_after)))
                except ValueError:
                    try:
                        retry_time = parsedate_to_datetime(retry_after)
                        delay = min(300, max(1, (
                            retry_time - datetime.now(timezone.utc)
                        ).total_seconds()))
                    except (ValueError, TypeError):
                        pass

            print(f"Rate limit reached. Retrying in {delay:.0f}s...")
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("Kickbase API rate limit exceeded after 5 attempts")
