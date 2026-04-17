"""
Wait for the connectivity-search-backend API to become healthy.

This script polls the /v1/nodes/ endpoint until it returns HTTP 200 or times out.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import httpx


def wait_for_api(
    api_base: str = "http://localhost:8015",
    timeout_minutes: int = 60,
    poll_seconds: int = 10,
) -> bool:
    """Wait for the API to respond with HTTP 200."""
    health_url = f"{api_base}/v1/nodes/"
    deadline = time.time() + (timeout_minutes * 60)
    attempt = 0

    print(f"Waiting for API at {health_url} (timeout {timeout_minutes} min)...")
    while time.time() < deadline:
        attempt += 1
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(health_url)
                if response.status_code == 200:
                    print("API is healthy.")
                    return True
        except httpx.HTTPError:
            pass

        if attempt % 6 == 0:
            elapsed = int((timeout_minutes * 60) - (deadline - time.time()))
            print(f"  Still waiting... ({elapsed}s elapsed)")

        time.sleep(poll_seconds)

    print("ERROR: API did not become healthy before timeout.")
    return False


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    argv = argv or sys.argv[1:]
    timeout_minutes = 60
    if argv:
        try:
            timeout_minutes = int(argv[0])
        except ValueError:
            print("Usage: python scripts/wait_for_api.py [timeout_minutes]")
            return 2

    ok = wait_for_api(timeout_minutes=timeout_minutes)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
