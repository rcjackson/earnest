#!/usr/bin/env python3
"""
Ingest University of Wyoming atmospheric sounding data.

Usage:
    # Fetch text listing → print CSV to stdout
    python sounding_ingest.py --station 72493 --date 2024-01-15 --hour 0

    # Save to file
    python sounding_ingest.py --station 72493 --date 2024-01-15 --hour 12 -o sounding.csv

    # Fetch native BUFR CSV (includes timestamp + lat/lon per level)
    python sounding_ingest.py --station 72493 --date 2024-01-15 --hour 0 --format csv

Station IDs:
    WMO numbers (e.g. 72493) — preferred for BUFR data.
    Standard sounding hours are 00Z and 12Z; BUFR sites may have more.

Notes:
    Wind speeds in the UW BUFR data are in m/s (column SPED), not knots.
    The --format list output re-exports as CSV using the parsed column names.
    To find a station number, browse: http://weather.uwyo.edu/upperair/sounding.shtml
"""

import argparse
import re
import sys
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_URL = "http://weather.uwyo.edu/wsgi/sounding"


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

def fetch(url: str) -> bytes:
    try:
        with urlopen(url, timeout=30) as resp:
            return resp.read()
    except HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.reason}  ({url})")
    except URLError as e:
        sys.exit(f"Network error: {e.reason}")


def build_url(station: str, dt: datetime, hour: int, fmt: str, src: str) -> str:
    params = urlencode({
        "datetime": f"{dt.strftime('%Y-%m-%d')} {hour:02d}:00:00",
        "id":       station,
        "src":      src,
        "type":     fmt,
    })
    return f"{BASE_URL}?{params}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_station_info(html: str) -> dict:
    """Extract station metadata from HTML headers."""
    info: dict = {}

    m = re.search(r"<H1>Observations for Station\s+(\S+)\s+at\s+(.+?)</H1>", html, re.IGNORECASE)
    if m:
        info["station_number"] = m.group(1)
        info["obs_time"] = m.group(2).strip()

    m = re.search(r"<H3>(.+?)</H3>", html, re.IGNORECASE)
    if m:
        info["station_name"] = m.group(1).strip()

    m = re.search(r"Latitude:\s*([-\d.]+)\s+Longitude:\s*([-\d.]+)", html)
    if m:
        info["latitude"]  = float(m.group(1))
        info["longitude"] = float(m.group(2))

    return info


def parse_text_list(html: str) -> tuple[dict, list[str], list[dict]]:
    """
    Parse a TEXT:LIST response.

    Returns (station_info, column_names, levels).
    """
    if re.search(r"Unable to retrieve|no data", html, re.IGNORECASE):
        sys.exit("No data available for the requested station/time.")

    pre = re.search(r"<PRE>(.*?)</PRE>", html, re.DOTALL)
    if not pre:
        sys.exit("Could not find data block in response.")

    info    = parse_station_info(html)
    columns: list[str] = []
    levels:  list[dict] = []

    lines    = pre.group(1).strip().splitlines()
    in_data  = False
    dash_count = 0

    for line in lines:
        stripped = line.strip()

        # First non-empty line after tag opening is the column header
        if not columns and stripped and not stripped.startswith("-"):
            columns = stripped.split()
            continue

        # Skip units line (second non-dash line)
        if columns and not in_data and not stripped.startswith("-"):
            continue  # units row

        if stripped.startswith("---"):
            dash_count += 1
            if dash_count >= 2:
                in_data = True
            continue

        if not in_data or not stripped:
            continue

        parts = stripped.split()
        if len(parts) != len(columns):
            continue
        try:
            levels.append({col: float(v) for col, v in zip(columns, parts)})
        except ValueError:
            continue

    return info, columns, levels


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_info(info: dict, n_levels: int, station_arg: str) -> None:
    name = info.get("station_name", "N/A")
    num  = info.get("station_number", station_arg)
    obs  = info.get("obs_time", "N/A")
    lat  = info.get("latitude",  "N/A")
    lon  = info.get("longitude", "N/A")
    print(f"# Station : {name} ({num})", file=sys.stderr)
    print(f"# Obs time: {obs}", file=sys.stderr)
    print(f"# Lat/Lon : {lat} / {lon}", file=sys.stderr)
    print(f"# Levels  : {n_levels}", file=sys.stderr)


def write_csv(columns: list[str], levels: list[dict], path: str | None) -> None:
    import csv
    fh = open(path, "w", newline="") if path else sys.stdout
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    writer.writerows(levels)
    if path:
        fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch University of Wyoming atmospheric sounding data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--station", "-s", required=True,
                        help="WMO station number (e.g. 72493)")
    parser.add_argument("--date", "-d", required=True,
                        help="Date in YYYY-MM-DD format")
    parser.add_argument("--hour", "-H", type=int, default=0,
                        help="Sounding hour UTC 0–23 (default: 0)")
    parser.add_argument("--format", "-f", default="list",
                        choices=["list", "csv"],
                        help=(
                            "Output format: list (TEXT:LIST parsed → CSV, default) "
                            "or csv (native BUFR CSV with per-level timestamps and lat/lon)"
                        ))
    parser.add_argument("--src", default="bufr",
                        help="Data source passed to UW API: bufr or UNKNOWN (default: bufr)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        sys.exit("Invalid date — use YYYY-MM-DD.")

    if not 0 <= args.hour <= 23:
        sys.exit("Hour must be between 0 and 23.")

    # Map user-facing format names to UW API type strings
    type_map = {"list": "TEXT:LIST", "csv": "TEXT:CSV"}
    uw_type = type_map[args.format]
    url     = build_url(args.station, dt, args.hour, uw_type, args.src)
    print(f"Fetching: {url}", file=sys.stderr)

    html = fetch(url).decode("utf-8", errors="replace")

    # Check for error before attempting parse
    if re.search(r"Unable to retrieve|no data", html, re.IGNORECASE):
        sys.exit("No data available for the requested station/time.")

    # --- Native CSV pass-through ---
    if args.format == "csv":
        out = args.output or None
        if out:
            with open(out, "w") as f:
                f.write(html)
            print(f"Saved CSV to {out}", file=sys.stderr)
        else:
            sys.stdout.write(html)
        return

    # --- TEXT:LIST → parsed CSV ---
    info, columns, levels = parse_text_list(html)

    if not levels:
        sys.exit("Page returned but no sounding levels found — station may have no data for this time.")

    print_info(info, len(levels), args.station)
    write_csv(columns, levels, args.output)

    if args.output:
        print(f"Saved {len(levels)} levels to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
