#!/usr/bin/env python3
"""Download Dukascopy XAUUSD candles with dukascopy-node.

One-minute bid data is the source series. The loader resamples it to the
chart timeframe, so 5m, 15m, and 1h do not need their own downloads.

    python scripts/download_data.py
    python scripts/download_data.py --from 2024-01-01 --to 2024-02-01
    python scripts/download_data.py --timeframes m1 m5 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def parse_from(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def year_chunks(start: date, end_token: str) -> list[tuple[date, str]]:
    """Return (start, dukascopy --date-to) pairs. The end date is exclusive."""
    today = date.today()
    if end_token in {"now", "today", ""}:
        end = today
        last_is_now = True
    else:
        end = parse_from(end_token)
        last_is_now = False
    chunks = []
    year = start.year
    while date(year, 1, 1) < end or (last_is_now and year == end.year and start <= end):
        chunk_start = max(start, date(year, 1, 1))
        next_year = date(year + 1, 1, 1)
        if next_year <= end and not (last_is_now and year == end.year):
            chunks.append((chunk_start, next_year.isoformat()))
        else:
            token = "now" if last_is_now and year >= end.year else end.isoformat()
            if chunk_start < end or token == "now":
                chunks.append((chunk_start, token))
            break
        year += 1
        if year > end.year + 1:
            break
    return chunks


def output_name(instrument: str, timeframe: str, price: str, start: date, end_token: str) -> str:
    return f"{instrument}-{timeframe}-{price}-{start.isoformat()}-{end_token}"


def download_chunk(
    *,
    instrument: str,
    timeframe: str,
    price: str,
    start: date,
    end_token: str,
    directory: Path,
    include_flats: bool,
    force: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = output_name(instrument, timeframe, price, start, end_token)
    dest = directory / f"{name}.csv"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip {dest.name}")
        return

    cmd = [
        "npx",
        "--yes",
        "dukascopy-node",
        "-i",
        instrument,
        "-from",
        start.isoformat(),
        "-to",
        end_token,
        "-t",
        timeframe,
        "-p",
        price,
        "-v",
        "-vu",
        "units",
        "-f",
        "csv",
        "-df",
        "iso",
        "-tz",
        "UTC",
        "-dir",
        str(directory),
        "-fn",
        name,
        "-bs",
        "24",
        "-bp",
        "200",
        "-r",
        "4",
        "-rp",
        "1000",
        "-re",
        "-fr",
    ]
    if include_flats:
        cmd.append("-fl")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dukascopy XAUUSD candles")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--from", dest="date_from", help="yyyy-mm-dd, overrides config")
    parser.add_argument("--to", dest="date_to", help="yyyy-mm-dd or now, overrides config")
    parser.add_argument("--timeframes", nargs="+", help="example: m1 m5 m15 h1")
    parser.add_argument("--force", action="store_true", help="redownload existing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    start = parse_from(args.date_from or data["date_from"])
    end_token = args.date_to or str(data.get("date_to", "now"))
    timeframes = args.timeframes or list(data.get("download_timeframes") or ["m1"])
    raw_dir = ROOT / data.get("raw_dir", "data/raw")
    instrument = data.get("instrument", "xauusd")
    price = data.get("price_type", "bid")
    include_flats = bool(data.get("include_flats", True))

    chunks = year_chunks(start, end_token)
    if not chunks:
        sys.exit("Nothing to download for that date range.")

    for timeframe in timeframes:
        for chunk_start, chunk_end in chunks:
            download_chunk(
                instrument=instrument,
                timeframe=timeframe,
                price=price,
                start=chunk_start,
                end_token=chunk_end,
                directory=raw_dir,
                include_flats=include_flats,
                force=args.force,
            )
    print(f"saved CSVs in {raw_dir}")


if __name__ == "__main__":
    main()
