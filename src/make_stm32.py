#!/usr/bin/env python3
"""
make_stm32_csv.py
=================
Convert ECG data into the exact CSV format expected by:
  - classify_stm32.py (Python)
  - CsvLoader::load() (C++/Qt)

OUTPUT FORMAT:
    heure_utc;latitude;longitude;altitude_m
    143022 UTC 22/05/2026;47.729168;7.311351;254.3
    index;raw;filtered
    0;2007;2007
    1;2011;2010
    ...

USAGE:
    # 1) Two input files (one for raw, one for filtered)
    python make_stm32_csv.py --raw raw.csv --filtered filtered.csv --out beat.csv

    # 2) Single file with two columns (raw, filtered) — auto-detected
    python make_stm32_csv.py --in combined.csv --out beat.csv

    # 3) Optional metadata (otherwise defaults are used)
    python make_stm32_csv.py --in combined.csv --out beat.csv \
        --gps "143022 UTC 22/05/2026;47.729168;7.311351;254.3"

    # 4) If values need to be re-scaled to the STM32 baseline (~2000)
    python make_stm32_csv.py --in combined.csv --out beat.csv \
        --add-baseline 2000
"""

import argparse
import csv
import os
import sys


# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_GPS  = "143022 UTC 22/05/2026;47.729168;7.311351;254.3"
DEFAULT_HEAD = "heure_utc;latitude;longitude;altitude_m"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _looks_like_float(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except (ValueError, AttributeError):
        return False


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def _detect_separator(sample_line: str) -> str:
    if ";" in sample_line:
        return ";"
    if "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\t"
    return " "


def read_numeric_column(path: str,
                        column: int = 0,
                        sep: str | None = None,
                        skip_non_numeric: bool = True) -> list[float]:
    """
    Read `path`, return the values in the given column index as floats.
    Skips lines whose chosen column does not parse as a float
    (headers, index columns of the wrong type, etc.).
    If `sep` is None, auto-detects from the first non-empty line.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [l.rstrip("\r\n") for l in f if l.strip() != ""]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    if sep is None:
        sep = _detect_separator(lines[0])

    values = []
    for ln, line in enumerate(lines, start=1):
        # split preserving empty trailing columns
        parts = line.split(sep)
        parts = [_strip_bom(p).strip() for p in parts]

        if column >= len(parts):
            if skip_non_numeric:
                continue
            raise ValueError(
                f"{path}: line {ln} has {len(parts)} columns, "
                f"but column {column} requested"
            )

        val = parts[column]
        if _looks_like_float(val):
            values.append(float(val))
        elif skip_non_numeric:
            continue
        else:
            raise ValueError(f"{path}: line {ln} not numeric: {val!r}")

    if not values:
        raise ValueError(
            f"{path}: no numeric values found in column {column} "
            f"(sep={sep!r})"
        )

    return values


def read_two_columns(path: str,
                     sep: str | None = None) -> tuple[list[float], list[float]]:
    """
    Read a 2-column CSV and return (raw_values, filtered_values).
    Skips header/non-numeric rows. Both lists always have the same length.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [l.rstrip("\r\n") for l in f if l.strip() != ""]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    if sep is None:
        sep = _detect_separator(lines[0])

    raw_vals, filt_vals = [], []
    for ln, line in enumerate(lines, start=1):
        parts = [_strip_bom(p).strip() for p in line.split(sep)]
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        if _looks_like_float(a) and _looks_like_float(b):
            raw_vals.append(float(a))
            filt_vals.append(float(b))

    if not raw_vals:
        raise ValueError(
            f"{path}: could not find any two-column numeric rows (sep={sep!r})"
        )

    return raw_vals, filt_vals


def align_signals(raw: list[float], filt: list[float]) -> tuple[list[float], list[float]]:
    """Trim to the shorter length, warn if unequal."""
    if len(raw) != len(filt):
        n = min(len(raw), len(filt))
        print(f"[warn] raw has {len(raw)} values, filtered has {len(filt)}; "
              f"truncating to {n}", file=sys.stderr)
        raw, filt = raw[:n], filt[:n]
    return raw, filt


def round_int_like(x: float) -> int:
    """Round to nearest int — the STM32 pipeline stores integers."""
    return int(round(x))


# ─── Main conversion ────────────────────────────────────────────────────────
def build_canonical_csv(raw_vals: list[float],
                        filt_vals: list[float],
                        gps_line: str,
                        out_path: str,
                        add_baseline: float | None = None) -> None:
    raw_vals, filt_vals = align_signals(raw_vals, filt_vals)

    if add_baseline is not None:
        raw_vals  = [v + add_baseline for v in raw_vals]
        filt_vals = [v + add_baseline for v in filt_vals]

    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        # GPS header + data
        fo.write(DEFAULT_HEAD + "\n")
        fo.write(gps_line.strip() + "\n")

        # ECG header
        fo.write("index;raw;filtered\n")

        # Data
        for i, (r, fv) in enumerate(zip(raw_vals, filt_vals)):
            fo.write(f"{i};{round_int_like(r)};{round_int_like(fv)}\n")

    print(f"[ok] wrote {out_path}  ({len(raw_vals)} samples)")


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Build the canonical STM32 CSV from raw+filtered data.")
    p.add_argument("--in", dest="infile", default=None,
                   help="Single CSV with two columns (raw, filtered)")
    p.add_argument("--raw", dest="raw_file", default=None,
                   help="CSV with the RAW values (column --raw-col)")
    p.add_argument("--filtered", dest="filt_file", default=None,
                   help="CSV with the FILTERED values (column --filt-col)")
    p.add_argument("--out", dest="out_file", required=True,
                   help="Output path (e.g. beat.csv)")
    p.add_argument("--raw-col", type=int, default=0,
                   help="Column index in --raw file (default 0)")
    p.add_argument("--filt-col", type=int, default=0,
                   help="Column index in --filtered file (default 0)")
    p.add_argument("--sep", default=None,
                   help="Override separator (;, ,, tab, space). "
                        "Default: auto-detect per file.")
    p.add_argument("--gps", default=DEFAULT_GPS,
                   help=f"GPS data line. Default: '{DEFAULT_GPS}'")
    p.add_argument("--add-baseline", type=float, default=None,
                   help="Add a constant to all values (e.g. 2000 to center "
                        "around the STM32 ADC baseline)")
    args = p.parse_args()

    # Guard: need either --in OR (--raw AND --filtered)
    if args.infile is None and (args.raw_file is None or args.filt_file is None):
        p.error("Provide either --in FILE, or both --raw FILE --filtered FILE")

    try:
        if args.infile:
            raw_vals, filt_vals = read_two_columns(args.infile, sep=args.sep)
        else:
            raw_vals  = read_numeric_column(args.raw_file,
                                            column=args.raw_col,
                                            sep=args.sep)
            filt_vals = read_numeric_column(args.filt_file,
                                            column=args.filt_col,
                                            sep=args.sep)

        # Sanity: warn on absurd scale
        allv = raw_vals + filt_vals
        vmin, vmax = min(allv), max(allv)
        if args.add_baseline is None and max(abs(vmin), abs(vmax)) < 20:
            print(f"[warn] values look small (min={vmin}, max={vmax}). "
                  f"Did you mean --add-baseline 2000 ?", file=sys.stderr)

        build_canonical_csv(raw_vals, filt_vals,
                            gps_line=args.gps,
                            out_path=args.out_file,
                            add_baseline=args.add_baseline)

    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()