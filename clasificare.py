

import os
import json
import shutil
import argparse
from collections import defaultdict
from datetime import datetime

DEFAULT_SRC = os.path.join("../extracted_articles_ro-md", "zdg", "articles")
DEFAULT_DST = "../articles_by_month_md"
MAX_PER_MONTH = 15
START_YEAR = 2019
END_YEAR = 2026


def parse_args():
    parser = argparse.ArgumentParser(description="Clasifică articole JSON pe luni.")
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help=f"Folderul sursă cu fișierele JSON (implicit: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help=f"Folderul destinație (implicit: {DEFAULT_DST})")
    parser.add_argument("--max", type=int, default=MAX_PER_MONTH,
                        help=f"Numărul maxim de articole per lună (implicit: {MAX_PER_MONTH})")
    return parser.parse_args()


def get_published_date(filepath: str):

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        published = data.get("published") or data.get("scraped_at") or data.get("published_backfilled_at")
        if not published:
            return None

        try:
            return datetime.strptime(published[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass
        try:
            return datetime.strptime(published[:10], "%Y-%m-%d")
        except ValueError:
            return None

    except (json.JSONDecodeError, OSError):
        return None


def classify(src_dir: str, dst_dir: str, max_per_month: int):
    if not os.path.isdir(src_dir):
        print(f"[EROARE] Folderul sursă nu există: {src_dir}")
        return

    # Grupăm fișierele pe cheia "YYYY-MM"
    buckets = defaultdict(list)

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(src_dir, fname)
        dt = get_published_date(fpath)

        if dt is None:
            print(f"  [SKIP] Nu s-a putut extrage data din: {fname}")
            continue

        if not (START_YEAR <= dt.year <= END_YEAR):
            continue  # în afara intervalului 2019-2026

        key = dt.strftime("%Y-%m")
        buckets[key].append(fpath)

    if not buckets:
        print("[AVERTISMENT] Niciun articol valid găsit în intervalul specificat.")
        return

    total_copied = 0
    total_skipped = 0

    for month_key in sorted(buckets.keys()):
        files = buckets[month_key]
        selected = files[:max_per_month]   # primele 15 (sortate după nume fișier)
        skipped = len(files) - len(selected)

        month_dir = os.path.join(dst_dir, month_key)
        os.makedirs(month_dir, exist_ok=True)

        for fpath in selected:
            dst_path = os.path.join(month_dir, os.path.basename(fpath))
            shutil.copy2(fpath, dst_path)
            total_copied += 1

        total_skipped += skipped
        print(f"  {month_key}: {len(selected):>3} articole copiate"
              + (f"  (omise: {skipped})" if skipped else ""))

    print(f"\n✔ Total copiate : {total_copied}")
    print(f"  Total omise   : {total_skipped}")
    print(f"  Destinație    : {os.path.abspath(dst_dir)}")


if __name__ == "__main__":
    args = parse_args()
    print(f"Sursă     : {args.src}")
    print(f"Destinație: {args.dst}")
    print(f"Max/lună  : {args.max}\n")
    classify(args.src, args.dst, args.max)