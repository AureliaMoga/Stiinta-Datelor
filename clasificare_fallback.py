import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

SOURCES = [
    ("hotnews", os.path.join("../extracted_articles_ro-md", "hotnews", "articles")),
    ("digi24", os.path.join("../extracted_articles_ro-md", "digi24", "articles")),
]

DEFAULT_DST = "../articles_by_month_ro"
MAX_PER_MONTH = 15
START_YEAR = 2019
END_YEAR = 2026


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Group JSON articles by month from multiple sources"
    )

    parser.add_argument(
        "--dst",
        default=DEFAULT_DST,
        help=f"Destination directory (default: {DEFAULT_DST})",
    )

    parser.add_argument(
        "--max",
        type=int,
        default=MAX_PER_MONTH,
        help=f"Maximum articles per month (default: {MAX_PER_MONTH})",
    )

    return parser.parse_args()


# Extract publication date from JSON file
def get_published_date(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        published = (
            data.get("published")
            or data.get("scraped_at")
            or data.get("published_backfilled_at")
        )

        if not published:
            return None

        try:
            return datetime.strptime(
                published[:19],
                "%Y-%m-%dT%H:%M:%S",
            )
        except ValueError:
            pass

        try:
            return datetime.strptime(
                published[:10],
                "%Y-%m-%d",
            )
        except ValueError:
            return None

    except (json.JSONDecodeError, OSError):
        return None


# Load valid JSON files grouped by month
def load_source(name: str, src_dir: str):
    buckets = defaultdict(list)

    if not os.path.isdir(src_dir):
        print(f"  [WARNING] Source directory not found: {src_dir}")
        return buckets

    for filename in sorted(os.listdir(src_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(src_dir, filename)
        published_date = get_published_date(filepath)

        if published_date is None:
            continue

        if not (START_YEAR <= published_date.year <= END_YEAR):
            continue

        month_key = published_date.strftime("%Y-%m")
        buckets[month_key].append(filepath)

    total_articles = sum(len(files) for files in buckets.values())

    print(
        f"  [{name}] {total_articles} valid articles found across "
        f"{len(buckets)} months"
    )

    return buckets


# Copy articles into monthly folders
def classify(dst_dir: str, max_per_month: int):
    # Load source data
    print("\n── Indexing sources ──")

    all_buckets = {}

    for source_name, source_dir in SOURCES:
        all_buckets[source_name] = load_source(
            source_name,
            source_dir,
        )

    # Collect available months
    all_months = set()

    for buckets in all_buckets.values():
        all_months.update(buckets.keys())

    if not all_months:
        print("\n[ERROR] No valid articles found")
        return

    # Process monthly selection
    print("\n── Classification ──")

    monthly_stats = []

    for month_key in sorted(all_months):
        selected_files = []
        source_summary = []

        for source_name, _ in SOURCES:
            if len(selected_files) >= max_per_month:
                break

            remaining = max_per_month - len(selected_files)

            available_files = all_buckets[source_name].get(
                month_key,
                [],
            )

            chosen_files = available_files[:remaining]

            if chosen_files:
                selected_files.extend(chosen_files)
                source_summary.append(
                    f"{source_name}:{len(chosen_files)}"
                )

        if not selected_files:
            continue

        month_dir = os.path.join(dst_dir, month_key)
        os.makedirs(month_dir, exist_ok=True)

        for filepath in selected_files:
            destination = os.path.join(
                month_dir,
                os.path.basename(filepath),
            )
            shutil.copy2(filepath, destination)

        summary = "  +  ".join(source_summary)

        print(
            f"  {month_key}: {len(selected_files):>3} articles "
            f"({summary})"
        )

        monthly_stats.append(len(selected_files))

    print(f"\n✔ Total months processed : {len(monthly_stats)}")
    print(f"  Total articles copied : {sum(monthly_stats)}")
    print(f"  Destination           : {os.path.abspath(dst_dir)}")


if __name__ == "__main__":
    args = parse_args()

    print(f"Destination : {args.dst}")
    print(f"Max/month   : {args.max}")

    classify(args.dst, args.max)