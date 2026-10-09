from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import add_period_args, add_source_arg, resolve_sources
from src.config import load_config
from src.scrape import get_scraper
from src.scrape.checkpoint import ScrapeStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def run_scraper(
    source: str,
    start_period: str,
    end_period: str,
    checkpoint_interval: int,
    rediscover_urls: bool,
) -> ScrapeStats:
    config = load_config()
    scraper = get_scraper(source, config)

    return scraper.run(
        start_period,
        end_period,
        checkpoint_every=checkpoint_interval,
        force_rediscover=rediscover_urls,
    )


def print_summary(source: str, stats: ScrapeStats) -> None:
    logger.info(
        "[%s] Finished | discovered=%d | new=%d | cached=%d | failed=%d",
        source,
        stats.discovered,
        stats.saved_new,
        stats.skipped_cached,
        stats.failed_total,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Economic news scraper"
    )

    add_source_arg(parser)
    add_period_args(parser)

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save progress every N articles",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel execution",
    )

    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Force URL discovery phase",
    )

    return parser


def execute_sequential(
    sources: list[str],
    start_period: str,
    end_period: str,
    checkpoint_every: int,
    rediscover: bool,
) -> None:
    for source in sources:
        stats = run_scraper(
            source,
            start_period,
            end_period,
            checkpoint_every,
            rediscover,
        )
        print_summary(source, stats)


def execute_parallel(
    sources: list[str],
    start_period: str,
    end_period: str,
    checkpoint_every: int,
    rediscover: bool,
    workers: int,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {
            executor.submit(
                run_scraper,
                source,
                start_period,
                end_period,
                checkpoint_every,
                rediscover,
            ): source
            for source in sources
        }

        for future in as_completed(tasks):
            source = tasks[future]

            try:
                stats = future.result()
                print_summary(source, stats)
            except Exception:
                logger.exception("[%s] Execution failed", source)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()

    start_period = (
        args.from_period
        or config["project"]["start_period"]
    )
    end_period = (
        args.to_period
        or config["project"]["end_period"]
    )

    sources = resolve_sources(args)

    worker_count = (
        1
        if args.sequential
        else min(args.workers, len(sources))
    )

    logger.info(
        "Starting scrape | sources=%s | period=%s..%s | workers=%d",
        sources,
        start_period,
        end_period,
        worker_count,
    )

    logger.info(
        "Logs: data/scrape_logs/<source>.log | "
        "State: data/scrape_state/<source>.json"
    )

    if worker_count == 1:
        execute_sequential(
            sources,
            start_period,
            end_period,
            args.checkpoint_every,
            args.rediscover,
        )
    else:
        execute_parallel(
            sources,
            start_period,
            end_period,
            args.checkpoint_every,
            args.rediscover,
            worker_count,
        )


if __name__ == "__main__":
    main()