from __future__ import annotations

import argparse
import time

from app.core import get_settings
from app.core.logging import configure_logging, get_logger, log_event
from app.services.job_processing_service import JobProcessingService


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Financial news enrichment worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one queued job and exit.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=settings.worker_poll_interval_seconds,
        help="Seconds to sleep between queue polls in loop mode.",
    )
    parser.add_argument(
        "--idle-log-interval",
        type=float,
        default=settings.worker_idle_log_interval_seconds,
        help="Seconds between idle logs when no queued job is available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging()
    service = JobProcessingService()
    log_event(
        logger,
        20,
        "worker_started",
        poll_interval_seconds=args.poll_interval,
        idle_log_interval_seconds=args.idle_log_interval,
    )

    if args.once:
        result = service.process_next_job()
        log_event(
            logger,
            20,
            "worker_run_once",
            processed=result.processed,
            retry_scheduled=result.retry_scheduled,
            news_id=result.news_id,
            message=result.message,
            analysis_status=result.analysis_status.value if result.analysis_status else None,
        )
        return

    last_idle_log_at = 0.0
    while True:
        result = service.process_next_job()
        if result.processed:
            event_name = "worker_retry_scheduled" if result.retry_scheduled else "worker_processed_job"
            log_event(
                logger,
                20,
                event_name,
                news_id=result.news_id,
                job_id=result.job.job_id if result.job else None,
                retry_scheduled=result.retry_scheduled,
                job_status=result.job.status.value if result.job else None,
                analysis_status=result.analysis_status.value if result.analysis_status else None,
                analysis_outcome=(
                    result.analysis_outcome.value if result.analysis_outcome else None
                ),
                error_code=result.error_code,
                message=result.message,
            )
        else:
            now = time.monotonic()
            if now - last_idle_log_at >= args.idle_log_interval:
                log_event(
                    logger,
                    20,
                    "worker_idle_no_job",
                    poll_interval_seconds=args.poll_interval,
                    message=result.message,
                )
                last_idle_log_at = now
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
