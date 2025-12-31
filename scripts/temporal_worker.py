#!/usr/bin/env python3
"""Temporal worker for LLM training activities.

Connects to Temporal server and executes training activities.
Handles long-running training with heartbeats.

Logging:
- Console: INFO level by default, DEBUG with --verbose
- File: logs/worker.log (DEBUG level, rotating)
- SQLite: data/training_state.db event_log table

Usage:
    # Start worker (connects to localhost:7233)
    python scripts/temporal_worker.py
    
    # Custom server address
    python scripts/temporal_worker.py --server localhost:7233
    
    # Verbose logging
    python scripts/temporal_worker.py --verbose
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.training_workflow import LLMTrainingWorkflow
from src.workflows.activities import (
    load_state_from_db,
    download_dataset,
    clean_dataset,
    train_tokenizer,
    tokenize_data,
    train_phase,
    evaluate_model,
)


def setup_logging(verbose: bool = False, log_dir: str = "logs/worker"):
    """Configure logging for worker and all modules.
    
    Args:
        verbose: If True, use DEBUG level for console
        log_dir: Directory for log files
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (rotating, DEBUG level)
    log_file = log_path / "worker.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger("temporalio").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"Worker started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info("=" * 60)
    
    return logger


async def main(server_address: str, verbose: bool):
    logger = setup_logging(verbose=verbose)
    
    logger.info(f"Connecting to Temporal server at {server_address}...")
    
    try:
        client = await Client.connect(server_address)
        logger.info("Connected to Temporal server successfully!")
    except Exception as e:
        logger.error(f"Failed to connect to Temporal: {e}")
        print("\nMake sure Temporal server is running:")
        print("  temporal server start-dev")
        return 1
    
    print("\n" + "=" * 60)
    print("  LLM TRAINING WORKER")
    print("=" * 60)
    print(f"  Server:     {server_address}")
    print(f"  Task Queue: llm-training")
    print(f"  Log File:   logs/worker.log")
    print(f"  SQLite Log: sqlite3 data/training_state.db 'SELECT * FROM event_log'")
    print("=" * 60)
    print("\nPress Ctrl+C to stop (training will resume when worker restarts)")
    print("-" * 60)
    
    # Create worker with all activities
    worker = Worker(
        client,
        task_queue="llm-training",
        workflows=[LLMTrainingWorkflow],
        activities=[
            load_state_from_db,
            download_dataset,
            clean_dataset,
            train_tokenizer,
            tokenize_data,
            train_phase,
            evaluate_model,
        ],
    )
    
    try:
        logger.info("Worker running, waiting for tasks...")
        await worker.run()
    except asyncio.CancelledError:
        logger.info("Worker stopped gracefully")
        print("\nWorker stopped gracefully")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal worker for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Logging outputs:
  - Console: INFO level (DEBUG with --verbose)
  - File: logs/worker.log (DEBUG, rotating 10MB × 5)
  - SQLite: data/training_state.db event_log table
        """,
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:7233",
        help="Temporal server address"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) console logging"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = asyncio.run(main(args.server, args.verbose))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nWorker interrupted by user")
        sys.exit(0)
