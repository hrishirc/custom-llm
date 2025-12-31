#!/usr/bin/env python3
"""Startup script for Native Training Orchestrator.
"""
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.native_orchestrator import NativeOrchestrator

import fcntl

def acquire_lock():
    """Ensure only one instance runs."""
    lock_file = "/tmp/native_orchestrator.lock"
    fp = open(lock_file, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except IOError:
        return None

def main():
    # Try to acquire lock
    lock_fp = acquire_lock()
    if not lock_fp:
        print("Error: Another instance of the orchestrator is already running.")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/native_orchestrator.log")
        ]
    )
    
    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = NativeOrchestrator(
        db_path="data/training_state.db",
        data_dir="data",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        phases=["1", "2", "2b"]
    )
    
    # Run
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\nOrchestrator stopped by user.")
    except Exception as e:
        print(f"\nOrchestrator failed: {e}")
        sys.exit(1)
    finally:
        # Release lock (OS does this automatically on exit, but good practice)
        pass

if __name__ == "__main__":
    main()
